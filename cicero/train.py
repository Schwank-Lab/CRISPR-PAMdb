import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset
from transformers import (
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    AutoModel,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    logging as hf_logging
)

import os
import pandas as pd
import argparse
import logging
from datetime import datetime

import utils  

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
hf_logging.set_verbosity_error()

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"


def compute_information_matrix(P: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute information matrix for a probability matrix P (e.g., softmax output).
    Each position is weighted by its KL divergence from a uniform distribution.
    """
    Q = 0.25  # uniform background
    P_clamped = torch.clamp(P, min=eps)
    pos_info = (P_clamped * torch.log2(P_clamped / Q)).sum(-1, keepdim=True)
    return P_clamped * pos_info


def augmented_cosine_similarity(logits, targets_logos, is_info=False, pam_lengths=None):
    def compute_information_matrix(P, eps=1e-8):
        Q = 0.25
        P_clamped = torch.clamp(P, min=eps)
        pos_info = (P_clamped * torch.log2(P_clamped / Q)).sum(-1, keepdim=True)
        return P_clamped * pos_info

    def augment_information_matrix(I, I_other):
        N_column = torch.clamp((I_other - I).sum(-1), 0)
        return torch.cat([I, N_column.unsqueeze(2)], dim=2)

    if is_info:
        I1 = logits.reshape(-1, 10, 4)
    else:
        I1 = compute_information_matrix(F.softmax(logits.reshape(-1, 10, 4), -1))

    I2 = targets_logos.reshape(-1, 10, 4)

    J1 = augment_information_matrix(I1, I2).reshape(logits.shape[0], -1)
    J2 = augment_information_matrix(I2, I1).reshape(logits.shape[0], -1)

    if pam_lengths is None:
        return F.cosine_similarity(J1, J2, dim=-1)
    else:
        pos_ids = torch.arange(10, device=logits.device).unsqueeze(0).expand(logits.shape[0], -1)
        length_mask = (pos_ids < pam_lengths.unsqueeze(1)).unsqueeze(2)
        mask_full = length_mask.expand(-1, -1, 5).reshape(-1, 50)
        J1_masked = J1 * mask_full
        J2_masked = J2 * mask_full
        return F.cosine_similarity(J1_masked, J2_masked, dim=-1)
    

def information_mat_to_probability_mat(logos: torch.Tensor, background: float = 0.25) -> torch.Tensor:
    """
    Converts an information matrix (e.g. output of compute_information_matrix) back to a 
    normalized probability matrix.

    If a row is all zeros (i.e., no information), it is replaced with a uniform background distribution.

    Args:
        logos (torch.Tensor): Tensor of shape (..., 4) representing information values per base.
        background (float): Probability value to use for uniform rows (default: 0.25).

    Returns:
        torch.Tensor: Normalized probability matrix of same shape as input.
    """
    bg = background * torch.ones_like(logos)
    row_sum = logos.sum(dim=-1)
    # Identify zero rows and replace them with uniform background
    zero_rows = torch.isclose(row_sum, torch.zeros_like(row_sum))
    logos = torch.where(zero_rows.unsqueeze(-1), bg, logos)

    # Normalize to get valid probability distributions
    return logos / logos.sum(dim=-1, keepdim=True)


class CustomPamPredictTrainer(Trainer):
    """
    HuggingFace Trainer subclass with:
    - Separate learning rates for base model and MLP head
    - Custom loss: CE + negative augmented cosine similarity
    """
    def __init__(self, *args, base_lr: float, mlp_lr: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_lr = base_lr
        self.mlp_lr = mlp_lr

    def create_optimizer(self):
        optimizer_grouped_parameters = [
            {"params": self.model.base_model.parameters(), "lr": self.base_lr},
            {"params": self.model.mlp_head.parameters(), "lr": self.mlp_lr},
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters)
        return self.optimizer

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        logits = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])

        targets_logos = inputs['pam_logos']
        logits_flat = logits.view(-1, targets_logos.shape[-1])
        target_prob_flat = information_mat_to_probability_mat(targets_logos).view(-1, targets_logos.shape[-1])

        # Cross-entropy with soft target logos
        log_preds = torch.log_softmax(logits_flat, dim=-1)
        loss_ce = -(log_preds * target_prob_flat).sum(dim=-1).mean()

        # Augmented cosine similarity
        loss_aug_cos = -augmented_cosine_similarity(logits, targets_logos).mean()

        loss = loss_ce + loss_aug_cos

        if return_outputs:
            return loss, {
                "logits": logits,
                "pam_logos": inputs["pam_logos"],
                "pam_converted": inputs["pam_converted"]
            }

        return loss
    

class ESMWithMLPHead(nn.Module):
    """Protein language model with an MLP head for predicting 10x4 PAM logits."""
    def __init__(self, base_model: nn.Module, out_dim: int, hidden_dim: int, dropout_prob: float = 0.0):
        super().__init__()
        self.base_model = base_model
        esm_hidden_size = self.base_model.config.hidden_size
        self.mlp_head = nn.Sequential(
            nn.Linear(esm_hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None, **kwargs) -> torch.Tensor:
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return self.mlp_head(cls_embedding)


class Cas9PAMDataset(Dataset):
    """Custom Dataset for Cas9 PAM sequence prediction."""
    def __init__(self, df: pd.DataFrame, tokenizer, use_PID: bool = False):
        self.df = df
        self.tokenizer = tokenizer
        self.use_PID = use_PID

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        seq = row['sequence']
        if self.use_PID and 'pid_start' in row and row['pid_start'] < 4000.:
            seq = seq[int(row['pid_start']):int(row['pid_end'])]

        inputs = self.tokenizer(seq, truncation=False, padding=False, return_tensors="pt")
        return {
            "cas9_id": row['id'],
            "input_ids": inputs['input_ids'].squeeze(0),
            "attention_mask": inputs['attention_mask'].squeeze(0),
            "pam_length": torch.tensor(len(row['pam']), dtype=torch.int),
            "pam_converted": torch.from_numpy(row['pam_converted']),
            "pam_logos": torch.from_numpy(row['pam_logits']),
            "cluster_id": torch.tensor(row['cluster_id'])
        }


def main():
    parser = argparse.ArgumentParser(description="Train Cas9 PAM predictor using ESM2 model.")
    parser.add_argument("--esm_model", type=str, default="esm2_t6_8M_UR50D", help="HuggingFace model checkpoint (e.g., esm2_t33_650M_UR50D)")
    parser.add_argument("--data_dir", type=str, default="data/", help="Path to training data directory")
    parser.add_argument("--reuse_experiment", type=str, default="exp0000", help="Name of existing experiment folder to reuse")
    parser.add_argument("--fold", type=int, default=0, help="Fold number (used to load specific train/val split)")
    parser.add_argument("--use_PID", type=utils.str2bool, default=False, help="Use PID subrange if available")
    parser.add_argument("--hidden_dim", type=int, default=1280, help="Hidden dimension of MLP head")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dropout_prob", type=float, default=0.2, help="Dropout probability in MLP head")
    training_args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    logger.info(f"Train PamPredict on Fold {training_args.fold}, using device: {device}")
    utils.seed_all(training_args.seed)

    train_df = pd.DataFrame(utils.load_h5(os.path.join(training_args.data_dir, f"train_{training_args.fold}.h5")))
    val_df = pd.DataFrame(utils.load_h5(os.path.join(training_args.data_dir, f"val_{training_args.fold}.h5")))
    logger.info(f"Loaded {len(train_df)} training samples and {len(val_df)} validation samples.")

    model_checkpoint = os.path.join("facebook", training_args.esm_model)
    base_model = AutoModel.from_pretrained(model_checkpoint, force_download=True)
    model = ESMWithMLPHead(base_model, out_dim=40, hidden_dim=training_args.hidden_dim, dropout_prob=training_args.dropout_prob).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, force_download=True)

    train_dataset = Cas9PAMDataset(train_df, tokenizer, use_PID=training_args.use_PID)
    val_dataset = Cas9PAMDataset(val_df, tokenizer, use_PID=training_args.use_PID)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    output_dir = os.path.join(
        "out",
        f"{training_args.esm_model}-pam_predict-{training_args.reuse_experiment or utils.get_next_experiment_dir('exp')}",
        f"run_{training_args.fold}"
    )

    hf_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=100,
        save_steps=100,
        logging_steps=100,
        label_names=["pam_logos", "pam_converted", "pam_length"],
        per_device_train_batch_size=1,
        per_device_eval_batch_size=8,
        num_train_epochs=15,
        gradient_accumulation_steps=4,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    trainer = CustomPamPredictTrainer(
        base_lr=1e-4,
        mlp_lr=1e-4,
        model=model,
        args=hf_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )
    trainer.add_callback(utils.LossPlotterCallback(save_dir=output_dir, save_interval=100))
    trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=5000))
    utils.save_config(vars(training_args), output_dir)

    trainer.train()
    logger.info("Training complete.")

if __name__ == "__main__":
    main()
