import os
import argparse
import logging
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from safetensors.torch import safe_open
from transformers import (
    AutoModel,
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)

from train import (
    ESMWithMLPHead,
    Cas9PAMDataset,
    augmented_cosine_similarity,
)
import utils
import test

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ESMWithMLPHeadAndConfidence(nn.Module):
    """
    Extended ESM-based model for PAM prediction with an additional confidence head.
    The base ESM and PAM prediction head are frozen during training.
    """

    def __init__(self, base_model, out_dim, hidden_dim, dropout_prob=0.0, conf_hidden_size=None):
        super(ESMWithMLPHeadAndConfidence, self).__init__()
        self.base_model = base_model
        esm_hidden_size = self.base_model.config.hidden_size

        self.pam_head = nn.Sequential(
            nn.Linear(esm_hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, out_dim)
        )

        if conf_hidden_size is None:
            conf_hidden_size = esm_hidden_size

        self.confidence_head = nn.Sequential(
            nn.Linear(esm_hidden_size, conf_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(conf_hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, input_ids, attention_mask=None, **kwargs):
        outputs = self.base_model(input_ids, attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        pam_logits = self.pam_head(cls_embedding)
        confidence_score = self.confidence_head(cls_embedding)
        return pam_logits, confidence_score


class CustomConfidenceTrainer(Trainer):
    """
    Custom Trainer to optimize only the confidence head of the model.
    """

    def __init__(self, *args, lr, **kwargs):
        super().__init__(*args, **kwargs)
        self.lr = lr

    def create_optimizer(self):
        optimizer_grouped_parameters = [
            {"params": self.model.confidence_head.parameters(), "lr": self.lr}
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters)
        return self.optimizer

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        pam_logits, confidence_score = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask']
        )
        targets_logos = inputs['pam_logos']
        targets_indices = inputs['pam_converted']
        target_similarity = augmented_cosine_similarity(pam_logits, targets_logos, is_info=False)
        loss_conf = F.mse_loss(confidence_score.squeeze(), target_similarity)

        if return_outputs:
            return loss_conf, {
                "confidence_score": confidence_score,
                "target_similarity": target_similarity
            }
        return loss_conf


def main():
    parser = argparse.ArgumentParser(description="Train confidence head for PAM predictor.")
    parser.add_argument("--esm_model", default="esm2_t6_8M_UR50D", type=str, help="Name of the ESM model")
    parser.add_argument("--data_dir", default="data/", type=str, help="Directory containing input data")
    parser.add_argument("--fold", default=0, type=int, help="Cross-validation fold number")
    parser.add_argument("--seed", default=42, type=int, help="Random seed for reproducibility")
    parser.add_argument("--exp_dir", default="exp0000", type=str, help="Directory for pretrained PAM model experiment")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    utils.seed_all(args.seed)

    load_dir = os.path.join("out", f"{args.esm_model}-pam_predict-{args.exp_dir}", f"run_{args.fold}")
    config_orig = utils.load_config(os.path.join(load_dir, 'config.json'))
    use_PID = config_orig.get('use_PID', False)

    config = {
        "out_dim": 40,
        "hidden_dim": config_orig['hidden_dim'],
    }

    dropout_prob = 0.2
    model_checkpoint = os.path.join("facebook", args.esm_model)

    out_dim = config['out_dim']
    hidden_dim = config['hidden_dim']
    base_model = AutoModel.from_pretrained(model_checkpoint)
    model = ESMWithMLPHead(base_model, out_dim=out_dim, hidden_dim=hidden_dim, dropout_prob=dropout_prob)

    checkpoint = test.get_latest_checkpoint(load_dir, 'first')
    logger.info(f"Loading checkpoint from: {checkpoint}")

    model_path = "model.safetensors"
    safetensors_path = os.path.join(checkpoint, model_path)

    with safe_open(safetensors_path, framework="pt", device="cpu") as f:
        state_dict = {key: f.get_tensor(key) for key in f.keys()}
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()

    confidence_model = ESMWithMLPHeadAndConfidence(base_model, out_dim=out_dim, hidden_dim=hidden_dim, dropout_prob=dropout_prob)
    confidence_model.base_model.load_state_dict(model.base_model.state_dict())
    confidence_model.pam_head.load_state_dict(model.mlp_head.state_dict())
    confidence_model = confidence_model.to(device)
    confidence_model.eval()

    for param in confidence_model.base_model.parameters():
        param.requires_grad = False
    for param in confidence_model.pam_head.parameters():
        param.requires_grad = False

    total_params = sum(p.numel() for p in confidence_model.parameters())
    trainable_params = sum(p.numel() for p in confidence_model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params}, Trainable parameters: {trainable_params}")

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, force_download=True)    
    train_df = pd.DataFrame(utils.load_h5(os.path.join(args.data_dir, f"train_{args.fold}.h5")))
    val_df = pd.DataFrame(utils.load_h5(os.path.join(args.data_dir, f"val_{args.fold}.h5")))
    logger.info(f"Training samples: {len(train_df)}")
    logger.info(f"Validation samples: {len(val_df)}")

    train_dataset = Cas9PAMDataset(df=train_df, tokenizer=tokenizer, use_PID=use_PID)
    val_dataset = Cas9PAMDataset(df=val_df, tokenizer=tokenizer, use_PID=use_PID)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    lr = 0.0003
    train_epochs = 20
    train_batch_size = 16
    val_batch_size = 16
    save_steps = 100
    logging_steps = 100
    eval_steps = 100

    output_dir = os.path.join(
        "out",
        f"{args.esm_model}-pam_predict-{args.exp_dir}",
        f"run_{args.fold}",
        "confidence"
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=eval_steps,
        save_steps=save_steps,
        logging_steps=logging_steps,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=val_batch_size,
        label_names=["pam_logos", "pam_converted"],
        num_train_epochs=train_epochs,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )
    utils.save_config(config, output_dir)

    trainer = CustomConfidenceTrainer(
        lr=lr,
        model=confidence_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    trainer.add_callback(utils.LossPlotterCallback(save_dir=output_dir, save_interval=eval_steps))

    logger.info("Starting confidence head training...")
    trainer.train()
    logger.info(f"Confidence head training completed. Model saved at {output_dir}")


if __name__ == '__main__':
    main()
