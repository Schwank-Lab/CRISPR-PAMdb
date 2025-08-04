import os
import re
import json
import torch
import argparse
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, DataCollatorWithPadding
from safetensors import safe_open
import logging
from tqdm import tqdm
from functools import partial

import utils
import train
import train_confidence

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Environment setup
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def custom_collate_fn(batch, tokenizer):
    token_fields = ["input_ids", "attention_mask"]
    token_batch = [{k: sample[k] for k in token_fields} for sample in batch]
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    collated_tokens = data_collator(token_batch)

    extra = {}
    for key in batch[0].keys():
        if key not in token_fields:
            values = [sample[key] for sample in batch]
            if isinstance(values[0], torch.Tensor):
                try:
                    extra[key] = torch.stack(values)
                except Exception:
                    extra[key] = values
            else:
                extra[key] = values

    return {**collated_tokens, **extra}


def get_latest_checkpoint(load_dir, take_ckpt='last'):
    checkpoint_dirs = [
        d for d in os.listdir(load_dir)
        if d.startswith("checkpoint-") and os.path.isdir(os.path.join(load_dir, d))
    ]
    if not checkpoint_dirs:
        raise ValueError(f"No checkpoint directories found in {load_dir}")
    pattern = re.compile(r"checkpoint-(\d+)$")
    checkpoints = []
    for d in checkpoint_dirs:
        match = pattern.match(d)
        if match:
            checkpoints.append((int(match.group(1)), d))
    if not checkpoints:
        raise ValueError(f"No valid checkpoint directories found in {load_dir}")
    selected = max(checkpoints, key=lambda x: x[0])[1] if take_ckpt == 'last' else min(checkpoints, key=lambda x: x[0])[1]
    return os.path.join(load_dir, selected)


def evaluate_fold(fold, args, model_checkpoint, tokenizer, device, dropout_prob):
    load_dir = os.path.join("out", f"{args.esm_model}-pam_predict-{args.exp_dir}", f"run_{fold}")
    checkpoint = get_latest_checkpoint(load_dir)
    config = utils.load_config(os.path.join(load_dir, 'config.json'))
    hidden_dim = config['hidden_dim']
    out_dim = config.get('out_dim', 40)
    use_PID = config.get('use_PID', False)

    base_model = AutoModel.from_pretrained(model_checkpoint)
    model = train.ESMWithMLPHead(base_model, out_dim=out_dim, hidden_dim=hidden_dim, dropout_prob=dropout_prob)

    safetensors_path = os.path.join(checkpoint, "model.safetensors")
    logger.info(f"Loading checkpoint from: {safetensors_path}")
    with safe_open(safetensors_path, framework="pt", device="cpu") as f:
        state_dict = {key: f.get_tensor(key) for key in f.keys()}
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device).eval()

    # Load confidence model if enabled
    confidence_model = None
    if args.use_confidence:
        confidence_model = train_confidence.ESMWithMLPHeadAndConfidence(base_model, out_dim, hidden_dim, dropout_prob)
        conf_dir = os.path.join(load_dir, "confidence")
        conf_checkpoint = get_latest_checkpoint(conf_dir, take_ckpt='first')
        conf_path = os.path.join(conf_checkpoint, "model.safetensors")
        logger.info(f"Loading confidence checkpoint from: {conf_path}")
        with safe_open(conf_path, framework="pt", device="cpu") as f:
            conf_state_dict = {key: f.get_tensor(key) for key in f.keys()}
        confidence_model.load_state_dict(conf_state_dict, strict=True)
        confidence_model = confidence_model.to(device).eval()

    # Load test data
    test_file = os.path.join(args.data_dir, f"test_{fold}.h5")
    test_df = pd.DataFrame(utils.load_h5(test_file))
    test_dataset = train.Cas9PAMDataset(df=test_df, tokenizer=tokenizer, use_PID=use_PID)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size,
                             collate_fn=lambda x: custom_collate_fn(x, tokenizer), shuffle=False)

    logits_all, targets_all, confidence_all = [], [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Testing fold {fold}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pam_logos = batch["pam_logos"].float().to(device)

            if confidence_model:
                logits, conf_score = confidence_model(input_ids=input_ids, attention_mask=attention_mask)
                confidence_all.append(conf_score.cpu())
            else:
                logits = model(input_ids=input_ids, attention_mask=attention_mask)

            logits_all.append(logits.cpu())
            targets_all.append(pam_logos.cpu())

    logits_all = torch.cat(logits_all, dim=0)
    targets_all = torch.cat(targets_all, dim=0)
    confidence_all = torch.cat(confidence_all, dim=0) if confidence_model else None

    acc = train.augmented_cosine_similarity(
        logits=logits_all,
        targets_logos=targets_all,
        is_info=False
    ).cpu().numpy()

    logger.info(f"Fold {fold} - median accuracy: {np.median(acc):.4f}")
    return acc, confidence_all.numpy() if confidence_model else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--esm_model", default="esm2_t33_650M_UR50D", type=str)
    parser.add_argument("--exp_dir", default="exp0000", type=str)
    parser.add_argument("--fold", default=None, type=lambda x: int(x) if x.lower() != "none" else None)
    parser.add_argument("--data_dir", default="data/", type=str)
    parser.add_argument("--test_batch_size", default=2, type=int)
    parser.add_argument("--use_confidence", default=True, type=bool)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    logger.info(f"Running evaluation on device: {device}")
    utils.seed_all(args.seed)
    dropout_prob = 0.0
    threshold = 0.8

    model_checkpoint = os.path.join("facebook", args.esm_model)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    # Handle folds
    if args.fold is None:
        exp_dir = os.path.join("out", f"{args.esm_model}-pam_predict-{args.exp_dir}")
        fold_dirs = [d for d in os.listdir(exp_dir) if d.startswith("run_")]
        folds = sorted([int(d.split("_")[-1]) for d in fold_dirs])
    else:
        folds = [args.fold]

    all_acc = []
    all_conf = []

    for fold in folds:
        acc, conf = evaluate_fold(fold, args, model_checkpoint, tokenizer, device, dropout_prob)
        all_acc.append(acc)
        if conf is not None:
            all_conf.append(conf)

    # Combine results across folds
    all_acc = np.concatenate(all_acc)
    logger.info(f"Overall median accuracy: {np.median(all_acc):.4f} | mean: {np.mean(all_acc):.4f}")

    if args.use_confidence:
        confidence_pred = np.concatenate(all_conf)
        confident_acc = all_acc[confidence_pred.squeeze() > threshold]
        if len(confident_acc) > 0:
            logger.info(f"Confidence-filtered median accuracy (> {threshold:.1f}): {np.median(confident_acc):.4f} "
                        f"({len(confident_acc)}/{len(all_acc)})")
        else:
            logger.info(f"No predictions above confidence threshold {threshold}")

if __name__ == "__main__":
    main()
