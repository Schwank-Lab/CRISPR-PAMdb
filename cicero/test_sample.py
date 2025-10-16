import os
import re
import json
import torch
import random
import argparse
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from safetensors import safe_open
from functools import partial
import logging
from pathlib import Path

import utils
import train
import train_confidence
import test

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

def load_fasta(filepath):
    sequences = {}
    with open(filepath, "r") as f:
        header = None
        seq_lines = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header:
                    sequences[header] = "".join(seq_lines)
                header = line[1:].strip()
                seq_lines = []
            else:
                seq_lines.append(line)
        if header:
            sequences[header] = "".join(seq_lines)
    return sequences


class SimpleFASTAInferenceDataset(Dataset):
    def __init__(self, fasta_dict, tokenizer):
        self.items = list(fasta_dict.items())
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        header, seq = self.items[idx]
        inputs = self.tokenizer(
            seq, return_tensors="pt", truncation=False, padding=False
        )
        return {
            "cas9_id": header,
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
        }
    

def evaluate_fold(load_dir, test_dataloader, model_checkpoint, dropout_prob, device, use_confidence=False):
    config = utils.load_config(os.path.join(load_dir, 'config.json'))
    hidden_dim = config['hidden_dim']
    out_dim = config.get('out_dim', 40)
    test_dataloader.dataset.use_PID = config.get('use_PID', False)

    base_model = AutoModel.from_pretrained(model_checkpoint)
    model = train.ESMWithMLPHead(base_model, out_dim=out_dim, hidden_dim=hidden_dim, dropout_prob=dropout_prob)

    checkpoint = test.get_latest_checkpoint(load_dir)
    safetensors_path = os.path.join(checkpoint, "model.safetensors")
    logging.info(f"Loading checkpoint from: {safetensors_path}")

    with safe_open(safetensors_path, framework="pt", device="cpu") as f:
        state_dict = {key: f.get_tensor(key) for key in f.keys()}
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()

    # Load confidence model if enabled
    if use_confidence:
        confidence_model = train_confidence.ESMWithMLPHeadAndConfidence(
            base_model, out_dim=out_dim, hidden_dim=hidden_dim, dropout_prob=dropout_prob
        )
        conf_dir = os.path.join(load_dir, "confidence")
        conf_checkpoint = test.get_latest_checkpoint(conf_dir, take_ckpt='first')
        conf_model_path = os.path.join(conf_checkpoint, "model.safetensors")
        logging.info(f"Loading confidence checkpoint from: {conf_model_path}")

        with safe_open(conf_model_path, framework="pt", device="cpu") as f:
            conf_state_dict = {key: f.get_tensor(key) for key in f.keys()}
        confidence_model.load_state_dict(conf_state_dict, strict=True)
        confidence_model = confidence_model.to(device).eval()

    all_logits = []
    all_cas9_ids = []
    all_confidence_preds = []

    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            if use_confidence:
                logits, confidence_score = confidence_model(input_ids=input_ids, attention_mask=attention_mask)
                all_confidence_preds.append(confidence_score.cpu().numpy())
            else:
                logits = model(input_ids=input_ids, attention_mask=attention_mask)

            logits = logits.reshape(-1, 10, 4)
            all_logits.append(logits.cpu().numpy())
            all_cas9_ids.extend(batch["cas9_id"])
            
    results = {
        "cas9_ids": np.array(all_cas9_ids),
        "logits": np.concatenate(all_logits),
        "confidence_preds": np.concatenate(all_confidence_preds) if use_confidence else None
    }

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--esm_model", default="esm2_t33_650M_UR50D", type=str)
    parser.add_argument("--exp_dir", default="exp0000", type=str)
    parser.add_argument("--data_file", default="data/sample_protein.fasta", type=str)
    parser.add_argument("--use_confidence", default=True, type=bool)
    parser.add_argument("--fold", default=None, type=lambda x: int(x) if x.lower() != "none" else None)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--test_batch_size", default=1, type=int)
    args = parser.parse_args()

    utils.seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    dropout_prob = 0.0
    threshold = 0.8

    logging.info(f"Evaluating PAM predictor on Gasiunas data using device: {device}")

    model_checkpoint = os.path.join("facebook", args.esm_model)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    curr_path = os.path.dirname(os.path.abspath(__file__))
    fasta_dict = load_fasta(os.path.join(curr_path, args.data_file))
    test_dataset = SimpleFASTAInferenceDataset(fasta_dict, tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size)

    if args.fold is None:
        exp_run_dir = f"out/{args.esm_model}-pam_predict-{args.exp_dir}"
        fold_dirs = [d for d in os.listdir(exp_run_dir) if d.startswith("run_")]
        fold_ids = sorted(int(d.split('_')[-1]) for d in fold_dirs)
    else:
        fold_ids = [args.fold]

    all_logits = []
    all_confidences = []

    for fold in fold_ids:
        load_dir = f"out/{args.esm_model}-pam_predict-{args.exp_dir}/run_{fold}"
        logging.info(f"Evaluating fold {fold} from {load_dir}")

        results = evaluate_fold(
            load_dir, test_dataloader, model_checkpoint, dropout_prob, device, use_confidence=args.use_confidence
        )

        logits_fold = results["logits"]
        all_logits.append(logits_fold)

        if args.use_confidence:
            all_confidences.append(results["confidence_preds"])

    if len(all_logits) > 1:
        averaged_logits = np.mean(np.array(all_logits), axis=0)
        confidence_pred = np.mean(np.array(all_confidences), axis=0) if args.use_confidence else None
    else:
        averaged_logits = all_logits[0]
        confidence_pred = all_confidences[0] if args.use_confidence else None

    logging.info(f"Test PAM prediction on <sample_protein.fasta> done")

    # === Make output directory for predictions ===
    pred_dir = Path(f"predictions/tmp/{args.esm_model}-{args.exp_dir}/")
    pred_dir.mkdir(parents=True, exist_ok=True)

    # Save logos
    for i in range(len(averaged_logits)):
        save_path = pred_dir / f"{results['cas9_ids'][i]}_pamlogo.png"
        conf = confidence_pred[i] if args.use_confidence else None
        utils.save_logo_plot(averaged_logits[i], save_path)

if __name__ == "__main__":
    main()
