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

import utils
import train
import train_confidence
import test

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


class GasiunasDataset(Dataset):
    def __init__(self, df, tokenizer, use_PID=False):
        self.df = df
        self.tokenizer = tokenizer
        self.use_PID = use_PID

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        seq = row['PI domain sequence'] if self.use_PID else row['Protein sequence']

        inputs = self.tokenizer(seq, truncation=False, padding=False, return_tensors="pt")
        input_ids = inputs['input_ids'].squeeze(0)
        attention_mask = inputs['attention_mask'].squeeze(0)

        pam_logos = torch.tensor(np.array(eval(row['Experimental PAM logo (A,C,G,T)'])), dtype=torch.float32)
        protein2pam_acc = torch.tensor(row['PAM prediction accuracy'], dtype=torch.float32)
        protein2pam_conf = torch.tensor(row['Protein2PAM confidence score'], dtype=torch.float32)

        return {
            "cas9_id": row['Protein ID'],
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pam_logos": pam_logos,
            "Protein2PAM_acc": protein2pam_acc,
            "Protein2PAM_conf": protein2pam_conf
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

    all_logits, all_labels_logos = [], []
    all_lengths, all_cas9_ids = [], []
    all_protein2pam_acc, all_protein2pam_conf = [], []
    all_confidence_preds = []

    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pam_logos = batch["pam_logos"].to(dtype=torch.float32).to(device)

            if use_confidence:
                logits, confidence_score = confidence_model(input_ids=input_ids, attention_mask=attention_mask)
                all_confidence_preds.append(confidence_score.cpu().numpy())
            else:
                logits = model(input_ids=input_ids, attention_mask=attention_mask)

            logits = logits.reshape(-1, 10, 4)

            all_logits.append(logits.cpu().numpy())
            all_labels_logos.append(pam_logos.cpu().numpy())
            all_lengths.append(attention_mask.sum(dim=1).cpu().numpy())
            all_cas9_ids.extend(batch["cas9_id"])
            all_protein2pam_acc.append(batch["Protein2PAM_acc"].cpu().numpy())
            all_protein2pam_conf.append(batch["Protein2PAM_conf"].cpu().numpy())

    results = {
        "logits": np.concatenate(all_logits),
        "labels": np.concatenate(all_labels_logos),
        "lengths": np.concatenate(all_lengths),
        "cas9_ids": np.array(all_cas9_ids),
        "protein2pam_acc": np.concatenate(all_protein2pam_acc),
        "protein2pam_conf": np.concatenate(all_protein2pam_conf),
        "confidence_preds": np.concatenate(all_confidence_preds) if use_confidence else None
    }

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--esm_model", default="esm2_t6_8M_UR50D", type=str)
    parser.add_argument("--exp_dir", default="exp0000", type=str)
    parser.add_argument("--data_dir", default="data/", type=str)
    parser.add_argument("--use_confidence", default=False, type=bool)
    parser.add_argument("--fold", default=None, type=lambda x: int(x) if x.lower() != "none" else None)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--test_batch_size", default=2, type=int)
    args = parser.parse_args()

    utils.seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    dropout_prob = 0.0
    threshold = 0.8

    logging.info(f"Evaluating PAM predictor on Gasiunas data using device: {device}")

    model_checkpoint = os.path.join("facebook", args.esm_model)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    test_df = pd.DataFrame(utils.load_h5_Gasiunas(os.path.join(args.data_dir, "test_Gasiunas.h5")))
    test_dataset = GasiunasDataset(test_df, tokenizer)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        collate_fn=partial(test.custom_collate_fn, tokenizer=tokenizer),
    )

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

        acc = train.augmented_cosine_similarity(
            torch.tensor(logits_fold), torch.tensor(results["labels"]), is_info=False
        ).cpu().numpy()

        logging.info(f"Fold {fold} augCosSim median accuracy: {np.median(acc):.4f}")

        if args.use_confidence:
            all_confidences.append(results["confidence_preds"])

    if len(all_logits) > 1:
        averaged_logits = np.mean(np.array(all_logits), axis=0)
        confidence_pred = np.mean(np.array(all_confidences), axis=0) if args.use_confidence else None
    else:
        averaged_logits = all_logits[0]
        confidence_pred = all_confidences[0] if args.use_confidence else None

    final_acc = train.augmented_cosine_similarity(
        torch.tensor(averaged_logits), torch.tensor(results["labels"]), is_info=False
    ).cpu().numpy()

    logging.info(f"Test PAM prediction on Gasiunas dataset - median accuracy: {np.median(final_acc):.4f}")

    if args.use_confidence:
        filtered_acc = final_acc[confidence_pred.squeeze() > threshold]
        if len(filtered_acc) > 0:
            logging.info(
                f"Filtered accuracy with confidence > {threshold}: {np.median(filtered_acc):.4f} "
                f"({len(filtered_acc)}/{len(final_acc)} samples)"
            )
        else:
            logging.info(f"No samples exceed the confidence threshold of {threshold}.")

if __name__ == "__main__":
    main()
