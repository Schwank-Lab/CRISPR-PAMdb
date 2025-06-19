import os
import re
import json
import torch
import random
import h5py
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scipy.stats as stats
import logomaker
from pathlib import Path
from transformers import TrainerCallback
import logging

import matplotlib
matplotlib.use("Agg")

def seed_all(seed: int) -> None:
    """Seed all random generators for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def str2bool(v) -> bool:
    """Convert string to boolean for argparse."""
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in ('yes', 'true', 't', '1'):
        return True
    if v in ('no', 'false', 'f', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


def load_h5_Gasiunas(h5_path: str) -> list:
    """Load HDF5 file from Gasiunas dataset into a list of sample dicts."""
    data_list = []
    with h5py.File(h5_path, "r") as f:
        for sample_id in f.keys():
            grp = f[sample_id]
            sample_data = {k: (v[()].decode("utf-8") if isinstance(v[()], bytes) else v[()])
                           for k, v in grp.items()}
            sample_data.setdefault("Protein ID", sample_id)
            data_list.append(sample_data)
    return data_list


def load_h5(h5_path: str, unlabeled: bool = False) -> list:
    """Load HDF5 file (labeled or unlabeled) into a list of sample dicts."""
    data_list = []
    with h5py.File(h5_path, "r") as f:
        for sample_id in f.keys():
            grp = f[sample_id]
            if unlabeled:
                seq = grp["sequence"][()]
                seq = seq.decode("utf-8") if isinstance(seq, bytes) else seq
                data_list.append({"id": sample_id, "sequence": seq})
            else:
                sequence = grp["sequence"][()]
                pam = grp["pam"][()]
                pam_logits = grp["pam_logits"][()]
                pam_converted = grp["pam_converted"][()]
                cluster_id = grp["cluster_id"][()]
                pid_start = grp["pid_start"][()] if "pid_start" in grp else None
                pid_end = grp["pid_end"][()] if "pid_end" in grp else None

                # Decode strings
                sequence = sequence.decode("utf-8") if isinstance(sequence, bytes) else sequence
                pam = pam.decode("utf-8") if isinstance(pam, bytes) else pam

                data_list.append({
                    "id": sample_id,
                    "sequence": sequence,
                    "pam": pam,
                    "pam_logits": pam_logits,
                    "pam_converted": pam_converted,
                    "cluster_id": cluster_id,
                    "pid_start": pid_start,
                    "pid_end": pid_end
                })
    return data_list


class LossPlotterCallback(TrainerCallback):
    """TrainerCallback to plot training/evaluation loss every N steps."""

    def __init__(self, save_dir: str, save_interval: int = 100):
        self.train_losses = []
        self.eval_losses = []
        self.save_dir = save_dir
        self.save_interval = save_interval

    def on_log(self, args, state, control, logs=None, **kwargs):
        step = state.global_step
        if logs is not None:
            if "loss" in logs:
                self.train_losses.append((step, logs["loss"]))
            if "eval_loss" in logs:
                self.eval_losses.append((step, logs["eval_loss"]))

            if step % self.save_interval == 0:
                fig, ax = plt.subplots(figsize=(8, 6))
                if self.train_losses:
                    x, y = zip(*self.train_losses)
                    ax.plot(x, y, label="Train Loss", marker="o")
                if self.eval_losses:
                    x, y = zip(*self.eval_losses)
                    ax.plot(x, y, label="Eval Loss", marker="x")

                ax.set_xlabel("Global Step")
                ax.set_ylabel("Loss")
                ax.set_title("Training & Evaluation Loss")
                ax.legend()
                os.makedirs(self.save_dir, exist_ok=True)
                fig.savefig(os.path.join(self.save_dir, "loss_plot.png"))
                plt.close(fig)
        return control


def get_next_experiment_dir(output_dir: str) -> str:
    """
    Create a new experiment directory based on existing ones using suffix -expXXXX.
    """
    output_dir = output_dir.rstrip(os.sep)
    parent_dir = os.path.dirname(output_dir)
    base_name = os.path.basename(output_dir)
    pattern = re.compile(re.escape(base_name) + r'-exp(\d{4})$')

    max_exp = -1
    for entry in os.listdir(parent_dir):
        match = pattern.match(entry)
        if match:
            max_exp = max(max_exp, int(match.group(1)))

    new_exp = max_exp + 1
    new_dir = os.path.join(parent_dir, f"{base_name}-exp{new_exp:04d}")
    return new_dir


def save_config(config_dict: dict, save_dir: str, filename: str = "config.json") -> None:
    """Save config dictionary to JSON file in the given directory."""
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    with open(filepath, 'w') as f:
        json.dump(config_dict, f, indent=4)
    logging.info(f"Config saved to {filepath}")


def load_config(config_path: str) -> dict:
    """Load config dictionary from a JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def logits_to_info_content(logits: np.ndarray, bg_prob: float = 0.25, eps: float = 1e-6) -> np.ndarray:
    """
    Convert logits to information content matrix (Rseq) across positions.
    """
    p = softmax(logits)
    R = np.sum(p * (np.log2(p + eps) - np.log2(bg_prob)), axis=1)
    return p * R[:, np.newaxis]


def softmax(x: np.ndarray) -> np.ndarray:
    """Row-wise softmax function for 2D NumPy arrays."""
    x_max = np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)
