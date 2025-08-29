import math
import random
import time

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import tqdm

from load_dataset import load_data, make_dataset, make_dataloader
from model import TTSModel
from tokenizer import tokenize, vocab_size
from utils import train_val_split, augment_mel, guided_attn_loss, cosine_teach_force, save_model, load_model

# -------------------------
# Config
# -------------------------
device      = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
EPOCHS      = 2000
D_MODEL     = 512
MEL_DIM     = 80
NUM_DATA    = 2          # tiny sanity set
BATCH_SIZE  = 2
VOCAB_SIZE  = vocab_size()
SPLIT       = 0.8
LR          = 5e-4


if __name__ == "__main__":
    torch.manual_seed(42); np.random.seed(42); random.seed(42)

    pairs = load_data(NUM_DATA)
    dataset = make_dataset(pairs, tokenize_fn=tokenize)
    dataset_train, dataset_val = train_val_split(dataset, SPLIT)
    dataloader_train = make_dataloader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
    dataloader_val   = make_dataloader(dataset_val,   batch_size=BATCH_SIZE, shuffle=False)

    model = TTSModel(D_MODEL, vocab_size=VOCAB_SIZE, mel_dim=MEL_DIM, device=device).to(device)
    criterion_mel = nn.L1Loss()
    criterion_mel_post = nn.L1Loss()
    criterion_stop = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    try:
        load_model(model, "tts_model.pt", optimizer, device)
        optimizer.param_groups[0]["lr"] = LR
        print("Loaded checkpoint.")
    except Exception as e:
        print("No/invalid checkpoint, training from scratch.", e)

    train_losses, val_losses = [], []

    # -------------------------
    # Training loop
    # -------------------------
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        teacher_forcing = 1

        # training bar
        bar = tqdm.tqdm(dataloader_train, desc=f"Epoch {epoch + 1}/{EPOCHS} [Train]", leave=True)
        for mel, label in bar:
            mel = mel.to(device)
            label = label.to(device)

            optimizer.zero_grad()

            mel_out, mel_post, stop_out, attn = model(
                label, mel,
                teacher_forcing=teacher_forcing,
                return_alignments=True
            )

            stop_target = torch.zeros_like(stop_out)
            stop_target[:, -1] = 1.

            loss_mel = criterion_mel(mel_out, mel) + criterion_mel_post(mel_post, mel)
            loss_attn = guided_attn_loss(attn, g=0.2)
            loss_stop = criterion_stop(stop_out, stop_target)

            loss = loss_mel + 2 * loss_attn + 0.1 * loss_stop
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / max(1, len(dataloader_train))
        train_losses.append(avg_train_loss)

        # -------------------------
        # Validation
        # -------------------------
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for mel, label in tqdm.tqdm(dataloader_val,
                                        desc=f"Epoch {epoch + 1}/{EPOCHS} [Val]",
                                        leave=True):
                mel = mel.to(device)
                label = label.to(device)

                mel_out, mel_post, _, _ = model(
                    label, mel,
                    teacher_forcing=1.0,
                    return_alignments=False
                )

                loss_mel = criterion_mel(mel_out, mel) + criterion_mel_post(mel_post, mel)
                running_val_loss += loss_mel.item()

        avg_val_loss = running_val_loss / max(1, len(dataloader_val))
        val_losses.append(avg_val_loss)

        scheduler.step()
        bar.close()
        print(
            f"\nEpoch [{epoch + 1}/{EPOCHS}] "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.1e} | "
            f"Teach Force: {teacher_forcing:.2f}"
        )

    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
