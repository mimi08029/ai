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

device      = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
EPOCHS      = 40
D_MODEL     = 512
MEL_DIM     = 80
NUM_DATA    = 4000        # tiny sanity set
BATCH_SIZE  = 4
VOCAB_SIZE  = vocab_size()
SPLIT       = 0.9
LR          = 1e-3



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
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
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

        bar = tqdm.tqdm(dataloader_train, desc=f"Epoch {epoch + 1}/{EPOCHS} [Train]", leave=True)
        for mel, label, stops, text_mask in bar:
            mel = mel.to(device)
            label = label.to(device)
            text_mask = text_mask.to(device)
            stops = stops.to(device)
            optimizer.zero_grad()

            mel_out, mel_post, stop_out, attn = model(
                label, augment_mel(mel),
                teacher_forcing=teacher_forcing,
                enc_mask=text_mask,
                return_alignments=True
            )

            loss_mel = criterion_mel(mel_out, mel) + criterion_mel_post(mel_post, mel)
            loss_stop = criterion_stop(stop_out, stops)
            loss_attn = guided_attn_loss(attn, g=0.2)

            loss = loss_mel + loss_attn * 0.5 + loss_stop * 0.1
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
            for mel, label, stops, text_mask in tqdm.tqdm(dataloader_val,
                                        desc=f"Epoch {epoch + 1}/{EPOCHS} [Val]",
                                        leave=True):
                mel = mel.to(device)
                label = label.to(device)
                text_mask = text_mask.to(device)
                stops = stops.to(device)
                mel_out_val, mel_post_val, stop_out, attn = model(
                    label, mel,
                    teacher_forcing=1.0,
                    return_alignments=True,
                    enc_mask=text_mask,
                )

                loss_mel = criterion_mel(mel_out_val, mel) + criterion_mel_post(mel_post_val, mel)
                loss_stop = criterion_stop(stop_out, stops)
                loss_attn = guided_attn_loss(attn, g=0.2)
                loss = loss_mel + loss_attn * 0.5 + loss_stop * 0.1
                running_val_loss += loss.item()

        if (epoch + 1) % (EPOCHS // 10) == 0:
            with torch.no_grad():
                _, mel_post, stop_out, attn = model.inference(
                    label, max_len=mel.size(1), return_alignments=True
                )

                fig, axes = plt.subplots(1, 3, figsize=(18, 5))

                im0 = axes[0].imshow(attn[0].detach().cpu().numpy(), aspect="auto", origin="lower")
                axes[0].set_title("Attention Alignment")
                fig.colorbar(im0, ax=axes[0])

                im1 = axes[1].imshow(mel_out[0].detach().cpu().numpy(), aspect="auto", origin="lower")
                axes[1].set_title("Mel out (Training)")
                fig.colorbar(im1, ax=axes[1])

                im2 = axes[2].imshow(mel_post[0].cpu().numpy().T, aspect="auto", origin="lower")
                axes[2].set_title(f"Inference Example (Epoch {epoch + 1})")
                fig.colorbar(im2, ax=axes[2])

                plt.tight_layout()
                plt.show()
                plt.close()

        save_model(model, "tts_model.pt", optimizer)
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
