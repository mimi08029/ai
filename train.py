import math
import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import tqdm

from load_dataset import load_data, make_dataset, make_dataloader
from model import TTSModel
from tokenizer import tokenize, vocab_size
from utils import train_val_split, augment_mel, guided_attn_loss, cosine_teach_force

device      = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
EPOCHS      = 20
D_MODEL     = 500
MEL_DIM     = 80
NUM_DATA    = 2
BATCH_SIZE  = 2
VOCAB_SIZE  = vocab_size()
SPLIT       = 0.8
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
    criterion_stop = nn.BCELoss()   # stop token loss
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, patience=2)

    try:
        model.load_state_dict(torch.load("tts_model.pt", map_location=device))
        print("Loaded checkpoint.")
    except Exception as e:
        print("No/invalid checkpoint, training from scratch.", e)

    train_losses, val_losses = [], []

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        teacher_forcing = cosine_teach_force((epoch + 1) / EPOCHS)

        for mel, label in tqdm.tqdm(dataloader_train, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
            mel   = mel.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            torch.cuda.empty_cache()
            mel_aug = augment_mel(mel)
            mel_out, mel_post, stop_out, attn = model(label, mel_aug,
                                                      teacher_forcing=teacher_forcing,
                                                      return_alignments=True)

            stop_target = torch.zeros_like(stop_out)

            loss_mel = criterion_mel(mel_out, mel) + criterion_mel(mel_post, mel)

            loss_attn = guided_attn_loss(attn, g=0.2)

            loss = loss_mel + 1.0 * loss_attn
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / max(1, len(dataloader_train))
        train_losses.append(avg_train_loss)

        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for mel, label in tqdm.tqdm(dataloader_val, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
                mel   = mel.to(device)
                label = label.to(device)
                print(mel.shape)
                mel_out, mel_post, _, _ = model.inference(label, max_len=mel.size(1))

                loss_mel = criterion_mel(mel_out, mel) + criterion_mel(mel_post, mel)
                running_val_loss += loss_mel.item()

        avg_val_loss = running_val_loss / max(1, len(dataloader_val))
        val_losses.append(avg_val_loss)

        print(f"Epoch [{epoch + 1}/{EPOCHS}] "
              f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.1e} | Teach Force: {teacher_forcing:.2f}")

        if (epoch + 1) % 5 == 0 and attn is not None:
            with torch.no_grad():
                plt.imshow(attn[0].detach().cpu().numpy(), aspect="auto", origin="lower")
                plt.title("Attention Alignment")
                plt.show()

                plt.imshow(mel_post[0].detach().cpu().numpy(), aspect="auto", origin="lower")
                plt.title("Mel Post")
                plt.show()

                mel_out, mel_post, stop_out, attn = model.inference(label, max_len=mel.size(1),
                                                                    return_alignments=True)
                plt.figure(figsize=(8,4))
                plt.title(f"Inference Example (Epoch {epoch+1})")
                plt.imshow(mel_post[0].cpu().numpy().T, aspect="auto", origin="lower")
                plt.colorbar()
                plt.show()

        torch.save(model.state_dict(), "tts_model.pt")

    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
