import math

import librosa
import numpy as np
import torch
import torchaudio
from matplotlib import pyplot as plt
from torch.utils.data import random_split

from tokenizer import tokenize
def wav_to_mel(
    y,
    sr=22050,
    n_fft=1024,
    hop_length=256,
    win_length=1024,
    n_mels=80,
    fmin=0,
    fmax=8000,
    min_level_db=-80.0,
    ref_level_db=1.0,   # new
):
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    spectrogram = np.abs(stft)
    mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
    mel = np.dot(mel_basis, spectrogram)
    mel = np.maximum(mel, 1e-5)

    mel_db = librosa.power_to_db(mel, ref=ref_level_db)
    mel_db = np.clip(mel_db, min_level_db, 0)

    mel_norm = (mel_db - min_level_db) / -min_level_db
    return mel_norm.astype(np.float32)

def train_val_split(dataset, split=0.8):
    n_total = len(dataset)
    n_train = int(n_total * split)
    n_val = n_total - n_train

    train_dataset, val_dataset = random_split(
        dataset,
        [n_train, n_val],
    )

    return train_dataset, val_dataset


def spec_augment(mel):
    freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=15)
    time_mask = torchaudio.transforms.TimeMasking(time_mask_param=10)

    mel = freq_mask(mel)
    mel = time_mask(mel)
    return mel


def add_noise(mel, noise_level):
    return mel + noise_level * torch.randn_like(mel)


def augment_mel(mel):
    mel = add_noise(mel, noise_level=0.1)
    return mel


def transpose_major(x):
    if isinstance(x, torch.Tensor):
        return x.transpose(-2, -1)
    else:
        if x.ndim == 2:
            return x.transpose(1, 0)
        return x.transpose(0, 2, 1)


def sample_text_to_emb(text):
    text_tensor = torch.tensor(tokenize(text), dtype=torch.long)
    return text_tensor.unsqueeze(0)


def save_model(model, path, optimizer=None):
    checkpoint = {
        "model_state_dict": model.state_dict(),
    }
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(checkpoint, path)
    print(f"Model saved to {path}")


def load_model(model, path, optimizer=None, map_location=None):
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    print(f"Model loaded from {path}")
    return model, optimizer


def guided_attn_loss(attn, g=0.2):
    B, T_dec, T_enc = attn.size()
    W = torch.arange(T_dec).unsqueeze(1) / T_dec
    J = torch.arange(T_enc).unsqueeze(0) / T_enc
    G = 1 - torch.exp(-(W - J) ** 2 / (2 * g * g))
    G = G.to(attn.device)
    return (attn * G.unsqueeze(0)).sum()

def cosine_teach_force(progress):
    return 0.2 + 0.8 * (0.5 * (1 + math.cos(math.pi * progress)))
