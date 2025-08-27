import librosa
import numpy as np
import torch
import torchaudio
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
    min_level_db=-60,
):
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    spectrogram = np.abs(stft) ** 2  # power spectrogram

    mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
    mel = np.dot(mel_basis, spectrogram)

    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db

def normalize_mel(mel_db: torch.Tensor):
    mel_min = np.min(mel_db)
    mel_max = np.max(mel_db)
    return (mel_db - mel_min) / (mel_max - mel_min + 1e-9)

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
    freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=25)
    time_mask = torchaudio.transforms.TimeMasking(time_mask_param=25)

    mel = freq_mask(mel)
    mel = time_mask(mel)
    return mel

def add_noise(mel, noise_level):
    return mel + noise_level * torch.randn_like(mel)

def random_gain(mel, min_gain=0.6, max_gain=1.4):
    gain = torch.empty(1).uniform_(min_gain, max_gain).item()
    return mel * gain

def augment_mel(mel):
    mel = add_noise(mel, noise_level=0.5)
    mel = spec_augment(mel)
    mel = random_gain(mel)
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
