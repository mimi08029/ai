from datasets import load_dataset
from huggingface_hub import login
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch

from utils import wav_to_mel, transpose_major
import os
_data_token = os.getenv("HF_TOKEN")

def load_data(num_points):
    dataset = load_dataset(
        "amphion/Emilia-Dataset",
        data_files={"en": "Emilia/EN/*.tar"},
        split="en",
        streaming=True
    )

    pairs = []
    for i, sample in enumerate(dataset):
        if i >= num_points:
            break

        audio = sample["mp3"]["array"]   # waveform as np.array
        text = sample["json"]["text"]    # transcript
        pairs.append((audio, text))

    return pairs


class TTSDataset(Dataset):
    def __init__(self, pairs, tokenize):
        self.pairs = pairs
        self.tokenize = tokenize

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        audio, text = self.pairs[idx]
        audio_norm = wav_to_mel(audio)
        audio_tensor = transpose_major(torch.tensor(audio_norm, dtype=torch.float32))
        text_tensor = torch.tensor(self.tokenize(text), dtype=torch.long)
        return audio_tensor, text_tensor


def make_dataset(pairs, tokenize_fn=None):
    return TTSDataset(pairs, tokenize=tokenize_fn)

def collate_fn(batch):
    audios, texts = zip(*batch)

    audios = [a.squeeze() for a in audios]
    audio_lengths = torch.tensor([a.size(0) for a in audios], dtype=torch.long)
    text_lengths = torch.tensor([t.size(0) for t in texts], dtype=torch.long)

    audios_padded = pad_sequence(audios, batch_first=True)
    texts_padded = pad_sequence(texts, batch_first=True, padding_value=0)

    stops = (torch.arange(audios_padded.size(1))[None, :] > audio_lengths[:, None]).to(torch.float)
    text_mask  = torch.arange(texts_padded.size(1))[None, :] < text_lengths[:, None]

    return audios_padded, texts_padded, stops, text_mask


def make_dataloader(dataset, batch_size, shuffle=True):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn
    )
