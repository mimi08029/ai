import torch
import numpy as np
import matplotlib.pyplot as plt

from load_dataset import load_data, make_dataset, make_dataloader
from model import TTSModel
from tokenizer import tokenize, vocab_size
from utils import sample_text_to_emb, transpose_major, load_model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

D_MODEL = 512
MEL_DIM = 80

def create_model(checkpoint_path="tts_model.pt"):
    """Load the trained TTS model from checkpoint."""
    model = TTSModel(D_MODEL, vocab_size=vocab_size(), mel_dim=MEL_DIM, device=device)
    load_model(model, checkpoint_path)
    model.eval()
    return model

def infer(model, text: str, max_len: int = 500):
    """Generate mel-spectrogram from input text."""
    out, out_post, _, attn = model.inference(sample_text_to_emb(text), max_len=max_len, return_alignments=True)
    return out_post

pairs = load_data(1)
dataset = make_dataset(pairs, tokenize_fn=tokenize)
dataloader = make_dataloader(dataset, 1, False)

if __name__ == "__main__":
    model = create_model("tts_model.pt")
    a, t, _, _ = next(iter(dataloader))
    mel, mel_post, stop, attn = model(t.to(device), a.to(device))

    plt.imshow(mel.detach().cpu().numpy().squeeze(), aspect="auto", origin="lower")
    plt.colorbar()
    plt.title("Generated Mel Spectrogram")
    plt.show()
