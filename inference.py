import torch
import numpy as np
import matplotlib.pyplot as plt
from numpy.ma.core import transpose

from model import TTSModel
from tokenizer import tokenize, vocab_size
from utils import sample_text_to_emb, transpose_major

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

D_MODEL = 512
MEL_DIM = 80

def load_model(checkpoint_path="tts_model.pt"):
    """Load the trained TTS model from checkpoint."""
    model = TTSModel(D_MODEL, vocab_size=vocab_size(), mel_dim=MEL_DIM, device=device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model

def infer(model, text: str, max_len: int = 500):
    """Generate mel-spectrogram from input text."""
    out = model.inference(sample_text_to_emb(text), max_len=max_len)
    return out


if __name__ == "__main__":
    model = load_model("tts_model.pt")

    text = "Hello, this is a test of our text to speech model."
    mel = infer(model, text, max_len=1000)

    plt.imshow(mel.cpu().numpy().squeeze(), aspect="auto", origin="lower")
    plt.colorbar()
    plt.title("Generated Mel Spectrogram")
    plt.show()
