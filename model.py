import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

# -----------------------------
# Helper: Conv block for PostNet
# -----------------------------
class ConvPostNetBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int, p: int, dropout: float = 0.5,
                 norm: str = "instance", activation: nn.Module = nn.Tanh):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=p)
        if norm == "batch":
            self.norm = nn.BatchNorm1d(out_ch)
        elif norm == "instance":
            self.norm = nn.InstanceNorm1d(out_ch, affine=True)
        elif norm == "group":
            g = max(1, min(8, out_ch))
            self.norm = nn.GroupNorm(g, out_ch)
        else:
            self.norm = nn.Identity()
        self.act = activation()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        return x


# -----------------------------
# PreNet (with strong dropout per Tacotron2)
# -----------------------------
class PreNet(nn.Module):
    def __init__(self, d_model: int, vocab_size: int, p_drop: float = 0.5):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.fc1 = nn.Linear(d_model, d_model)
        self.fc2 = nn.Linear(d_model, d_model)
        self.fc3 = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(p_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T]
        h = self.emb(x)                 # [B, T, D]
        h = F.relu(self.fc1(h))
        h = self.drop(h)
        h = F.relu(self.fc2(h))
        h = self.drop(h)
        h = self.fc3(h)
        return h                        # [B, T, D]


# -----------------------------
# PostNet (outputs a residual CORRECTION; no internal residual add)
# -----------------------------
class PostNet(nn.Module):
    def __init__(self, mel_dim: int, num_layers: int = 5, kernel_size: int = 5, channels: int = 512,
                 norm: str = "instance", dropout: float = 0.5):
        super().__init__()
        pads = (kernel_size // 2)
        blocks = []
        # first layer
        blocks.append(ConvPostNetBlock(mel_dim, channels, kernel_size, pads, dropout=dropout, norm=norm))
        # middle layers
        for _ in range(num_layers - 2):
            blocks.append(ConvPostNetBlock(channels, channels, kernel_size, pads, dropout=dropout, norm=norm))
        # final layer (no activation, no dropout)
        self.conv_last = nn.Conv1d(channels, mel_dim, kernel_size=1, padding=0)
        self.norm_last = nn.Identity()
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, mel]
        y = x.transpose(1, 2)  # [B, mel, T]
        for blk in self.blocks:
            y = blk(y)
        y = self.conv_last(y)
        y = y.transpose(1, 2)  # [B, T, mel]
        return y  # residual correction; caller should add to input mels


# -----------------------------
# Location Sensitive Attention
# -----------------------------
class LocationSensitiveAttention(nn.Module):
    def __init__(self, d_model: int, kernel_size: int = 31):
        super().__init__()
        self.query_layer = nn.Linear(d_model, d_model, bias=False)
        self.memory_layer = nn.Linear(d_model, d_model, bias=False)
        self.location_layer = nn.Conv1d(2, d_model, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.v = nn.Linear(d_model, 1, bias=False)

    def forward(self, query: torch.Tensor, memory: torch.Tensor,
                prev_attn: torch.Tensor, cum_attn: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # query: [B, D], memory: [B, T_enc, D], prev_attn/cum_attn: [B, T_enc]
        loc = torch.cat([prev_attn.unsqueeze(1), cum_attn.unsqueeze(1)], dim=1)  # [B, 2, T_enc]
        loc = self.location_layer(loc).transpose(1, 2)                           # [B, T_enc, D]
        q = self.query_layer(query).unsqueeze(1)                                 # [B, 1, D]
        m = self.memory_layer(memory)                                            # [B, T_enc, D]
        e = self.v(torch.tanh(q + m + loc)).squeeze(-1)                          # [B, T_enc]
        if mask is not None:
            e = e.masked_fill(mask, float('-inf'))
        a = torch.softmax(e, dim=1)
        ctx = torch.bmm(a.unsqueeze(1), memory).squeeze(1)                        # [B, D]
        return ctx, a


# -----------------------------
# Encoder (BiLSTM + projection)
# -----------------------------
class Encoder(nn.Module):
    def __init__(self, d_model: int, num_layers: int = 1, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(d_model, d_model, num_layers=num_layers, bidirectional=True,
                             batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.linear = nn.Linear(d_model * 2, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.lstm(x)                 # [B, T, 2D]
        y = self.linear(y)                  # [B, T, D]
        return y


# -----------------------------
# Decoder with stop-token branch and dropout around LSTMCells
# -----------------------------
class Decoder(nn.Module):
    def __init__(self, d_model: int, mel_dim: int = 80, p_drop: float = 0.2):
        super().__init__()
        self.mel_dim = mel_dim
        self.att_rnn = nn.LSTMCell(d_model + mel_dim, d_model)
        self.dec_rnn = nn.LSTMCell(d_model + d_model, d_model)
        self.att = LocationSensitiveAttention(d_model)
        self.lin_mel = nn.Linear(d_model, mel_dim)
        self.lin_stop = nn.Linear(d_model, 1)   # stop-token (gate) logits
        self.drop_in = nn.Dropout(p_drop)
        self.drop_h = nn.Dropout(p_drop)

    def _init_state(self, B: int, D: int, T_enc: int, device) -> Tuple[torch.Tensor, ...]:
        ctx = torch.zeros(B, D, device=device)
        a = torch.zeros(B, T_enc, device=device)
        a_cum = torch.zeros(B, T_enc, device=device)
        h_att = torch.zeros(B, D, device=device); c_att = torch.zeros(B, D, device=device)
        h_dec = torch.zeros(B, D, device=device); c_dec = torch.zeros(B, D, device=device)
        return (ctx, a, a_cum, h_att, c_att, h_dec, c_dec)

    def forward(self, enc: torch.Tensor, mel_in: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Teacher-forced decoding
        B, T_dec, _ = mel_in.shape
        device = enc.device
        D = enc.size(-1); T_enc = enc.size(1)
        state = self._init_state(B, D, T_enc, device)

        mels = []
        stops = []
        for t in range(T_dec):
            prev = mel_in[:, t, :]  # [B, mel]
            m, s, state = self._step(enc, prev, state)
            mels.append(m.unsqueeze(1))
            stops.append(s.unsqueeze(1))
        mel_out = torch.cat(mels, dim=1)       # [B, T_dec, mel]
        stop_logits = torch.cat(stops, dim=1)  # [B, T_dec, 1]
        return mel_out, stop_logits

    def _step(self, enc: torch.Tensor, prev_mel: torch.Tensor,
              state: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, ...]]:
        ctx, a, a_cum, h_att, c_att, h_dec, c_dec = state
        # attention LSTM
        att_in = torch.cat([prev_mel, ctx], dim=-1)
        att_in = self.drop_in(att_in)
        h_att, c_att = self.att_rnn(att_in, (h_att, c_att))
        h_att = self.drop_h(h_att)
        # attention
        ctx, a = self.att(h_att, enc, a, a_cum)
        a_cum = a_cum + a
        # decoder LSTM
        dec_in = torch.cat([h_att, ctx], dim=-1)
        dec_in = self.drop_in(dec_in)
        h_dec, c_dec = self.dec_rnn(dec_in, (h_dec, c_dec))
        h_dec = self.drop_h(h_dec)
        # projections
        mel = self.lin_mel(h_dec)
        stop = self.lin_stop(h_dec).squeeze(-1)  # [B]
        new_state = (ctx, a, a_cum, h_att, c_att, h_dec, c_dec)
        return mel, stop, new_state

    @torch.no_grad()
    def infer(self, enc: torch.Tensor, go_frame: torch.Tensor, max_len: int = 1000,
              stop_threshold: float = 0.6, min_len: int = 10, early_stop_patience: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
        # Autoregressive decoding with stop-token termination
        device = enc.device
        B, T_enc, D = enc.shape
        state = self._init_state(B, D, T_enc, device)
        prev = go_frame  # [B, mel]
        outputs = []
        gate_logits = []
        over_thresh = 0
        for t in range(max_len):
            mel, stop, state = self._step(enc, prev, state)
            outputs.append(mel.unsqueeze(1))
            gate_logits.append(stop.unsqueeze(1))
            prev = mel
            if t + 1 >= min_len:
                if torch.sigmoid(stop).mean().item() > stop_threshold:
                    over_thresh += 1
                else:
                    over_thresh = 0
                if over_thresh >= early_stop_patience:
                    break
        mel_out = torch.cat(outputs, dim=1) if outputs else torch.zeros(B, 0, self.mel_dim, device=device)
        gates = torch.cat(gate_logits, dim=1) if gate_logits else torch.zeros(B, 0, 1, device=device)
        return mel_out, gates


# -----------------------------
# Full Model
# -----------------------------
class TTSModel(nn.Module):
    def __init__(self, d_model: int, vocab_size: int, mel_dim: int, device: torch.device):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.mel_dim = mel_dim
        self.device = device
        # modules
        self.pre = PreNet(d_model, vocab_size, p_drop=0.5)
        self.encoder = Encoder(d_model, num_layers=1, dropout=0.2)
        self.decoder = Decoder(d_model, mel_dim, p_drop=0.2)
        self.post = PostNet(mel_dim, num_layers=5, kernel_size=5, channels=512, norm="instance", dropout=0.5)
        self.to(device)

    def get_go(self, B: int) -> torch.Tensor:
        return torch.zeros((B, 1, self.mel_dim), device=self.device)

    # Backward compatible forward: by default returns only mel_post like your original code.
    # If return_gate=True, also returns stop-token logits so you can add a gate loss.
    def forward(self, x: torch.Tensor, mel: torch.Tensor, return_gate: bool = False):
        # x: [B, T_txt], mel: [B, T_mel, mel_dim]
        enc_in = self.pre(x)
        enc_out = self.encoder(enc_in)
        # Teacher forcing inputs (prepend <GO>)
        go = self.get_go(enc_out.size(0))
        mel_tf = torch.cat([go, mel[:, :-1, :]], dim=1)
        mel_out, stop_logits = self.decoder(enc_out, mel_tf)   # raw decoder mels
        mel_post = self.post(mel_out) + mel_out                # add residual correction
        if return_gate:
            return mel_post, stop_logits
        return mel_post

    @torch.no_grad()
    def inference(self, x: torch.Tensor, max_len: int = 1000):
        self.eval()
        device = next(self.parameters()).device
        x = x.to(device)
        enc_in = self.pre(x)
        enc_out = self.encoder(enc_in)
        B = enc_out.size(0)
        go = self.get_go(B).squeeze(1)  # [B, mel]
        mel_out, gates = self.decoder.infer(enc_out, go, max_len=max_len)
        mel_post = self.post(mel_out) + mel_out
        return mel_post  # for compatibility; you can also return gates if you want
