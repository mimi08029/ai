import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from matplotlib import pyplot as plt


# --------------------------
# Zoneout LSTM Cell
# --------------------------
class ZoneoutLSTMCell(nn.Module):
    """LSTMCell with Zoneout regularization (Krueger et al., 2016)."""

    def __init__(self, input_size: int, hidden_size: int, zoneout_prob: float = 0.1):
        super().__init__()
        self.cell = nn.LSTMCell(input_size, hidden_size)
        self.zoneout_prob = zoneout_prob

    def forward(
        self,
        x: torch.Tensor,
        state: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h, c = state
        h_new, c_new = self.cell(x, (h, c))

        if self.training and self.zoneout_prob > 0.0:
            mask_h = torch.empty_like(h).bernoulli_(1 - self.zoneout_prob)
            mask_c = torch.empty_like(c).bernoulli_(1 - self.zoneout_prob)
            h = mask_h * h_new + (1 - mask_h) * h
            c = mask_c * c_new + (1 - mask_c) * c
        else:
            h, c = h_new, c_new

        return h, c


# --------------------------
# Conv PostNet Block
# --------------------------
class ConvPostNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k, p, dropout=0.0,
                 norm="instance", activation=nn.Tanh):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=p)
        self.norm = nn.InstanceNorm1d(out_ch, affine=True)
        self.act = activation()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.act(self.norm(self.conv(x))))


# --------------------------
# Embedding + Conv stack
# --------------------------
class EmbNet(nn.Module):
    def __init__(self, d_model, vocab_size, p_drop=0.5, n_conv=3, k=5):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.emb.weight.data.normal_(0, 1.0 / math.sqrt(d_model))

        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size=k, padding=(k - 1) // 2),
                nn.BatchNorm1d(d_model),
                nn.ReLU(),
                nn.Dropout(p_drop)
            ) for _ in range(n_conv)
        ])

    def forward(self, x):
        h = self.emb(x).transpose(1, 2)
        for conv in self.convs:
            h = conv(h)
        return h.transpose(1, 2)


# --------------------------
# Decoder PreNet
# --------------------------
class DecoderPreNet(nn.Module):
    def __init__(self, in_dim, hidden_sizes=[256, 256], p_drop=0.5):
        super().__init__()
        layers, cur = [], in_dim
        for hdim in hidden_sizes:
            layers += [nn.Linear(cur, hdim), nn.ReLU(), nn.Dropout(p_drop)]
            cur = hdim
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# --------------------------
# PostNet
# --------------------------
class PostNet(nn.Module):
    def __init__(self, mel_dim, num_layers=5, kernel_size=5, norm="instance", dropout=0.2):
        super().__init__()
        pads = kernel_size // 2
        blocks = [ConvPostNetBlock(mel_dim, mel_dim, kernel_size, pads, dropout, norm)]
        for _ in range(num_layers - 2):
            blocks.append(ConvPostNetBlock(mel_dim, mel_dim, kernel_size, pads, dropout, norm))
        self.blocks = nn.ModuleList(blocks)
        self.conv_last = nn.Conv1d(mel_dim, mel_dim, kernel_size=1)

    def forward(self, x):
        y = x.transpose(1, 2)
        for blk in self.blocks:
            y = blk(y)
        y = self.conv_last(y).transpose(1, 2)
        return y


# --------------------------
# Location-Sensitive Attention
# --------------------------
class LocationSensitiveAttention(nn.Module):
    def __init__(self, d_model, kernel_size=31, out_channels=32):
        super().__init__()
        self.query_layer = nn.Linear(d_model, d_model)
        self.memory_layer = nn.Linear(d_model, d_model)
        self.location_conv = nn.Conv1d(2, out_channels, kernel_size, padding=kernel_size // 2)
        self.location_proj = nn.Linear(out_channels, d_model)
        self.v = nn.Linear(d_model, 1)

    def forward(self, query, memory, prev_attn, cum_attn, mask=None):
        B, T_enc, D = memory.shape
        loc = torch.cat([prev_attn.unsqueeze(1), cum_attn.unsqueeze(1)], 1)
        loc = self.location_proj(self.location_conv(loc).transpose(1, 2))
        q = self.query_layer(query).unsqueeze(1)
        m = self.memory_layer(memory)
        e = self.v(torch.tanh(q + m + loc)).squeeze(-1)

        if mask is not None:
            mask = mask.bool()
            e = e.masked_fill(mask, -1e9)
            if mask.all(dim=1).any():
                e[mask.all(dim=1), 0] = 0.0

        a = torch.softmax(e, dim=1)
        ctx = torch.bmm(a.unsqueeze(1), memory).squeeze(1)
        return ctx, a


# --------------------------
# Encoder (BiLSTM + Linear)
# --------------------------
class Encoder(nn.Module):
    def __init__(self, d_model, num_layers=1, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            d_model, d_model, num_layers=num_layers, bidirectional=True,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0
        )
        self.linear = nn.Linear(d_model * 2, d_model)

    def forward(self, x):
        y, _ = self.lstm(x)
        return self.linear(y)


# --------------------------
# Decoder with Zoneout
# --------------------------
class Decoder(nn.Module):
    def __init__(self, d_model, mel_dim=80, zoneout_prob=0.1, p_drop=0.2):
        super().__init__()
        self.mel_dim = mel_dim
        self.pre_net_ff = 256
        self.prenet = DecoderPreNet(mel_dim, [self.pre_net_ff, self.pre_net_ff], p_drop=0.5)

        self.att_rnn = ZoneoutLSTMCell(d_model + self.pre_net_ff, d_model, zoneout_prob)
        self.dec_rnn = ZoneoutLSTMCell(d_model + d_model, d_model, zoneout_prob)

        self.att = LocationSensitiveAttention(d_model)
        self.lin_mel = nn.Linear(d_model, mel_dim)
        self.lin_stop = nn.Linear(d_model, 1)
        self.drop_in = nn.Dropout(p_drop)
        self.drop_h = nn.Dropout(p_drop)

    def _init_state(self, B, D, T_enc, device, dtype):
        zeros = lambda *s: torch.zeros(*s, device=device, dtype=dtype)
        return (zeros(B, D), zeros(B, T_enc), zeros(B, T_enc),
                zeros(B, D), zeros(B, D), zeros(B, D), zeros(B, D))

    def _step(self, enc, prev_mel, state, attn_mask):
        ctx, a, a_cum, h_att, c_att, h_dec, c_dec = state
        pm = self.prenet(prev_mel)
        h_att, c_att = self.att_rnn(self.drop_in(torch.cat([pm, ctx], -1)), (h_att, c_att))
        h_att = self.drop_h(h_att)
        ctx, a = self.att(h_att, enc, a, a_cum, mask=attn_mask)
        a_cum = a_cum + a
        h_dec, c_dec = self.dec_rnn(self.drop_in(torch.cat([h_att, ctx], -1)), (h_dec, c_dec))
        h_dec = self.drop_h(h_dec)
        mel, stop = self.lin_mel(h_dec), self.lin_stop(h_dec)
        return mel, stop, a, (ctx, a, a_cum, h_att, c_att, h_dec, c_dec)

    def forward(self, enc, mel_in, teacher_forcing=1.0, attn_mask=None, return_alignments=False):
        B, T_dec, _ = mel_in.shape
        device, dtype = enc.device, enc.dtype
        D, T_enc = enc.size(-1), enc.size(1)
        state = self._init_state(B, D, T_enc, device, dtype)

        mels, stops, aligns = [], [], []
        prev_mel = mel_in[:, 0]

        for t in range(T_dec):
            if t > 0 and torch.rand(1).item() < teacher_forcing:
                prev_mel = mel_in[:, t]
            m, s, a, state = self._step(enc, prev_mel, state, attn_mask)
            mels.append(m.unsqueeze(1)); stops.append(s.unsqueeze(1))
            if return_alignments: aligns.append(a.unsqueeze(1))
            prev_mel = m.detach()

        mel_out = torch.cat(mels, 1)
        stop_out = torch.cat(stops, 1).squeeze(-1)
        attn_out = torch.cat(aligns, 1) if return_alignments else None
        return mel_out, stop_out, attn_out


# --------------------------
# Full Model
# --------------------------
class TTSModel(nn.Module):
    def __init__(self, d_model, vocab_size, mel_dim, device):
        super().__init__()
        self.device = device
        self.emb = EmbNet(d_model, vocab_size, p_drop=0.5)
        self.encoder = Encoder(d_model, num_layers=1, dropout=0.2)
        self.decoder = Decoder(d_model, mel_dim, zoneout_prob=0.1, p_drop=0.2)
        self.post = PostNet(mel_dim, 5, 5, "instance", 0.5)
        self.to(device)

    def get_go(self, B, dtype=torch.float32):
        return torch.zeros((B, 1, self.decoder.mel_dim), device=self.device, dtype=dtype)

    def forward(self, x, mel, teacher_forcing=1.0, enc_mask=None, return_alignments=False):
        enc_out = self.encoder(self.emb(x))
        go = self.get_go(enc_out.size(0), dtype=mel.dtype)
        mel_tf = torch.cat([go, mel[:, :-1, :]], 1)
        mel_out, stop_out, attn = self.decoder(
            enc_out, mel_tf, teacher_forcing, enc_mask, return_alignments
        )
        mel_post = self.post(mel_out) + mel_out
        return mel_out, mel_post, stop_out, attn

    def inference(self, x, max_len=500, teacher_forcing=1.0, enc_mask=None, return_alignments=False):
        enc_out = self.encoder(self.emb(x))
        B, T_enc, D = enc_out.shape
        go = self.get_go(B, dtype=torch.float32).squeeze(1)
        prev_output = go
        outputs, stop_outputs = [], []
        alignments = [] if return_alignments else None

        state = self.decoder._init_state(B, D, T_enc, device=self.device, dtype=torch.float32)

        for t in range(max_len):
            mel_out, stop_out, attn, state = self.decoder._step(
                enc_out, prev_output, state, enc_mask
            )

            outputs.append(mel_out.unsqueeze(1))
            stop_outputs.append(stop_out.unsqueeze(1))

            if return_alignments:
                alignments.append(attn)

            prev_output = mel_out
            if (torch.sigmoid(stop_out) > 0.5).any().item():
                break

        mel_out = torch.cat(outputs, dim=1)  # [B, T_dec, mel_dim]
        stop_out = torch.cat(stop_outputs, dim=1)  # [B, T_dec, 1]
        mel_post = self.post(mel_out) + mel_out

        if return_alignments:
            attn = torch.stack(alignments, dim=1)  # [B, T_dec, T_enc]
            return mel_out, mel_post, stop_out, attn
        return mel_out, mel_post, stop_out


# --------------------------
# Test Run
# --------------------------
if __name__ == "__main__":
    B, T_enc, T_dec, d_model, mel_dim, vocab_size = 2, 20, 30, 128, 80, 50
    device = torch.device("cpu")

    model = TTSModel(d_model, vocab_size, mel_dim, device)
    x = torch.randint(0, vocab_size, (B, T_enc), device=device)
    mel_in = torch.randn(B, T_dec, mel_dim, device=device)

    mel_out, mel_post, stop_out, attn = model(x, mel_in, teacher_forcing=1.0, return_alignments=True)
    print("mel_out:", mel_out.shape)
    print("mel_post:", mel_post.shape)
    print("stop_out:", stop_out.shape)
    print("attn:", attn.shape if attn is not None else None)
