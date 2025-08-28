import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

class ConvPostNetBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int, p: int, dropout: float = 0.2,
                 norm: str = "instance", activation: nn.Module = nn.Tanh):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=p)
        self.norm = nn.InstanceNorm1d(out_ch, affine=True)
        self.act = activation()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        return x

class PreNet(nn.Module):
    def __init__(self, d_model: int, vocab_size: int, p_drop: float = 0.2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.fc1 = nn.Linear(d_model, d_model)
        self.fc2 = nn.Linear(d_model, d_model)
        self.fc3 = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(p_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.emb(x)
        h = F.relu(self.fc1(h))
        h = self.drop(h)
        h = F.relu(self.fc2(h))
        h = self.drop(h)
        h = self.fc3(h)
        return h

class DecoderPreNet(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, p_drop: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.p = p_drop

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(x))
        h = F.dropout(h, p=self.p, training=True)
        h = F.relu(self.fc2(h))
        h = F.dropout(h, p=self.p, training=True)
        return h

class PostNet(nn.Module):
    def __init__(self, mel_dim: int, num_layers: int = 5, kernel_size: int = 5,
                 norm: str = "instance", dropout: float = 0.2):
        super().__init__()
        pads = (kernel_size // 2)
        blocks = []
        blocks.append(ConvPostNetBlock(mel_dim, mel_dim, kernel_size, pads, dropout=dropout, norm=norm))
        for _ in range(num_layers - 2):
            blocks.append(ConvPostNetBlock(mel_dim, mel_dim, kernel_size, pads, dropout=dropout, norm=norm))
        self.conv_last = nn.Conv1d(mel_dim, mel_dim, kernel_size=1, padding=0)
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.transpose(1, 2)
        for blk in self.blocks:
            y = blk(y)
        y = self.conv_last(y)
        y = y.transpose(1, 2)
        return y

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

        B, T_enc, D = memory.shape

        loc = torch.cat([prev_attn.unsqueeze(1), cum_attn.unsqueeze(1)], dim=1)
        loc = self.location_layer(loc).transpose(1, 2)

        q = self.query_layer(query).unsqueeze(1)
        m = self.memory_layer(memory)

        e = self.v(torch.tanh(q + m + loc)).squeeze(-1)

        if mask is not None:
            if not (mask.dtype == torch.bool and mask.shape == e.shape):
                raise ValueError(f"mask must be bool with shape {e.shape}, got {mask.dtype}, {mask.shape}")
            e = e.masked_fill(mask, -1e9)
            all_masked = mask.all(dim=1)
            if all_masked.any():
                e[all_masked, 0] = 0.0

        a = torch.softmax(e, dim=1)
        ctx = torch.bmm(a.unsqueeze(1), memory).squeeze(1)
        return ctx, a

class Encoder(nn.Module):
    def __init__(self, d_model: int, num_layers: int = 1, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(d_model, d_model, num_layers=num_layers, bidirectional=True,
                             batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.linear = nn.Linear(d_model * 2, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.lstm(x)
        y = self.linear(y)
        return y

class Decoder(nn.Module):
    def __init__(self, d_model: int, mel_dim: int = 80, p_drop: float = 0.2):
        super().__init__()
        self.mel_dim = mel_dim
        self.prenet = DecoderPreNet(mel_dim, d_model, p_drop=p_drop)
        self.att_rnn = nn.LSTMCell(d_model + d_model, d_model)
        self.dec_rnn = nn.LSTMCell(d_model + d_model, d_model)
        self.att = LocationSensitiveAttention(d_model)
        self.lin_mel = nn.Linear(d_model, mel_dim)
        self.lin_stop = nn.Linear(d_model, 1)
        self.drop_in = nn.Dropout(p_drop)
        self.drop_h = nn.Dropout(p_drop)

    def _init_state(self, B: int, D: int, T_enc: int, device, dtype):
        zeros = lambda *s: torch.zeros(*s, device=device, dtype=dtype)
        ctx = zeros(B, D)
        a = zeros(B, T_enc)
        a_cum = zeros(B, T_enc)
        h_att = zeros(B, D); c_att = zeros(B, D)
        h_dec = zeros(B, D); c_dec = zeros(B, D)
        return (ctx, a, a_cum, h_att, c_att, h_dec, c_dec)

    def _step(self, enc, prev_mel, state, attn_mask):
        ctx, a, a_cum, h_att, c_att, h_dec, c_dec = state

        pm = self.prenet(prev_mel)

        att_in = torch.cat([pm, ctx], dim=-1)
        att_in = self.drop_in(att_in)
        h_att, c_att = self.att_rnn(att_in, (h_att, c_att))
        h_att = self.drop_h(h_att)

        ctx, a = self.att(h_att, enc, a, a_cum, mask=attn_mask)
        a_cum = a_cum + a

        dec_in = torch.cat([h_att, ctx], dim=-1)
        dec_in = self.drop_in(dec_in)
        h_dec, c_dec = self.dec_rnn(dec_in, (h_dec, c_dec))
        h_dec = self.drop_h(h_dec)

        mel = self.lin_mel(h_dec)
        stop = torch.sigmoid(self.lin_stop(h_dec))

        new_state = (ctx, a, a_cum, h_att, c_att, h_dec, c_dec)
        return mel, stop, a, new_state

    def forward(self, enc, mel_in, teacher_forcing=1.0,
                attn_mask=None, return_alignments=False):
        B, T_dec, _ = mel_in.shape
        device, dtype = enc.device, enc.dtype
        D, T_enc = enc.size(-1), enc.size(1)
        state = self._init_state(B, D, T_enc, device, dtype)

        mels, stops, alignments = [], [], []
        prev_mel = mel_in[:, 0]

        for t in range(T_dec):
            if (t > 0) and (torch.rand((), device=device) < teacher_forcing):
                prev_mel = mel_in[:, t]

            m, s, a, state = self._step(enc, prev_mel, state, attn_mask)
            mels.append(m.unsqueeze(1))
            stops.append(s.unsqueeze(1))
            if return_alignments:
                alignments.append(a.unsqueeze(1))
            prev_mel = m.detach()

        mel_out = torch.cat(mels, dim=1)
        stop_out = torch.cat(stops, dim=1).squeeze(-1)
        attn_out = torch.cat(alignments, dim=1) if return_alignments else None
        return mel_out, stop_out, attn_out

    @torch.no_grad()
    def infer(self, enc, go_frame, max_len=1000, attn_mask=None, return_alignments=False):
        device, dtype = enc.device, enc.dtype
        B, T_enc, D = enc.shape
        state = self._init_state(B, D, T_enc, device, dtype)
        prev = go_frame
        outputs, stops, aligns = [], [], []

        for _ in range(max_len):
            mel, s, a, state = self._step(enc, prev, state, attn_mask)
            outputs.append(mel.unsqueeze(1))
            stops.append(s.unsqueeze(1))
            if return_alignments:
                aligns.append(a.unsqueeze(1))
            prev = mel

        mel_out = torch.cat(outputs, dim=1)
        stop_out = torch.cat(stops, dim=1).squeeze(-1)
        attn = torch.cat(aligns, dim=1) if (return_alignments and aligns) else None
        return mel_out, stop_out, attn

class TTSModel(nn.Module):
    def __init__(self, d_model, vocab_size, mel_dim, device):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.mel_dim = mel_dim
        self.device = device
        self.pre = PreNet(d_model, vocab_size, p_drop=0.2)
        self.encoder = Encoder(d_model, num_layers=1, dropout=0.2)
        self.decoder = Decoder(d_model, mel_dim, p_drop=0.2)
        self.post = PostNet(mel_dim, num_layers=5, kernel_size=5,
                            norm="instance", dropout=0.2)
        self.to(device)

    def get_go(self, B, dtype=torch.float32):
        return torch.zeros((B, 1, self.mel_dim), device=self.device, dtype=dtype)

    def forward(self, x, mel, teacher_forcing=1.0,
                enc_mask=None, return_alignments=False):
        enc_in = self.pre(x)
        enc_out = self.encoder(enc_in)

        go = self.get_go(enc_out.size(0), dtype=mel.dtype)
        mel_tf = torch.cat([go, mel[:, :-1, :]], dim=1)

        mel_out, stop_out, attn = self.decoder(
            enc_out, mel_tf, teacher_forcing=teacher_forcing,
            attn_mask=enc_mask, return_alignments=return_alignments
        )

        mel_post = self.post(mel_out) + mel_out
        return mel_out, mel_post, stop_out, attn

    @torch.no_grad()
    def inference(self, x, max_len=1000, enc_mask=None, return_alignments=False):
        self.eval()
        x = x.to(self.device)

        enc_in = self.pre(x)
        enc_out = self.encoder(enc_in)
        B = enc_out.size(0)

        go = self.get_go(B, dtype=enc_out.dtype).squeeze(1)

        mel_out, stop_out, attn = self.decoder.infer(
            enc_out, go, max_len=max_len,
            attn_mask=enc_mask, return_alignments=return_alignments
        )

        mel_post = self.post(mel_out) + mel_out
        return mel_out, mel_post, stop_out, attn