"""Arquitecturas: LSTM bidireccional (v0) y Transformer encoder (v1 / v2)."""
from __future__ import annotations

import torch
import torch.nn as nn

from .features import N_FEAT, T_FIJO


class LSTMClasificador(nn.Module):
    """v0 — LSTM bidireccional de 2 capas; se promedia la salida en el tiempo (many-to-one)."""

    def __init__(self, n_clases: int, hidden: int = 192, capas: int = 2, dropout: float = 0.3):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(N_FEAT, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.lstm = nn.LSTM(hidden, hidden, capas, batch_first=True, bidirectional=True,
                            dropout=dropout if capas > 1 else 0)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(2 * hidden, n_clases)

    def embed(self, x):
        h, _ = self.lstm(self.inp(x))
        return self.drop(h.mean(1))

    def forward(self, x):
        return self.head(self.embed(x))


class TransformerClasificador(nn.Module):
    """v1 — Transformer encoder con token [CLS] y embeddings posicionales aprendidos.

    La atención permite relacionar cuadros lejanos (inicio y fin de la seña), algo que en
    la LSTM tiene que pasar por toda la cadena recurrente.
    """

    def __init__(self, n_clases: int, d: int = 192, capas: int = 4, heads: int = 4,
                 dropout: float = 0.2, t: int = T_FIJO):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(N_FEAT, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(dropout))
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.pos = nn.Parameter(torch.randn(1, t + 1, d) * 0.02)
        enc = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, batch_first=True,
                                         activation="gelu", norm_first=True)
        self.enc = nn.TransformerEncoder(enc, capas, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, n_clases)

    def embed(self, x):
        h = self.inp(x)
        h = torch.cat([self.cls.expand(len(h), -1, -1), h], 1) + self.pos[:, : h.shape[1] + 1]
        return self.norm(self.enc(h)[:, 0])

    def forward(self, x):
        return self.head(self.embed(x))

    def atencion_cls(self, x):
        """Atención promedio del [CLS] hacia cada cuadro en la última capa (explicabilidad)."""
        h = self.inp(x)
        h = torch.cat([self.cls.expand(len(h), -1, -1), h], 1) + self.pos[:, : h.shape[1] + 1]
        for i, layer in enumerate(self.enc.layers):
            if i == len(self.enc.layers) - 1:
                z = layer.norm1(h)
                _, w = layer.self_attn(z, z, z, need_weights=True, average_attn_weights=True)
                return w[:, 0, 1:]
            h = layer(h)


def construir(arq: str, n_clases: int, **kw) -> nn.Module:
    if arq == "lstm":
        return LSTMClasificador(n_clases, **kw)
    if arq == "transformer":
        return TransformerClasificador(n_clases, **kw)
    raise ValueError(arq)
