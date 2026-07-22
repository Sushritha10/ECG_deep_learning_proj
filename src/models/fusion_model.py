"""
Dual-branch demographic-aware arrhythmia classifier.

Architecture (per proposal):
  ECG Branch         : 1D-CNN stem -> Transformer encoder -> pooled latent vector
  Demographic Branch : small MLP over [age, sex, height, weight, bmi] -> embedding
  Fusion             : concat(ecg_latent, demo_embedding) -> MLP classification head
  Output             : 5 raw logits (sigmoid outside / BCEWithLogitsLoss)

Input shapes:
  ecg  : (batch, 12, 1000)
  demo : (batch, 5)          -- age, sex, height, weight, bmi (already standardized in ETL)
Output:
  logits: (batch, 5)
"""
import math

import torch
import torch.nn as nn

from src.models.ecg_cnn import ConvBlock


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding, added to the CNN stem's output sequence
    before it goes into the Transformer encoder."""

    def __init__(self, d_model: int, max_len: int = 2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        return x + self.pe[:, : x.size(1), :]


class ECGTransformerEncoder(nn.Module):
    """
    CNN stem (downsamples 1000 -> ~125 timesteps while expanding channels),
    followed by a Transformer encoder over the resulting sequence, then
    mean-pooled into a single latent vector per recording.
    """

    def __init__(
        self,
        n_leads: int = 12,
        cnn_channels: tuple = (32, 64, 128),
        d_model: int = 128,
        n_heads: int = 4,
        n_transformer_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()

        blocks = []
        in_ch = n_leads
        for out_ch in cnn_channels:
            blocks.append(ConvBlock(in_ch, out_ch, kernel_size=7, dropout=dropout))
            in_ch = out_ch
        self.cnn_stem = nn.Sequential(*blocks)  # (batch, cnn_channels[-1], T') , T' = 1000 / 2^len(cnn_channels)

        assert cnn_channels[-1] == d_model, (
            "Last CNN channel must equal d_model (no projection layer needed); "
            "adjust cnn_channels[-1] or add a projection if you change this."
        )

        self.pos_encoding = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_transformer_layers)

    def forward(self, ecg: torch.Tensor) -> torch.Tensor:
        """
        ecg: (batch, n_leads, signal_length)
        returns: (batch, d_model) pooled latent vector
        """
        x = self.cnn_stem(ecg)              # (batch, d_model, T')
        x = x.transpose(1, 2)                # (batch, T', d_model) -- Transformer expects seq first
        x = self.pos_encoding(x)
        x = self.transformer(x)              # (batch, T', d_model)
        latent = x.mean(dim=1)               # mean pool over time -> (batch, d_model)
        return latent


class DemographicEncoder(nn.Module):
    """Small MLP embedding for [age, sex, height, weight, bmi]."""

    def __init__(self, in_features: int = 5, hidden: int = 32, out_features: int = 16, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_features),
            nn.ReLU(inplace=True),
        )

    def forward(self, demo: torch.Tensor) -> torch.Tensor:
        return self.net(demo)


class FusionModel(nn.Module):
    """
    Full dual-branch model: ECG (CNN+Transformer) + Demographics (MLP) -> fusion -> classifier.

    Set `use_demo=False` to run in "ECG-only" mode using the SAME architecture
    (handy for a fair ablation comparison against the CNN-only baseline, since
    this ablates just the demographic branch rather than swapping model families).
    """

    def __init__(
        self,
        n_leads: int = 12,
        n_classes: int = 5,
        demo_features: int = 5,
        ecg_d_model: int = 128,
        demo_embed_dim: int = 16,
        use_demo: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.use_demo = use_demo

        self.ecg_encoder = ECGTransformerEncoder(n_leads=n_leads, d_model=ecg_d_model, dropout=dropout)

        fusion_in = ecg_d_model
        if use_demo:
            self.demo_encoder = DemographicEncoder(in_features=demo_features, out_features=demo_embed_dim, dropout=dropout)
            fusion_in += demo_embed_dim

        self.classifier = nn.Sequential(
            nn.Linear(fusion_in, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, ecg: torch.Tensor, demo: torch.Tensor = None) -> torch.Tensor:
        ecg_latent = self.ecg_encoder(ecg)  # (batch, ecg_d_model)

        if self.use_demo:
            assert demo is not None, "use_demo=True but no demo tensor was passed"
            demo_embed = self.demo_encoder(demo)  # (batch, demo_embed_dim)
            fused = torch.cat([ecg_latent, demo_embed], dim=1)
        else:
            fused = ecg_latent

        logits = self.classifier(fused)
        return logits


if __name__ == "__main__":
    # quick shape sanity check
    batch_size = 8
    dummy_ecg = torch.randn(batch_size, 12, 1000)
    dummy_demo = torch.randn(batch_size, 5)

    model = FusionModel(use_demo=True)
    out = model(dummy_ecg, dummy_demo)
    print("Fusion model output shape:", out.shape)  # expect (8, 5)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Fusion model params: {n_params:,}")

    ecg_only_model = FusionModel(use_demo=False)
    out2 = ecg_only_model(dummy_ecg)
    print("ECG-only (same arch) output shape:", out2.shape)  # expect (8, 5)