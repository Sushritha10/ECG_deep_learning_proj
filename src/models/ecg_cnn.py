"""
ECG-only 1D-CNN baseline model.

This is deliberately the SIMPLEST branch of the eventual dual-branch
architecture (proposal ablation (a): "ECG-only baseline"). It ignores
demographic metadata entirely and just learns from the 12-lead waveform.

Input : (batch, 12, 1000)  -- 12 leads, 1000 samples (10s @ 100Hz)
Output: (batch, 5)         -- raw logits for [NORM, MI, STTC, CD, HYP]
                              (apply sigmoid outside, or use BCEWithLogitsLoss)
"""
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Conv1d -> BatchNorm -> ReLU -> MaxPool, with dropout for regularization."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 7,
                 pool_size: int = 2, dropout: float = 0.2):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=kernel_size // 2,  # 'same' padding
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(pool_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        return x


class ECGCNNBaseline(nn.Module):
    """
    5-block 1D-CNN over raw 12-lead ECG, global average pooled, then an
    MLP classification head with sigmoid (multi-label) output.
    """

    def __init__(
        self,
        n_leads: int = 12,
        n_classes: int = 5,
        channels: tuple = (32, 64, 128, 128, 256),
        kernel_size: int = 7,
        dropout: float = 0.3,
    ):
        super().__init__()

        blocks = []
        in_ch = n_leads
        for out_ch in channels:
            blocks.append(ConvBlock(in_ch, out_ch, kernel_size=kernel_size, dropout=dropout))
            in_ch = out_ch
        self.conv_blocks = nn.Sequential(*blocks)

        # Global average pooling over the time dimension -> (batch, channels[-1])
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Linear(channels[-1], 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, ecg: torch.Tensor) -> torch.Tensor:
        """
        ecg: (batch, n_leads, signal_length)
        returns: (batch, n_classes) raw logits
        """
        x = self.conv_blocks(ecg)          # (batch, channels[-1], T')
        x = self.global_pool(x).squeeze(-1)  # (batch, channels[-1])
        logits = self.classifier(x)         # (batch, n_classes)
        return logits


if __name__ == "__main__":
    # quick shape sanity check
    model = ECGCNNBaseline()
    dummy = torch.randn(8, 12, 1000)  # batch of 8
    out = model(dummy)
    print("Output shape:", out.shape)  # expect (8, 5)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")
