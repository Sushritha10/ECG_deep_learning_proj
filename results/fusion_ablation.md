# Dual-Branch Fusion Model — Ablation Results

Models: `src/models/fusion_model.py` (FusionModel, with `use_demo` toggle)
Trained with: `python -m src.train_fusion --epochs 30`
Device: MPS (Apple Silicon)
Data: PTB-XL, `records100` (100Hz), splits per official `strat_fold`
Train: 17,084 | Val: 2,146

## Ablation Comparison

| Model | Architecture | Best Val Macro F1 |
|---|---|---|
| CNN-only baseline | 1D-CNN | **0.6987** |
| ecg_only_transformer | 1D-CNN + Transformer, no demographics | **0.6926** |
| fusion_model | 1D-CNN + Transformer + demographics (concat fusion) | **0.6843** |

## Key finding
Within the same architecture family (Transformer variants), adding demographics via
simple concatenation slightly *decreased* val macro F1 (0.6926 -> 0.6843), rather than
improving it as hypothesized. The simplest CNN-only baseline outperformed both
Transformer variants.

## Interpretation
- This does not match the direction reported by Naeem (2026) / Atwa et al. (2025), who
  found demographic conditioning improved performance -- but both used more expressive
  fusion mechanisms (FiLM-based conditioning, graph attention) rather than simple
  concatenation of a 5-dim demographic vector onto a 128-dim ECG embedding.
- With such a large dimensionality mismatch (5 vs 128), concatenation may let the
  demographic signal get diluted or ignored, or introduce noise the model can't use well.
- The Transformer branch may also need more epochs/data than the CNN needs to fully
  converge; 30 epochs may under-train it relative to the simpler CNN.

## Next steps under consideration
- Try FiLM-style conditioning (scale/shift ECG features using demographics) instead of
  concatenation, matching Naeem's approach, as a direct comparison point
- Proceed to Grad-CAM explainability regardless of which model is used for visualization
- Evaluate fairness metrics (male-female F1 gap) across models, per proposal

## Full epoch logs

### fusion_model (use_demo=True)
Epoch 1/30  | train_loss=0.4018 train_f1=0.4863 | val_loss=0.3871 val_f1=0.5670
Epoch 2/30  | train_loss=0.3301 train_f1=0.6164 | val_loss=0.3759 val_f1=0.5773
Epoch 3/30  | train_loss=0.3117 train_f1=0.6533 | val_loss=0.3431 val_f1=0.6697
Epoch 10/30 | train_loss=0.2690 train_f1=0.7099 | val_loss=0.3267 val_f1=0.6843  <- best
Epoch 30/30 | train_loss=0.2361 train_f1=0.7496 | val_loss=0.3452 val_f1=0.6765

### ecg_only_transformer (use_demo=False)
Epoch 1/30  | train_loss=0.4008 train_f1=0.4854 | val_loss=0.3403 val_f1=0.5708
Epoch 3/30  | train_loss=0.3131 train_f1=0.6511 | val_loss=0.3136 val_f1=0.6747
Epoch 12/30 | train_loss=0.2609 train_f1=0.7191 | val_loss=0.3075 val_f1=0.6926  <- best
Epoch 30/30 | train_loss=0.2399 train_f1=0.7465 | val_loss=0.3214 val_f1=0.6750