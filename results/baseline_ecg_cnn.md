# ECG-Only 1D-CNN Baseline — Training Results

Model: `src/models/ecg_cnn.py` (ECGCNNBaseline)
Trained with: `python -m src.train_baseline --epochs 30`
Device: MPS (Apple Silicon)
Data: PTB-XL, `records100` (100Hz), splits per official `strat_fold`
Train: 17,084 | Val: 2,146

## Result
**Best val macro F1: 0.6987** (epoch 23)
Checkpoint: `checkpoints/ecg_cnn_baseline.pt` (not committed to git — regenerable via training script)

## Observations
- Train F1 climbed steadily (0.54 -> 0.76) over 30 epochs
- Val F1 plateaued around epoch 14-23 (~0.69-0.70), with train/val gap widening slightly after epoch ~15 -> mild overfitting, not severe
- This serves as the ablation (a) "ECG-only" baseline per the project proposal, to be compared against the full fusion model (ECG Transformer + demographics)

## Full epoch log
Epoch 1/30  | train_loss=0.3668 train_f1=0.5406 | val_loss=0.3538 val_f1=0.5791
Epoch 2/30  | train_loss=0.3176 train_f1=0.6357 | val_loss=0.3421 val_f1=0.6086
Epoch 3/30  | train_loss=0.3042 train_f1=0.6679 | val_loss=0.3118 val_f1=0.6511
Epoch 4/30  | train_loss=0.2955 train_f1=0.6797 | val_loss=0.3091 val_f1=0.6536
Epoch 5/30  | train_loss=0.2907 train_f1=0.6866 | val_loss=0.3157 val_f1=0.6613
Epoch 6/30  | train_loss=0.2848 train_f1=0.6962 | val_loss=0.3187 val_f1=0.6441
Epoch 7/30  | train_loss=0.2818 train_f1=0.6957 | val_loss=0.3166 val_f1=0.6589
Epoch 8/30  | train_loss=0.2768 train_f1=0.7058 | val_loss=0.3244 val_f1=0.6574
Epoch 9/30  | train_loss=0.2736 train_f1=0.7070 | val_loss=0.2984 val_f1=0.6978
Epoch 10/30 | train_loss=0.2716 train_f1=0.7110 | val_loss=0.3070 val_f1=0.6620
Epoch 11/30 | train_loss=0.2677 train_f1=0.7188 | val_loss=0.3113 val_f1=0.6857
Epoch 12/30 | train_loss=0.2663 train_f1=0.7172 | val_loss=0.3047 val_f1=0.6764
Epoch 13/30 | train_loss=0.2641 train_f1=0.7220 | val_loss=0.3114 val_f1=0.6807
Epoch 14/30 | train_loss=0.2538 train_f1=0.7292 | val_loss=0.3006 val_f1=0.6892
Epoch 15/30 | train_loss=0.2510 train_f1=0.7340 | val_loss=0.3388 val_f1=0.6795
Epoch 16/30 | train_loss=0.2499 train_f1=0.7383 | val_loss=0.3173 val_f1=0.6885
Epoch 17/30 | train_loss=0.2470 train_f1=0.7415 | val_loss=0.3086 val_f1=0.6777
Epoch 18/30 | train_loss=0.2420 train_f1=0.7477 | val_loss=0.3204 val_f1=0.6932
Epoch 19/30 | train_loss=0.2393 train_f1=0.7508 | val_loss=0.3146 val_f1=0.6935
Epoch 20/30 | train_loss=0.2382 train_f1=0.7524 | val_loss=0.3244 val_f1=0.6889
Epoch 21/30 | train_loss=0.2375 train_f1=0.7509 | val_loss=0.3109 val_f1=0.6900
Epoch 22/30 | train_loss=0.2338 train_f1=0.7557 | val_loss=0.3225 val_f1=0.6927
Epoch 23/30 | train_loss=0.2317 train_f1=0.7594 | val_loss=0.3182 val_f1=0.6987  <- best
Epoch 24/30 | train_loss=0.2312 train_f1=0.7601 | val_loss=0.3208 val_f1=0.6962
Epoch 25/30 | train_loss=0.2306 train_f1=0.7578 | val_loss=0.3199 val_f1=0.6969
Epoch 26/30 | train_loss=0.2299 train_f1=0.7618 | val_loss=0.3271 val_f1=0.6923
Epoch 27/30 | train_loss=0.2277 train_f1=0.7608 | val_loss=0.3214 val_f1=0.6975
Epoch 28/30 | train_loss=0.2285 train_f1=0.7619 | val_loss=0.3194 val_f1=0.6946
Epoch 29/30 | train_loss=0.2262 train_f1=0.7665 | val_loss=0.3255 val_f1=0.6971
Epoch 30/30 | train_loss=0.2262 train_f1=0.7639 | val_loss=0.3250 val_f1=0.6984