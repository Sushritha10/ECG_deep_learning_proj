# Personalized Arrhythmia Detection from 12-Lead ECG Signals

Demographic-aware multi-label arrhythmia classification on PTB-XL using a
dual-branch CNN + Transformer model fused with patient metadata (age, sex,
height, weight → BMI), with temporal Grad-CAM explainability.

IE7615.02 — Neural Networks and Deep Learning, Summer 2026, Northeastern University

## Team
- Deshakulakarni Srikantha, Sushritha Bharadwaj
- Prashanth Jaganathan
- Jahangir Babar

## Project layout

```
ecg-project/
├── configs/                # YAML configs for data/model/training
├── data/                   # NOT committed — see .gitignore
│   ├── raw/                # raw PTB-XL download lands here
│   └── processed/          # ETL output (npz/hdf5 + split manifests)
├── notebooks/              # EDA and result exploration
├── scripts/
│   └── download_data.sh    # fetches PTB-XL from PhysioNet
├── src/
│   ├── data/
│   │   ├── etl.py          # main ETL pipeline (this is your Week 1-2 deliverable)
│   │   └── dataset.py       # PyTorch Dataset/DataLoader over processed data
│   ├── models/              # ECG branch, demographic branch, fusion model (Week 5-6)
│   └── utils/               # metrics, Grad-CAM, fairness eval (Week 7)
├── tests/
├── environment.yml
├── requirements.txt
└── .gitignore
```

## Setup

```bash
# 1. clone
git clone https://github.com/<org>/ecg-project.git
cd ecg-project

# 2. environment
conda env create -f environment.yml
conda activate ecg-project
# or: pip install -r requirements.txt

# 3. download data (~2.5 GB, takes a while)
bash scripts/download_data.sh

# 4. run ETL
python -m src.data.etl --raw-dir data/raw --out-dir data/processed
```

## Getting the team onto GitHub

I can't create the GitHub org/repo for you directly (no GitHub access from here),
but here's the fastest path — 5 minutes:

1. **Create the repo.** One teammate goes to https://github.com/new
   - Name: `ecg-project` (or whatever you prefer)
   - Visibility: Private (recommended while it's coursework) or Public
   - Add a `.gitignore` = None (we already have one below), license = None
   - Click **Create repository**

2. **Add teammates as collaborators** (simplest option — no org needed):
   - Repo → Settings → Collaborators → "Add people" → enter each teammate's
     GitHub username or email → they accept an email invite.
   - If you'd rather have a proper GitHub *Organization* (nicer for grouping
     multiple repos, e.g. paper repo + code repo): go to
     https://github.com/organizations/new, create the org, then create the
     repo inside it and add members as owners/members from Org → People.

3. **Push this scaffold:**
   ```bash
   cd ecg-project        # this folder
   git init
   git add .
   git commit -m "Initial project scaffold + ETL pipeline"
   git branch -M main
   git remote add origin https://github.com/<org-or-user>/ecg-project.git
   git push -u origin main
   ```

4. **Branching convention** (suggested, keep it simple for a class project):
   - `main` — always working
   - `feature/etl`, `feature/ecg-branch`, `feature/demo-branch`,
     `feature/fusion-model`, `feature/gradcam` — one per teammate/component
   - Open a PR into `main` when a piece is ready; at least one other
     teammate reviews before merge.

## Milestone tracking

Use GitHub Issues + a Projects (kanban) board mapped to the proposal's
timeline:
- Week 1–2: Dataset download, EDA, preprocessing pipeline
- Week 3–4: ECG-only 1D-CNN baseline
- Week 5–6: Dual-branch fusion model + ablations
- Week 7: Grad-CAM explainability
- Week 8: Paper + slides

`Repo → Projects → New project → Board` and add one issue per milestone/row
above, assign owners, done.
