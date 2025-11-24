# Latin — Language Model Training & Contrastive Learning

This repository contains code, configuration, and scripts for training Masked Language Models (MLM) and Contrastive (SBERT-style) models on language data. The project includes training pipelines, inference code, and utilities to build and evaluate multilingual/sentence embedding models.

## Quick Overview
- `mlm_pretrain.py` / `train_mlm_sbert.sh` — Pretrain a masked language model or Sentence-BERT style model.
- `contrastive.py` / `ctrain.sh` — Train contrastive or SBERT-like models using MLM representations as a starting point.
- `contrastive.py` / `ctrain.sh` — Train contrastive or SBERT-like models using MLM representations as a starting point. `ctrain.sh` is a convenience wrapper; configuration and model paths are provided via `config/contrastive.yaml` or hydra overrides.
- `canon_trainer.py` / `catrain.sh` — A trainer and evaluation workflow used for experiments that combines pretraining, contrastive training (optional), and evaluation. `catrain.sh` is a wrapper that runs the `canon_trainer.py` entrypoint and uses `config/canon_trainer.yaml`.
- `inference.py` — Run inference using a trained SentenceTransformer compatible model (e.g., from Hugging Face).
- `create_data.py` / `create_unkeyed_data.py` — Scripts to prepare datasets used by different training workflows.
- `outputs/` — Saved model checkpoints, results and logs.

## Prerequisites
- Python 3.8+ recommended
- Install dependencies with conda or pip. The repository includes `environment.yml` and `requirements.txt` for reference.

Conda example:
```bash
conda env create -f environment.yml -n latin-env
conda activate latin-env
pip install -r requirements.txt
```

## Configuration
Configurations are found in the `config/` folder. The most commonly used files:
- `mlm_bert.yaml` / `mlm_bert_xlm.yaml` — Configs for MLM pretraining
- `contrastive.yaml` — Config for contrastive training workflow
- `inference.yaml` — Config to run model inference
- `unkeyed_trainer.yaml`, `canon_trainer.yaml`, `supervised_topic.yaml`, `topic.yaml` — Additional training tasks

Important: Many configs contain absolute or project-root paths. You should change `project_root` and other path settings to match your environment before running scripts.

## Running Training & Pretrain

1) Pretrain MLM (optional) — use `train_mlm_sbert.sh` which calls `mlm_pretrain.py`:
```bash
# Example: start pretraining
./train_mlm_sbert.sh config/mlm_bert.yaml
```

2) Contrastive training using SBERT:
```bash
# Example: start contrastive training
./ctrain.sh \
    /path/to/mlm_checkpoint \
    /path/to/contrastive_checkpoint
```
Notes: `ctrain.sh` supports `num_files` to filter training folders by number of documents. Use `-1` for no filtering. The contrastive training config can be adjusted in `config/contrastive.yaml` to set `model_path`/`model_name`, `out_path`, `num_files`, and other training hyperparameters.

Inputs and behavior for contrastive training:
- The code reads model paths and other parameters from `config/contrastive.yaml` (or hydra overrides). To use a specific MLM checkpoint, set the `model_path` field in the config or override it with `python contrastive.py model_path=/path/to/mlm_checkpoint`.
- `ctrain.sh` itself does not parse args—if you need to run with different models or paths, either edit `config/contrastive.yaml` or use hydra CLI overrides as described above.

Example hydra overrides (contrastive):
```bash
# Use a specific model path (overrides config/contrastive.yaml)
python contrastive.py model_path=/path/to/mlm_checkpoint
```

## Inference
The `inference.py` script can load local models or models directly from Hugging Face if they are compatible with the SentenceTransformer interface. Edit `config/inference.yaml` to configure the dataset and model. Note: `inference.py` has been integrated into `canon_trainer.py` for scripted experiments (see `catrain.sh` and `config/canon_trainer.yaml`).

Usage example:
```bash
python inference.py --config config/inference.yaml --model ./outputs/2024-03-29/23-16-06/pretrain
```
If you want to load a model from the Hugging Face hub, set the `model` path to a HF identifier and make sure it is compatible with `sentence-transformers`.

## Data Preparation
- To prepare your datasets, use `create_data.py` or `create_unkeyed_data.py` depending on the format required.
- Check `config/*` for data path entries and adapt them to your local filesystem structure.

CaTrain and `canon_trainer.py` specific notes:
- `catrain.sh` runs `canon_trainer.py`, which orchestrates a full pipeline used for many experiments. The `canon_trainer.py` config (`config/canon_trainer.yaml`) specifies whether to run ML pretraining, contrastive training (via `contrastive_model_path`), data creation (it can call `create_data.py`), and inference.
- `canon_trainer.py` will perform contrastive training if `contrastive_model_path` is not provided (or controlled via config).

Example: run a full experiment with `catrain.sh` (uses `config/canon_trainer.yaml`):
```bash
./catrain.sh
```
If you prefer hydra overrides:
```bash
# Example: select a different mlm model path for the run
python canon_trainer.py mlm_model_path=/path/to/mlm_checkpoint
```

## Outputs & Logs
- Trained models and artifacts are stored in the `outputs/` directory. Each run typically contains `pretrain/`, `train/`, `results/`, and logs saved under `wandb/` or similar.

## Helpful Tips
- Check `config/<...>.yaml` files for the `project_root` and `data` fields — these are commonly required edits.
- Use `num_files` in `ctrain.sh` to control dataset filtering per-folder.
- For inference with HF models, ensure you choose an HF repo that provides a Sentence-BERT style saved configuration.

## Troubleshooting
- If model loading fails, set `--no-cache-dir` or use a valid model path.
- If training unexpectedly terminates, inspect `wandb/` logs or `outputs/*/train/config.json`.

## Contact
If you run into problems or have questions, open an issue or contact the repository maintainer.


