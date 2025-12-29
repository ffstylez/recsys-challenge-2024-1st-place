# Execution Process for RecSys Challenge 2024 Pipeline

## Overview
The pipeline consists of **3 sequential stages** that must be executed in order due to dependencies between components.

## Execution Order

### Stage 1: KAMI Pipeline (Feature Engineering + GBDT)
**VM**: `kami-vm` (n2-highmem-96)
**Duration**: ~6-8 hours
**Dependencies**: Raw dataset only

```bash
# 1. Start KAMI VM
gcloud compute instances start kami-vm --zone=us-central1-a
gcloud compute ssh kami-vm

# 2. Download datasets
mkdir -p input output
# Download ebnerd_large and ebnerd_testset to input/

# 3. Run KAMI pipeline
docker compose -f compose.cpu.yaml build
docker compose -f compose.cpu.yaml run --rm kaggle bash

# Inside container:
inv create-candidates    # ~30 minutes
inv create-features      # ~4-5 hours  
inv create-datasets      # ~30 minutes
inv train               # ~1-2 hours

# 4. Upload outputs
gsutil -m cp -r output/ gs://recsys-2024-artifacts/kami/

# 5. Stop VM
exit
gcloud compute instances stop kami-vm --zone=us-central1-a
```

### Stage 2: KFujikawa Pipeline (Neural Networks)
**VM**: `kfujikawa-vm` (g2-standard-32)
**Duration**: ~8-12 hours
**Dependencies**: Raw dataset only

```bash
# 1. Start KFujikawa VM
gcloud compute instances start kfujikawa-vm --zone=us-central1-a
gcloud compute ssh kfujikawa-vm

# 2. Setup and run pipeline
git clone [repository]
cd kfujikawa
pip install -U poetry
poetry install

# 3. Execute pipeline stages
./src/exp/v0xxx_preprocess/run.sh        # ~2 hours
./src/exp/v1xxx_training/run_scratch.sh   # ~6-8 hours
./src/exp/v8xxx_ensemble/run_ensemble.sh  # ~30 minutes
./src/exp/v1xxx_training/run_pl.sh        # ~2-3 hours

# 4. Upload outputs
gsutil -m cp -r data/ gs://recsys-2024-artifacts/kfujikawa/

# 5. Stop VM
gcloud compute instances stop kfujikawa-vm --zone=us-central1-a
```

### Stage 3: Sugawarya Pipeline (Ensemble + Submission)
**VM**: `sugawarya-vm` (c2d-highmem-112)  
**Duration**: ~2-4 hours
**Dependencies**: Outputs from both KAMI and KFujikawa

```bash
# 1. Start Sugawarya VM
gcloud compute instances start sugawarya-vm --zone=us-central1-a
gcloud compute ssh sugawarya-vm

# 2. Download all required inputs
gsutil -m cp -r gs://recsys-2024-artifacts/kami/output/ ./kami/
gsutil -m cp -r gs://recsys-2024-artifacts/kfujikawa/data/ ./kfujikawa/
# Download original datasets for target labels

# 3. Setup and run ensemble
git clone [repository]
cd sugawarya
pip install -U poetry
poetry install

# 4. Execute ensemble pipeline
poetry run python src/weighted_mean.py   # ~1 hour (Bayesian optimization)
poetry run python src/stacking.py        # ~2-3 hours (meta-learning)
poetry run python src/make_submission.py # ~30 minutes

# 5. Download final submission
gsutil cp output/v999_final_submission.zip gs://recsys-2024-artifacts/final/

# 6. Stop VM
gcloud compute instances stop sugawarya-vm --zone=us-central1-a
```

## Critical Dependencies

### Data Flow
```
Raw Dataset → KAMI → Feature-engineered predictions
Raw Dataset → KFujikawa → Neural network predictions
Both Predictions → Sugawarya → Final ensemble submission
```

### Required Files for Sugawarya
**From KAMI**:
- `kami/output/experiments/015_train_third/large067_001/validation_result_first.parquet`
- `kami/output/experiments/015_train_third/large067_001/test_result_third.parquet`
- `kami/output/experiments/016_catboost/large067/validation_result_first.parquet`
- `kami/output/experiments/016_catboost/large067/test_result_third.parquet`

**From KFujikawa**:
- `kfujikawa/data/kfujikawa/v1xxx_training/v1157_111_fix_past_v2/fold_0/predictions/validation.parquet`
- `kfujikawa/data/kfujikawa/v1xxx_training/v1157_111_fix_past_v2/fold_2/predictions/test.parquet`
- `kfujikawa/data/kfujikawa/v1xxx_training/v1170_111_L8_128d/fold_{0,2}/predictions/`
- `kfujikawa/data/kfujikawa/v1xxx_training/v1184_111_PL_bert_L4_256d/fold_{0,2}/predictions/`

## Debug Mode
For testing with smaller datasets, add `--debug` flag to all commands:
```bash
# KAMI
inv create-candidates --debug
inv create-features --debug

# Sugawarya  
poetry run python src/weighted_mean.py --debug
poetry run python src/stacking.py --debug
```

## Estimated Total Runtime
- **KAMI**: 6-8 hours
- **KFujikawa**: 8-12 hours  
- **Sugawarya**: 2-4 hours
- **Total**: 16-24 hours (can run KAMI and KFujikawa in parallel)

## Cost Optimization
1. **Parallel Execution**: Run KAMI and KFujikawa simultaneously (no dependencies)
2. **Immediate Shutdown**: Stop VMs immediately after each stage
3. **Storage Management**: Delete intermediate files from bucket after final submission
4. **Debug Testing**: Always test with `--debug` mode first to validate setup

## Final Deliverable
- `v999_final_submission.zip` containing competition-format predictions
- Expected AUC performance: ~0.8791 (based on validation results)