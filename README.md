# Rhythm

`Rhythm` is a compact implementation of the paper-relevant temporal
segmentation recommendation pipeline. It keeps the core training and evaluation
path and removes deprecated experimental branches:

- Time mask is always enabled for training/evaluation.
- Causal mask is always enabled for training/evaluation.
- LLM embedding, float16 embedding generation, attention-weight dumping, attention parameter cloning, and warmup checkpoint branches are not implemented.
- Backward-compatible checkpoints are loaded by matching `state_dict` keys and tensor shapes.

## Train

Install dependencies first:

```bash
pip install -r Rhythm/requirements.txt
```

```bash
bash Rhythm/run.sh
```

or:

```bash
python -m Rhythm.main \
  --dataset Rhythm/data/DHRD.csv \
  --train_dir sasrec_fixedtimemask_cross \
  --maxlen 50 \
  --hidden_units 128 \
  --num_blocks 1 \
  --num_heads 4 \
  --time_type hour \
  --use_cross True \
  --use_fixed True \
  --loss_type full_ce
```

## Dataset Format

Supported inputs are CSV/TSV/RecBole `.inter` files with:

- `user_id`
- `product_id` or `item_id`
- optional `timestamp`
- optional time column: `hour`, `month`, or `day_of_week`

The split follows the leave-one-out protocol: the last two interactions per
user are validation and test; all earlier interactions are training.

## Checkpoints

Evaluate a saved checkpoint:

```bash
python -m Rhythm.main \
  --dataset Rhythm/data/DHRD.csv \
  --train_dir rhythm_eval \
  --checkpoint Rhythm/model/DHRD/best_model.pt \
  --eval_only True
```

The repository also includes ready-to-run evaluation scripts for the three
bundled datasets/checkpoints:

```bash
bash Rhythm/test_eval_dhrd.sh
bash Rhythm/test_eval_megamarket.sh
bash Rhythm/test_eval_tafeng.sh
```

These scripts always use full-item evaluation. The deprecated `candidate_num`
argument is ignored if passed through backward-compatible command lines.

## Case Study

Generate segmentation case-study figures for all bundled datasets:

```bash
bash Rhythm/run_case_study.sh
```

For a faster dry run, scan only the first 100 users and relax the rank filter:

```bash
MAX_SCAN_USERS=100 RANK_THRESHOLD=100 MAX_CASES=2 bash Rhythm/run_case_study.sh
```

By default, case users are prioritized by temporal coverage: wider active
time-unit span first, then more active time units, then more history
interactions. To sample users randomly instead:

```bash
DATASET_NAME=DHRD USER_SELECTION=random RANDOM_POOL_SIZE=2000 RANK_THRESHOLD=100 MAX_CASES=5 bash Rhythm/run_case_study.sh
```

Each dataset writes one fused PNG overview plus per-case JSON and CSV tables
under `Rhythm/case_study_outputs/`. In the overview, rows are cases, columns are
active discovered segments sorted by target similarity, and the right-side rings
show the corresponding time-unit segmentation. Time units with no history item
are colored gray and are not used as segment explanations.

For paper-ready combined PDF figures with larger fonts:

```bash
python -m Rhythm.plot_case_study_paper --layout both
```

This writes `case_study_paper_vertical.pdf` and
`case_study_paper_horizontal.pdf` under `Rhythm/case_study_outputs/`. The
vertical version is recommended when readability is more important than keeping
all three datasets on one row.
