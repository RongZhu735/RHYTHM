# RHYTHM

Official implementation of **Beyond Static Temporal Structures: Discovering Temporal Periodicity for Sequential Recommendation**(ICDM 2026).

RHYTHM is a sequential recommendation framework that discovers personalized temporal contexts from user behavior. It projects interaction timestamps into a cyclic phase space, hierarchically identifies semantically coherent temporal regions, and iteratively refreshes the discovered mappings during training. The resulting periodic structure guides attention toward contextually congruent historical interactions, reducing temporal noise and improving the modeling of user preferences.

The repository provides the implementation, training and evaluation scripts, and supplementary experimental analysis.

## Train

Install dependencies first:

```bash
pip install -r Rhythm/requirements.txt
```

Run the training script:

```bash
bash Rhythm/run.sh
```

Alternatively, train with explicit arguments:

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

Supported inputs are CSV, TSV, and RecBole `.inter` files with the following fields:

* `user_id`
* `product_id` or `item_id`
* Optional `timestamp`
* Optional time column: `hour`, `month`, or `day_of_week`

The data split follows the leave-one-out protocol: the last two interactions of each user are used for validation and testing, respectively, and all earlier interactions are used for training.

## Checkpoints and Evaluation

Evaluate a saved checkpoint:

```bash
python -m Rhythm.main \
  --dataset Rhythm/data/DHRD.csv \
  --train_dir rhythm_eval \
  --checkpoint Rhythm/model/DHRD/best_model.pt \
  --eval_only True
```

Ready-to-run evaluation scripts are provided for the three bundled datasets and checkpoints:

```bash
bash Rhythm/test_eval_dhrd.sh
bash Rhythm/test_eval_megamarket.sh
bash Rhythm/test_eval_tafeng.sh
```

These scripts use full-item evaluation. The deprecated `candidate_num` argument is ignored if passed through backward-compatible command lines.

## Supplementary Analysis

Additional experiments on robustness under short interaction histories are available in the supplementary folder.

