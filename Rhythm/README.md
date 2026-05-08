# Rhythm

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

### Overall Performance Comparison

\* denotes statistical significance of $p < 0.01$ for the comparison with the best baseline.

| Dataset | Metric | SASRec | BERT4Rec | Caser | TokenRec | TiSASRec | MEAN TIME | TCPSRec | FEARec | MOJITO | TALE | Ours |
|---------|--------|--------|----------|-------|----------|----------|-----------|---------|--------|--------|------|------|
| **Ta-Feng** | HR@5 | 0.0886 | 0.0988 | 0.0873 | 0.0101 | 0.0900 | 0.0831 | 0.1357 | 0.1289 | 0.0722 | 0.1364† | **0.1470*** |
| | HR@10 | 0.1260 | 0.1309 | 0.1182 | 0.0190 | 0.1318 | 0.1048 | 0.1776† | 0.1689 | 0.0957 | 0.1628 | **0.1885*** |
| | HR@20 | 0.1688 | 0.1678 | 0.1542 | 0.0379 | 0.1782 | 0.1349 | 0.2249† | 0.2126 | 0.1246 | 0.1893 | **0.2331*** |
| | NDCG@5 | 0.0542 | 0.0702 | 0.0624 | 0.0062 | 0.0569 | 0.0639 | 0.0938 | 0.0939 | 0.0569 | 0.1048† | **0.1084*** |
| | NDCG@10 | 0.0663 | 0.0806 | 0.0725 | 0.0090 | 0.0704 | 0.0709 | 0.1074 | 0.1068 | 0.0641 | 0.1133† | **0.1216*** |
| | NDCG@20 | 0.0771 | 0.0899 | 0.0815 | 0.0137 | 0.0822 | 0.0785 | 0.1193 | 0.1178 | 0.0717 | 0.1201† | **0.1330*** |
| **DHRD** | HR@5 | 0.2525 | 0.2153 | 0.2446 | 0.1254 | 0.2574 | 0.2954 | 0.3444† | 0.3401 | 0.2469 | 0.3075 | **0.3616*** |
| | HR@10 | 0.3584 | 0.3072 | 0.3119 | 0.1684 | 0.3654 | 0.4326 | 0.4800† | 0.4759 | 0.3533 | 0.4143 | **0.4906*** |
| | HR@20 | 0.4754 | 0.4085 | 0.3816 | 0.2286 | 0.4805 | 0.5775 | **0.6131** | 0.6069 | 0.4650 | 0.5101 | 0.6116† |
| | NDCG@5 | 0.1685 | 0.1464 | 0.1752 | 0.0905 | 0.1721 | 0.1957 | 0.2257† | 0.2240 | 0.1686 | 0.2144 | **0.2460*** |
| | NDCG@10 | 0.2026 | 0.1760 | 0.1970 | 0.1043 | 0.2070 | 0.2399 | 0.2695† | 0.2680 | 0.2029 | 0.2489 | **0.2877*** |
| | NDCG@20 | 0.2322 | 0.2017 | 0.2146 | 0.1195 | 0.2361 | 0.2767 | 0.3033† | 0.3012 | 0.2311 | 0.2732 | **0.3184*** |
| **MegaMarket** | HR@5 | 0.0954 | 0.0679 | 0.1017 | 0.0238 | 0.0712 | 0.0959 | 0.1463† | 0.1353 | 0.0464 | 0.1129 | **0.1611*** |
| | HR@10 | 0.1269 | 0.0957 | 0.1265 | 0.0396 | 0.1030 | 0.1303 | 0.1889† | 0.1719 | 0.0666 | 0.1443 | **0.2044*** |
| | HR@20 | 0.1668 | 0.1299 | 0.1591 | 0.0708 | 0.1394 | 0.1731 | 0.2339† | 0.2134 | 0.0918 | 0.1711 | **0.2451*** |
| | NDCG@5 | 0.0681 | 0.0473 | 0.0782 | 0.0156 | 0.0499 | 0.0639 | 0.1032† | 0.0984 | 0.0292 | 0.0843 | **0.1170*** |
| | NDCG@10 | 0.0784 | 0.0562 | 0.0862 | 0.0207 | 0.0602 | 0.0751 | 0.1169† | 0.1102 | 0.0358 | 0.0945 | **0.1310*** |
| | NDCG@20 | 0.0884 | 0.0648 | 0.0944 | 0.0285 | 0.0694 | 0.0859 | 0.1283† | 0.1207 | 0.0421 | 0.1013 | **0.1413*** |


To obtain the test results for these three datasets, run the following scripts:

- `test_eval_dhrd.sh`
- `test_eval_megamarket.sh`
- `test_eval_tafeng.sh`
