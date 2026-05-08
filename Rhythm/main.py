from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "Rhythm"

from .checkpoint import load_legacy_checkpoint
from .config import RhythmConfig, str2bool
from .data import load_dataset
from .evaluate import evaluate, evaluate_valid
from .model import SASRec
from .sampler import SequenceBatchSampler
from .segmenter import IntentAwareSegmenter


def choose_device(device: str) -> str:
    if device != "auto":
        return device
    if not torch.cuda.is_available():
        return "cpu"
    free_memory = []
    for gpu_id in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(gpu_id)
        free_memory.append(props.total_memory - torch.cuda.memory_allocated(gpu_id))
    return f"cuda:{int(np.argmax(free_memory))}"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def is_best_by_ndcg10(current_ndcg10: float, best_ndcg10: float) -> bool:
    return current_ndcg10 > best_ndcg10


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train/evaluate Rhythm core model.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--train_dir", required=True)
    parser.add_argument("--batch_size", default=512, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--maxlen", default=50, type=int)
    parser.add_argument("--hidden_units", default=128, type=int)
    parser.add_argument("--num_blocks", default=1, type=int)
    parser.add_argument("--num_epochs", default=300, type=int)
    parser.add_argument("--stop", default=10, type=int)
    parser.add_argument("--num_heads", default=4, type=int)
    parser.add_argument("--attn_dropout_rate", default=0.5, type=float)
    parser.add_argument("--ff_dropout_rate", default=0.5, type=float)
    parser.add_argument("--l2_emb", default=0.0, type=float)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--eval_batch_size", default=256, type=int)
    parser.add_argument("--num_workers", default=1, type=int)
    parser.add_argument("--use_cross", type=str2bool, default=True)
    parser.add_argument("--use_cross_withfnn", type=str2bool, default=False)
    parser.add_argument("--learnable_intent", type=str2bool, default=False)
    parser.add_argument("--loss_type", default="full_ce", choices=["bce", "sampled_softmax", "ce", "full_ce"])
    parser.add_argument("--n_negatives", default=256, type=int)
    parser.add_argument("--time_type", default="hour", choices=["hour", "month", "day_of_week"])
    parser.add_argument("--use_fixed", type=str2bool, default=True)
    parser.add_argument("--kmax", default=4, type=int)
    parser.add_argument("--min_delta", default=0.1, type=float)
    parser.add_argument("--seed", default=2026, type=int)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint to load.")
    parser.add_argument("--eval_only", type=str2bool, default=False)
    return parser


def config_from_cli() -> RhythmConfig:
    parser = build_parser()
    namespace, unknown = parser.parse_known_args()
    if unknown:
        print("Ignoring deprecated/unknown arguments:", " ".join(unknown))
    return RhythmConfig(**vars(namespace))


def tensor_batch(batch, device: str):
    seq = torch.tensor(batch.seq, dtype=torch.int32, device=device)
    pos = torch.tensor(batch.pos, dtype=torch.int32, device=device)
    times = torch.tensor(batch.times, dtype=torch.int32, device=device)
    neg = None if batch.neg is None else torch.tensor(batch.neg, dtype=torch.int32, device=device)
    intent = torch.tensor(batch.intent_id_sequence, dtype=torch.int32, device=device)
    return seq, pos, neg, times, intent


def log_metrics(prefix: str, metrics: tuple[float, float, float, float, float, float, float]) -> str:
    ndcg5, ndcg10, ndcg20, hr5, hr10, hr20, mrr = metrics
    return (
        f"{prefix}: NDCG@5={ndcg5:.4f} NDCG@10={ndcg10:.4f} NDCG@20={ndcg20:.4f} "
        f"HR@5={hr5:.4f} HR@10={hr10:.4f} HR@20={hr20:.4f} MRR={mrr:.4f}"
    )


def main() -> None:
    args = config_from_cli()
    args.device = choose_device(args.device)
    seed_everything(args.seed)

    os.makedirs(args.resolved_output_dir, exist_ok=True)
    with open(os.path.join(args.resolved_output_dir, "args.txt"), "w", encoding="utf-8") as fp:
        fp.write("\n".join(f"{key},{value}" for key, value in args.as_args_dict().items()))

    dataset = load_dataset(args.dataset, time_type=args.time_type)
    print(f"Loaded dataset: users={dataset.usernum}, items={dataset.itemnum}")
    print(f"average sequence length: {dataset.average_train_length:.2f}")
    print("Fixed branches: time_mask_train/eval=True, causal_train/eval=True, use_llm_embedding=False")

    sampler = SequenceBatchSampler(
        dataset.user_train,
        dataset.user_train_times,
        dataset.usernum,
        dataset.itemnum,
        batch_size=args.batch_size,
        maxlen=args.maxlen,
        n_negatives=args.n_negatives,
        time_type=args.time_type,
        use_full_ce=args.use_full_ce,
        seed=args.seed,
    )

    model = SASRec(dataset.usernum, dataset.itemnum, args).to(args.device)
    intent_adapter = IntentAwareSegmenter(
        embedding_dim=args.hidden_units,
        embedding_table=model.embedding.embedding_table,
    ).to(args.device)

    if args.checkpoint:
        report = load_legacy_checkpoint(model, args.checkpoint, map_location=args.device)
        print(f"Loaded checkpoint report: {report.summary()}")

    if args.eval_only:
        print(log_metrics("Valid", evaluate_valid(model, dataset, args, intent_adapter)))
        print(log_metrics("Test", evaluate(model, dataset, args, intent_adapter)))
        sampler.close()
        return

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_ndcg10 = -float("inf")
    best_epoch = 0
    no_improvement_count = 0
    best_model_path = os.path.join(args.resolved_output_dir, "best_model.pt")
    log_path = os.path.join(args.resolved_output_dir, "log.txt")

    num_batch = max(1, len(dataset.user_train) // args.batch_size)
    total_eval_time = 0.0
    t0 = time.time()

    try:
        with open(log_path, "w", encoding="utf-8") as log_file:
            for epoch in range(1, args.num_epochs + 1):
                model.train()
                total_loss = 0.0
                total_auc = 0.0
                progress = tqdm(range(num_batch), total=num_batch, desc=f"Epoch {epoch}", unit="batch", ncols=100)

                for step in progress:
                    batch = sampler.next_batch()
                    seq, pos, neg, times, intent = tensor_batch(batch, args.device)
                    segments = intent_adapter.process_intent_id_sequence(
                        intent,
                        use_fixed=args.use_fixed,
                        K_max=args.kmax,
                        min_delta=args.min_delta,
                    )

                    optimizer.zero_grad()
                    loss, auc = model(seq, pos, neg, times, args.use_time_mask_train, segments)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                    total_loss += loss.item()
                    total_auc += auc.item()
                    avg_loss = total_loss / (step + 1)
                    avg_auc = total_auc / (step + 1)
                    progress.set_postfix(loss=f"{loss.item():.4f}", avg_loss=f"{avg_loss:.4f}", avg_auc=f"{avg_auc:.4f}")

                total_eval_time += time.time() - t0
                print(f"Epoch {epoch} validation:")
                valid_metrics = evaluate_valid(model, dataset, args, intent_adapter)
                print(f"epoch:{epoch}, time:{total_eval_time:.2f}(s)")
                print(log_metrics("Valid", valid_metrics))
                log_file.write(log_metrics("Valid", valid_metrics) + "\n")
                log_file.flush()

                _, valid_ndcg10, _, _, _, _, _ = valid_metrics
                is_better = is_best_by_ndcg10(valid_ndcg10, best_ndcg10)

                if is_better:
                    best_ndcg10 = valid_ndcg10
                    best_epoch = epoch
                    no_improvement_count = 0
                    torch.save(model.state_dict(), best_model_path)
                    print(f"New best model (NDCG@10 improved) saved to {best_model_path}")
                else:
                    no_improvement_count += 1
                    print(f"No improvement: {no_improvement_count}/{args.stop}")

                if no_improvement_count >= args.stop:
                    print(f"Earlystop triggered after {args.stop} epochs without improvement.")
                    break
                t0 = time.time()
    finally:
        sampler.close()

    print("\n=== Final Evaluation with Best Model ===")
    print(f"Best epoch={best_epoch}, NDCG@10={best_ndcg10:.4f}")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=args.device))
    test_metrics = evaluate(model, dataset, args, intent_adapter)
    print(log_metrics("Test", test_metrics))
    with open(log_path, "a", encoding="utf-8") as log_file:
        log_file.write("\n=== Final Test Results ===\n")
        log_file.write(
            f"BestEpoch: {best_epoch} BestNDCG@10: {best_ndcg10:.4f} "
            + log_metrics("Test", test_metrics)
            + "\n"
        )


if __name__ == "__main__":
    main()
