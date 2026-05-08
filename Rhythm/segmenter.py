from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class HardSegmentMaskMechanism(nn.Module):
    """Greedy intent segmentation used by the dynamic time-mask branch."""

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim

    def _segment_interval(self, vectors: torch.Tensor, start: int, end: int, min_delta: float):
        interval_vectors = vectors[start:end]
        interval_length = end - start
        split_positions = torch.arange(1, interval_length, device=vectors.device)
        if len(split_positions) == 0:
            return None, -float("inf")

        cumsum = torch.cumsum(interval_vectors, dim=0)
        left_counts = split_positions.float()
        left_sums = cumsum[split_positions - 1]
        left_means = left_sums / left_counts.unsqueeze(1)

        right_counts = (interval_length - split_positions).float()
        right_sums = cumsum[-1] - left_sums
        right_means = right_sums / right_counts.unsqueeze(1)

        deltas = 1.0 - F.cosine_similarity(left_means, right_means, dim=1)
        max_delta, max_idx = torch.max(deltas, dim=0)
        if max_delta.item() < min_delta:
            return None, -float("inf")
        return start + split_positions[max_idx].item(), max_delta.item()

    def _merge_segments(self, vectors: torch.Tensor, segments: list[tuple[int, int]], k_max: int):
        seq_len = len(vectors)
        while len(segments) > k_max:
            means = []
            for start, end in segments:
                segment_vec = vectors[start:end] if start < end else torch.cat([vectors[start:], vectors[:end]], dim=0)
                means.append(torch.mean(segment_vec, dim=0) if segment_vec.size(0) else torch.zeros_like(vectors[0]))
            means_tensor = torch.stack(means, dim=0)
            sims = torch.cat(
                [
                    F.cosine_similarity(means_tensor[:-1], means_tensor[1:], dim=1),
                    F.cosine_similarity(means_tensor[-1:], means_tensor[:1], dim=1),
                ],
                dim=0,
            )
            min_idx = torch.argmin(1.0 - sims).item()

            if min_idx == len(segments) - 1:
                merged = (segments[-1][0], segments[0][1])
                segments.pop()
                segments.pop(0)
                segments.insert(0, merged)
            else:
                merged = (segments[min_idx][0], segments[min_idx + 1][1])
                segments.pop(min_idx + 1)
                segments.pop(min_idx)
                segments.insert(min_idx, merged)
        return segments

    def _greedy_segment(self, vectors: torch.Tensor, k_max: int, min_delta: float):
        segments = [(0, len(vectors))]
        max_initial_segments = k_max + k_max // 2 + 1

        for _ in range(max_initial_segments - 1):
            best_split = None
            best_delta = -float("inf")
            best_idx = -1
            for idx, (start, end) in enumerate(segments):
                split, delta = self._segment_interval(vectors, start, end, min_delta)
                if delta > best_delta:
                    best_split = split
                    best_delta = delta
                    best_idx = idx
            if best_split is None:
                break

            start, end = segments[best_idx]
            segments[best_idx] = (start, best_split)
            segments.insert(best_idx + 1, (best_split, end))

        return self._merge_segments(vectors, segments, k_max)

    def forward(self, sequence_embeddings: torch.Tensor, k_max: int = 6, min_delta: float = 0.1) -> torch.Tensor:
        batch_size, seq_len = sequence_embeddings.shape[:2]
        segment_ids = torch.zeros((batch_size, seq_len), dtype=torch.long, device=sequence_embeddings.device)

        for batch_idx in range(batch_size):
            raw_segments = self._greedy_segment(sequence_embeddings[batch_idx], k_max, min_delta)
            for seg_id, (start, end) in enumerate(raw_segments):
                if start < end:
                    positions = list(range(start, end))
                else:
                    positions = list(range(start, seq_len)) + list(range(0, end))
                if positions:
                    segment_ids[batch_idx, torch.tensor(positions, dtype=torch.long, device=segment_ids.device)] = seg_id
        return segment_ids


class IntentAwareSegmenter(nn.Module):
    """Convert item IDs grouped by time unit into segment IDs for time masking."""

    def __init__(self, embedding_dim: int, embedding_table: nn.Module | None = None):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.embedding_table = embedding_table
        self.segment_mechanism = HardSegmentMaskMechanism(embedding_dim)

    @staticmethod
    def fixed_segments(batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
        if seq_len == 24:
            hours = torch.arange(24, device=device).repeat(batch_size, 1)
            segments = torch.zeros_like(hours)
            segments[(hours >= 5) & (hours < 11)] = 1
            segments[(hours >= 11) & (hours < 17)] = 2
            segments[(hours >= 17) & (hours < 23)] = 3
            segments[(hours < 5) | (hours >= 23)] = 4
            return segments
        if seq_len == 12:
            months = torch.arange(12, device=device).repeat(batch_size, 1)
            segments = torch.zeros_like(months)
            segments[months < 3] = 1
            segments[(months >= 3) & (months < 6)] = 2
            segments[(months >= 6) & (months < 9)] = 3
            segments[months >= 9] = 4
            return segments
        if seq_len == 7:
            segments = torch.zeros((batch_size, 7), dtype=torch.long, device=device)
            segments[:, 0] = 1
            segments[:, 6] = 1
            return segments
        return torch.zeros((batch_size, seq_len), dtype=torch.long, device=device)

    def process_intent_id_sequence(
        self,
        intent_id_sequence: np.ndarray | torch.Tensor,
        use_fixed: bool = False,
        K_max: int = 6,
        min_delta: float = 0.1,
    ) -> torch.Tensor:
        if isinstance(intent_id_sequence, np.ndarray):
            intent_id_sequence = torch.from_numpy(intent_id_sequence)
        if intent_id_sequence.dim() == 2:
            intent_id_sequence = intent_id_sequence.unsqueeze(0)

        batch_size, seq_len, n_items = intent_id_sequence.shape
        device = intent_id_sequence.device

        if use_fixed:
            return self.fixed_segments(batch_size, seq_len, device)
        if self.embedding_table is None:
            raise ValueError("embedding_table is required for dynamic segmentation")

        flattened = intent_id_sequence.long().view(batch_size * seq_len, n_items)
        item_embeddings = self.embedding_table(flattened)
        non_zero_mask = (flattened != 0).unsqueeze(2).float()
        summed = torch.sum(item_embeddings * non_zero_mask, dim=1)
        counts = torch.clamp(torch.sum(non_zero_mask, dim=1), min=1e-8)
        intent_vectors = (summed / counts).view(batch_size, seq_len, self.embedding_dim)
        return self.segment_mechanism(intent_vectors, k_max=K_max, min_delta=min_delta)

    def forward(
        self,
        input_data: torch.Tensor,
        mask_type: str = "segment",
        is_id_sequence: bool = False,
        K_max: int = 6,
        min_delta: float = 0.1,
    ) -> dict[str, Any]:
        if mask_type != "segment":
            batch_size, seq_len = input_data.shape[:2]
            return {
                "segments": [[] for _ in range(batch_size)],
                "segment_ids": torch.zeros((batch_size, seq_len), dtype=torch.long, device=input_data.device),
            }
        if is_id_sequence:
            segment_ids = self.process_intent_id_sequence(input_data, use_fixed=False, K_max=K_max, min_delta=min_delta)
        else:
            segment_ids = self.segment_mechanism(input_data, k_max=K_max, min_delta=min_delta)
        return {"segments": [], "segment_ids": segment_ids}
