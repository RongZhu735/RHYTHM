from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .config import RhythmConfig
from .layers import CrossAttention, Embedding, FeedForwardNetwork, LayerNormalization, MultiHeadAttention, positional_encoding


class SASRec(nn.Module):
    """Core Rhythm model.

    The implemented path matches the open-source target mode:
    time mask on, causal mask on, no LLM embeddings, no attention weight dumping,
    no warmup checkpoint branch, and no attention-parameter cloning branch.
    """

    def __init__(self, usernum: int, itemnum: int, args: RhythmConfig):
        super().__init__()
        self.usernum = usernum
        self.itemnum = itemnum
        self.args = args
        self.hidden_units = args.hidden_units
        self.maxlen = args.maxlen
        self.learnable_intent = args.learnable_intent

        if self.learnable_intent:
            self.time_mask_weights = nn.Parameter(torch.randn(1, args.maxlen, args.maxlen) * 0.01)

        self.embedding = Embedding(itemnum + 1, args.hidden_units, zero_pad=True, scale=True)
        self.register_buffer("positional_encoding", positional_encoding(args.hidden_units, args.maxlen), persistent=False)

        self.attention_layers_time = nn.ModuleList(
            [MultiHeadAttention(args.hidden_units, args.num_heads, args.attn_dropout_rate) for _ in range(args.num_blocks)]
        )
        self.ffn_layers_time = nn.ModuleList(
            [FeedForwardNetwork(args.hidden_units, args.hidden_units, args.ff_dropout_rate) for _ in range(args.num_blocks)]
        )
        self.cross_attention = CrossAttention(args.hidden_units, args.num_heads, args.attn_dropout_rate)
        self.cross_ffn = FeedForwardNetwork(args.hidden_units, args.hidden_units, args.ff_dropout_rate)
        self.last_layernorm = LayerNormalization(args.hidden_units)
        self.dropout = nn.Dropout(args.attn_dropout_rate)

        self.use_cross = args.use_cross
        self.use_cross_withfnn = args.use_cross_withfnn
        if self.use_cross and self.use_cross_withfnn:
            raise ValueError("Choose only one of use_cross or use_cross_withfnn.")

    def _add_positional_encoding(self, seq_emb: torch.Tensor) -> torch.Tensor:
        seq_len = seq_emb.size(1)
        return seq_emb + self.positional_encoding[:seq_len, :].unsqueeze(0).to(seq_emb.device)

    def _fallback_segments(self, seq_times: torch.Tensor) -> torch.Tensor:
        if self.args.time_type == "hour":
            segments = torch.zeros_like(seq_times, dtype=torch.long)
            segments[(seq_times >= 5) & (seq_times < 11)] = 1
            segments[(seq_times >= 11) & (seq_times < 17)] = 2
            segments[(seq_times >= 17) & (seq_times < 23)] = 3
            segments[(seq_times < 5) | (seq_times >= 23)] = 4
            return segments
        if self.args.time_type == "month":
            segments = torch.zeros_like(seq_times, dtype=torch.long)
            segments[seq_times < 3] = 1
            segments[(seq_times >= 3) & (seq_times < 6)] = 2
            segments[(seq_times >= 6) & (seq_times < 9)] = 3
            segments[seq_times >= 9] = 4
            return segments
        if self.args.time_type == "day_of_week":
            segments = torch.zeros_like(seq_times, dtype=torch.long)
            segments[(seq_times == 0) | (seq_times == 6)] = 1
            return segments
        raise ValueError(f"Unsupported time_type: {self.args.time_type}")

    def generate_time_mask(
        self,
        seq_times: torch.Tensor,
        use_time_mask: bool = True,
        segments: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if not use_time_mask:
            return None, None

        batch_size, seq_len = seq_times.shape
        if segments is not None:
            segments = segments.long()
            if segments.dim() != 2:
                raise ValueError(f"segments must be 2-D, got shape={tuple(segments.shape)}")
            if segments.size(1) == seq_len:
                seq_segments = segments
            else:
                time_dim = segments.size(1)
                time_indices = torch.clamp(seq_times.long(), 0, time_dim - 1)
                batch_indices = torch.arange(batch_size, device=seq_times.device).unsqueeze(1).expand(-1, seq_len)
                seq_segments = segments[batch_indices, time_indices]
        else:
            seq_segments = self._fallback_segments(seq_times.long())

        segment_i = seq_segments.unsqueeze(2)
        segment_j = seq_segments.unsqueeze(1)
        time_mask = (segment_i == segment_j).to(torch.int32)
        diagonal = torch.eye(seq_len, device=seq_times.device, dtype=torch.bool).unsqueeze(0).expand(batch_size, -1, -1)
        time_mask = time_mask | diagonal.to(torch.int32)
        return time_mask, seq_segments

    def apply_time_mask_to_seq_emb(self, seq_emb: torch.Tensor, segments: torch.Tensor | None) -> torch.Tensor:
        batch_size, seq_len, _ = seq_emb.size()
        causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=seq_emb.device, dtype=torch.float))
        causal_mask = causal_mask.unsqueeze(0).expand(batch_size, -1, -1)

        if segments is not None:
            segment_i = segments.unsqueeze(2)
            segment_j = segments.unsqueeze(1)
            structure_mask = (segment_i == segment_j).float() * causal_mask
        else:
            structure_mask = causal_mask

        if self.learnable_intent:
            weights = self.time_mask_weights[:, :seq_len, :seq_len].to(seq_emb.device)
            # Keep the legacy behavior for old checkpoints: use raw learnable
            # weights under the periodic-causal structure mask.
            weighted_mask = weights * structure_mask
            return torch.matmul(weighted_mask, seq_emb)

        masked_seq_emb = torch.matmul(structure_mask, seq_emb)
        segment_counts = torch.clamp(structure_mask.sum(dim=2, keepdim=True), min=1)
        return masked_seq_emb / segment_counts

    def encode(
        self,
        seq: torch.Tensor,
        times: torch.Tensor,
        use_time_mask: bool,
        segments: torch.Tensor | None = None,
        training: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        padding_mask = (seq != 0).unsqueeze(-1).float()
        time_mask, seq_segments = self.generate_time_mask(times, use_time_mask=use_time_mask, segments=segments)
        seq_emb, item_emb_table = self.embedding(seq)
        seq_emb = self._add_positional_encoding(seq_emb)
        seq_emb = self.dropout(seq_emb)
        seq_emb *= padding_mask

        causality = self.args.use_causal_mask_train if training else self.args.use_causal_mask_eval

        if self.use_cross or self.use_cross_withfnn:
            intent_vector = self.apply_time_mask_to_seq_emb(seq_emb, seq_segments)
            intent_vector *= padding_mask

            for layer_idx, attention_layer in enumerate(self.attention_layers_time):
                seq_emb, _ = attention_layer(seq_emb, causality=causality, mask=padding_mask)
                if self.use_cross_withfnn:
                    seq_emb = self.ffn_layers_time[layer_idx](seq_emb)
                    seq_emb *= padding_mask

            seq_emb, _ = self.cross_attention(seq_emb, intent_vector, intent_vector, mask=padding_mask, causality=True)
            if self.use_cross_withfnn:
                seq_emb = self.cross_ffn(seq_emb)
            final_emb = seq_emb * padding_mask
        else:
            final_emb = seq_emb
            for layer_idx, attention_layer in enumerate(self.attention_layers_time):
                final_emb, _ = attention_layer(final_emb, causality=causality, mask=padding_mask, time_mask=time_mask)
                final_emb = self.ffn_layers_time[layer_idx](final_emb)
                final_emb *= padding_mask

        return self.last_layernorm(final_emb), item_emb_table

    def forward(
        self,
        seq: torch.Tensor,
        pos: torch.Tensor,
        neg: torch.Tensor | None,
        times: torch.Tensor,
        use_time_mask: bool = True,
        segments: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        final_emb, item_emb_table = self.encode(seq, times, use_time_mask, segments, training=True)
        istarget = (pos != 0).float()
        loss_type = self.args.loss_type

        if loss_type in {"ce", "full_ce"}:
            logits = torch.matmul(final_emb, item_emb_table.transpose(0, 1))
            logits = logits - logits.max(dim=-1, keepdim=True)[0]
            labels = pos.long()
            flat_logits = logits.view(-1, logits.size(-1))
            flat_labels = labels.view(-1)
            valid_indices = istarget.view(-1).nonzero().squeeze()
            if valid_indices.numel() > 0:
                loss = F.cross_entropy(flat_logits[valid_indices], flat_labels[valid_indices])
            else:
                loss = torch.tensor(0.0, device=seq.device, requires_grad=True)
        elif loss_type == "sampled_softmax":
            if neg is None:
                raise ValueError("sampled_softmax requires negative samples")
            pos_emb = item_emb_table[pos.long()]
            pos_logits = (final_emb * pos_emb).sum(dim=-1)
            neg_emb = item_emb_table[neg.long()]
            if neg.dim() == 2:
                neg_logits = (final_emb * neg_emb).sum(dim=-1).unsqueeze(-1)
            else:
                neg_logits = torch.matmul(neg_emb, final_emb.unsqueeze(2).transpose(-2, -1)).squeeze(-1)
            logits = torch.cat([pos_logits.unsqueeze(-1), neg_logits], dim=-1)
            logits = logits - logits.max(dim=-1, keepdim=True)[0]
            labels = torch.zeros((seq.size(0), seq.size(1)), dtype=torch.long, device=seq.device)
            valid_indices = istarget.view(-1).nonzero().squeeze()
            if valid_indices.numel() > 0:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1))[valid_indices], labels.view(-1)[valid_indices])
            else:
                loss = torch.tensor(0.0, device=seq.device, requires_grad=True)
        elif loss_type == "bce":
            if neg is None:
                raise ValueError("bce requires negative samples")
            pos_emb = item_emb_table[pos.long()]
            neg_emb = item_emb_table[neg.long()]
            pos_logits = (final_emb * pos_emb).sum(dim=-1)
            neg_logits = (final_emb * neg_emb).sum(dim=-1)
            loss = -torch.log(torch.sigmoid(pos_logits) + 1e-24) * istarget
            loss = loss - torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * istarget
            loss = loss.sum() / torch.clamp(istarget.sum(), min=1)
        else:
            raise ValueError(f"Unsupported loss_type: {loss_type}")

        auc = self._training_auc(final_emb, item_emb_table, pos, neg, istarget)
        return loss, auc

    def _training_auc(
        self,
        final_emb: torch.Tensor,
        item_emb_table: torch.Tensor,
        pos: torch.Tensor,
        neg: torch.Tensor | None,
        istarget: torch.Tensor,
    ) -> torch.Tensor:
        valid_positions = istarget.bool()
        if valid_positions.sum() == 0:
            return torch.tensor(0.5, device=final_emb.device)

        if self.args.loss_type in {"ce", "full_ce"}:
            logits = torch.matmul(final_emb, item_emb_table.transpose(0, 1))
            labels = pos.long()
            valid_logits = logits[valid_positions]
            valid_labels = labels[valid_positions]
            pos_scores = valid_logits.gather(1, valid_labels.unsqueeze(1)).squeeze(1)
            comparison = (pos_scores.unsqueeze(1) > valid_logits).float()
            pos_mask = torch.zeros_like(comparison)
            pos_mask.scatter_(1, valid_labels.unsqueeze(1), 1.0)
            comparison = comparison * (1 - pos_mask)
            return (comparison.sum(dim=1) / self.itemnum).mean()

        if neg is None:
            return torch.tensor(0.5, device=final_emb.device)
        pos_emb = item_emb_table[pos.long()]
        pos_logits = (final_emb * pos_emb).sum(dim=-1)
        neg_emb = item_emb_table[neg.long()]
        if neg.dim() == 2:
            neg_logits = (final_emb * neg_emb).sum(dim=-1).unsqueeze(-1)
        else:
            neg_logits = torch.matmul(neg_emb, final_emb.unsqueeze(2).transpose(-2, -1)).squeeze(-1)
        return (pos_logits[valid_positions].unsqueeze(-1) > neg_logits[valid_positions]).float().mean()

    def predict(
        self,
        input_seq: torch.Tensor,
        item_idx: torch.Tensor,
        times: torch.Tensor,
        segments: torch.Tensor | None = None,
    ) -> torch.Tensor:
        final_emb, item_emb_table = self.encode(
            input_seq,
            times,
            use_time_mask=self.args.use_time_mask_eval,
            segments=segments,
            training=False,
        )
        last_emb = final_emb[:, -1, :]
        item_emb = item_emb_table[item_idx.long()]
        return torch.matmul(last_emb.unsqueeze(1), item_emb.transpose(1, 2)).squeeze(1)
