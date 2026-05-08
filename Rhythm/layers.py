from __future__ import annotations

import numpy as np
import torch
from torch import nn


def positional_encoding(dim: int, sentence_length: int, dtype=torch.float32) -> torch.Tensor:
    encoded_vec = np.array(
        [pos / np.power(10000, 2 * i / dim) for pos in range(sentence_length) for i in range(dim)]
    )
    encoded_vec[::2] = np.sin(encoded_vec[::2])
    encoded_vec[1::2] = np.cos(encoded_vec[1::2])
    return torch.tensor(encoded_vec.reshape([sentence_length, dim]), dtype=dtype)


class LayerNormalization(nn.Module):
    """Legacy LayerNorm with `gamma`/`beta` parameter names."""

    def __init__(self, hidden_size: int, epsilon: float = 1e-8):
        super().__init__()
        self.hidden_size = hidden_size
        self.epsilon = epsilon
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.gamma = nn.Parameter(torch.ones(hidden_size))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        mean = torch.mean(inputs, dim=-1, keepdim=True)
        variance = torch.var(inputs, dim=-1, keepdim=True)
        normalized = (inputs - mean) / ((variance + self.epsilon) ** 0.5)
        return self.gamma * normalized + self.beta


class Embedding(nn.Module):
    """Traditional item embedding only. LLM embedding branches are intentionally removed."""

    def __init__(self, vocab_size: int, hidden_size: int, zero_pad: bool = True, scale: bool = True):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.zero_pad = zero_pad
        self.scale = scale
        self.embedding_table = nn.Embedding(vocab_size, hidden_size, padding_idx=0 if zero_pad else None)
        nn.init.xavier_normal_(self.embedding_table.weight)
        if self.zero_pad:
            self.embedding_table.weight.data[0] = 0

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings = self.embedding_table(inputs.long())
        if self.scale:
            embeddings = embeddings * (self.hidden_size**0.5)
        return embeddings, self.embedding_table.weight


class FeedForwardNetwork(nn.Module):
    def __init__(self, input_size: int, output_size: int, dropout_rate: float = 0.0):
        super().__init__()
        ff_size = input_size * 4
        self.layer1 = nn.Linear(input_size, ff_size)
        self.layer2 = nn.Linear(ff_size, output_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        self.ln = LayerNormalization(output_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.layer1(inputs)
        outputs = self.relu(outputs)
        outputs = self.dropout(outputs)
        outputs = self.layer2(outputs)
        outputs = self.dropout(outputs)
        return self.ln(outputs + inputs)


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int = 8, dropout_rate: float = 0.0):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        self.W_q = nn.Linear(hidden_size, hidden_size)
        self.W_k = nn.Linear(hidden_size, hidden_size)
        self.W_v = nn.Linear(hidden_size, hidden_size)
        self.W_o = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.ln = LayerNormalization(hidden_size)

    def forward(
        self,
        queries: torch.Tensor,
        causality: bool = False,
        mask: torch.Tensor | None = None,
        time_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        residual = queries
        q = self.W_q(queries)
        k = self.W_k(queries)
        v = self.W_v(queries)

        q = q.view(q.size(0), q.size(1), self.num_heads, self.head_size).transpose(1, 2)
        k = k.view(k.size(0), k.size(1), self.num_heads, self.head_size).transpose(1, 2)
        v = v.view(v.size(0), v.size(1), self.num_heads, self.head_size).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-1, -2)) / (self.head_size**0.5)
        batch_size, _, seq_len, _ = scores.size()

        if time_mask is not None:
            time_mask = time_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            scores = scores.masked_fill(time_mask == 0, -1e30)
        elif mask is not None:
            attn_mask = mask.squeeze(-1).unsqueeze(1).unsqueeze(-1)
            attn_mask = attn_mask.expand(-1, self.num_heads, -1, seq_len)
            scores = scores.masked_fill(attn_mask == 0, -1e30)

        if causality:
            causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=scores.device)).view(1, 1, seq_len, seq_len)
            scores = scores.masked_fill(causal_mask == 0, -1e30)

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(context.size(0), context.size(2), self.hidden_size)
        output = self.W_o(context)
        output = self.dropout(output)
        return self.ln(output + residual), attn_weights


class CrossAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int = 8, dropout_rate: float = 0.0):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        self.W_q = nn.Linear(hidden_size, hidden_size)
        self.W_k = nn.Linear(hidden_size, hidden_size)
        self.W_v = nn.Linear(hidden_size, hidden_size)
        self.W_o = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.ln = LayerNormalization(hidden_size)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        mask: torch.Tensor | None = None,
        causality: bool = False,
        time_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        residual = queries
        q = self.W_q(queries)
        k = self.W_k(keys)
        v = self.W_v(values)

        q = q.view(q.size(0), q.size(1), self.num_heads, self.head_size).transpose(1, 2)
        k = k.view(k.size(0), k.size(1), self.num_heads, self.head_size).transpose(1, 2)
        v = v.view(v.size(0), v.size(1), self.num_heads, self.head_size).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-1, -2)) / (self.head_size**0.5)

        if time_mask is not None:
            time_mask = time_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            scores = scores.masked_fill(time_mask == 0, -1e30)

        if causality:
            batch_size, _, seq_len_q, seq_len_kv = scores.size()
            causal_mask = torch.ones((seq_len_q, seq_len_kv), dtype=torch.bool, device=scores.device)
            causal_mask = torch.tril(causal_mask, diagonal=0)
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, self.num_heads, -1, -1)
            scores = scores.masked_fill(~causal_mask, -1e30)

        if mask is not None:
            batch_size, seq_len, _ = mask.size()
            attn_mask = mask.squeeze(-1).unsqueeze(1).unsqueeze(-1)
            attn_mask = attn_mask.expand(-1, self.num_heads, -1, scores.size(3))
            scores = scores.masked_fill(attn_mask == 0, -1e30)

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(context.size(0), context.size(2), self.hidden_size)
        output = self.W_o(context)
        output = self.dropout(output)
        return self.ln(output + residual), attn_weights
