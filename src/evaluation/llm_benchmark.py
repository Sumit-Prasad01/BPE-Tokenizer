"""Downstream small language model evaluation benchmarking Loss Per Byte / Bits Per Character."""

import math
import time
from pathlib import Path
from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import LMEvalConfig
from src.tokenizer.tokenizer import Tokenizer
from utils.logger import logger


# ==========================================
# 1. Lightweight Causal Transformer (nanoGPT)
# ==========================================
class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float = 0.1):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head

        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Causal mask
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, n_embd: int, dropout: float = 0.1):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float = 0.1):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        block_size: int = 512,
        n_layer: int = 6,
        n_head: int = 6,
        n_embd: int = 384,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.block_size = block_size
        self.vocab_size = vocab_size

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(vocab_size, n_embd),
                wpe=nn.Embedding(block_size, n_embd),
                drop=nn.Dropout(dropout),
                h=nn.ModuleList([Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)]),
                ln_f=nn.LayerNorm(n_embd),
            )
        )
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        # Weight tying
        self.transformer.wte.weight = self.lm_head.weight

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        device = idx.device
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss


# ==========================================
# 2. LM Benchmark Runner
# ==========================================
class LMBenchmarkRunner:
    """Trains a mini-transformer to evaluate Validation Loss and Loss Per Byte."""

    def __init__(self, tokenizer: Tokenizer, config: LMEvalConfig | None = None):
        self.tokenizer = tokenizer
        self.config = config or LMEvalConfig()

        if self.config.training.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = self.config.training.device

    def train_and_evaluate(
        self,
        text_corpus: str,
        max_steps: int | None = None,
    ) -> dict[str, Any]:
        """Trains for a fixed step budget and computes normalized Loss Per Byte and BPC."""
        steps = max_steps or self.config.training.max_steps
        logger.info(
            f"🧠 Starting Downstream LM benchmark with '{self.tokenizer.name}' "
            f"on {self.device.upper()} for {steps} steps..."
        )

        # 1. Tokenize corpus
        t0 = time.perf_counter()
        token_ids = self.tokenizer.encode(text_corpus)
        raw_bytes = len(text_corpus.encode("utf-8"))
        num_tokens = len(token_ids)

        if num_tokens < self.config.model.block_size * 2:
            raise ValueError(f"Corpus too small for LM evaluation: {num_tokens} tokens")

        bytes_per_token = raw_bytes / num_tokens

        # 2. Split train and validation
        split_idx = int(num_tokens * self.config.dataset.train_split_ratio)
        train_ids = token_ids[:split_idx]
        val_ids = token_ids[split_idx:]

        train_tensor = torch.tensor(train_ids, dtype=torch.long)
        val_tensor = torch.tensor(val_ids, dtype=torch.long)

        # 3. Initialize MiniGPT
        model = MiniGPT(
            vocab_size=self.tokenizer.vocab_size,
            block_size=self.config.model.block_size,
            n_layer=self.config.model.n_layer,
            n_head=self.config.model.n_head,
            n_embd=self.config.model.n_embd,
            dropout=self.config.model.dropout,
        ).to(self.device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.training.learning_rate,
            betas=(self.config.training.beta1, self.config.training.beta2),
            weight_decay=self.config.training.weight_decay,
        )

        block_size = self.config.model.block_size
        batch_size = self.config.training.batch_size

        def get_batch(data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            ix = torch.randint(len(data) - block_size, (batch_size,))
            x = torch.stack([data[i : i + block_size] for i in ix]).to(self.device)
            y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix]).to(self.device)
            return x, y

        @torch.no_grad()
        def estimate_loss(eval_iters: int = 20) -> float:
            model.eval()
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                x, y = get_batch(val_tensor)
                _, loss = model(x, y)
                losses[k] = loss.item()
            model.train()
            return float(losses.mean().item())

        # 4. Training loop
        loss_history: list[dict[str, float]] = []
        for step in range(1, steps + 1):
            xb, yb = get_batch(train_tensor)
            optimizer.zero_grad(set_to_none=True)
            _, loss = model(xb, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if step % self.config.training.eval_interval == 0 or step == steps:
                val_loss = estimate_loss(self.config.training.eval_steps)
                loss_per_byte = val_loss / bytes_per_token
                bpc = loss_per_byte / math.log(2)
                loss_history.append({
                    "step": step,
                    "train_loss": round(float(loss.item()), 4),
                    "val_loss": round(val_loss, 4),
                    "loss_per_byte": round(loss_per_byte, 4),
                    "bpc": round(bpc, 4),
                })
                logger.info(
                    f"Step {step:4d}/{steps:4d} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Loss/Byte: {loss_per_byte:.4f} | "
                    f"BPC: {bpc:.4f}"
                )

        final_val_loss = estimate_loss(eval_iters=40)
        final_ppl = math.exp(final_val_loss)
        final_loss_per_byte = final_val_loss / bytes_per_token
        final_bpc = final_loss_per_byte / math.log(2)

        return {
            "model_name": self.tokenizer.name,
            "vocab_size": self.tokenizer.vocab_size,
            "training_steps": steps,
            "bytes_per_token": round(bytes_per_token, 4),
            "val_loss_per_token": round(final_val_loss, 4),
            "val_perplexity": round(final_ppl, 2),
            "val_loss_per_byte": round(final_loss_per_byte, 4),
            "val_bits_per_character": round(final_bpc, 4),
            "loss_history": loss_history,
        }
