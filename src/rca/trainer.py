"""
RCA Trainer
===========

Training loop with gradient checkpointing, DDP, FSDP, XLA,
torch.compile, and automatic mixed precision.

Author: Rajaaditya.R
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import os
import math
import time

# Optional imports for distributed training
try:
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data.distributed import DistributedSampler
    DDP_AVAILABLE = True
except ImportError:
    DDP_AVAILABLE = False

try:
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        MixedPrecision,
        ShardingStrategy,
    )
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    FSDP_AVAILABLE = True
except ImportError:
    FSDP_AVAILABLE = False

# TPU/XLA support
try:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    XLA_AVAILABLE = True
except ImportError:
    XLA_AVAILABLE = False


@dataclass
class TrainingArguments:
    """Configuration for training."""

    output_dir: str = "./output"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_grad_norm: float = 1.0

    # Logging
    logging_steps: int = 10

    # Saving
    save_steps: int = 500
    save_total_limit: int = 3

    # Evaluation
    eval_steps: int = 500
    eval_strategy: str = "steps"  # "steps" or "epoch"

    # Hardware
    fp16: bool = False
    bf16: bool = False
    gradient_accumulation_steps: int = 1

    # CUDA
    use_cuda_scan: bool = True

    # Distributed
    use_ddp: bool = False
    use_fsdp: bool = False
    use_xla: bool = False
    local_rank: int = -1

    # torch.compile
    use_torch_compile: bool = False
    compile_mode: str = "reduce-overhead"


class RCATrainer:
    """
    Trainer for RCA models.

    Supports:
    - Mixed precision (fp16/bf16)
    - Gradient accumulation
    - Gradient checkpointing (via model config)
    - Cosine LR schedule with warmup
    - DDP multi-GPU training
    - FSDP for large models (5B+)
    - TPU/XLA training
    - torch.compile acceleration
    - Automatic checkpointing
    """

    def __init__(
        self,
        model: nn.Module,
        args: TrainingArguments,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        collate_fn=None,
    ):
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.collate_fn = collate_fn

        # Determine device
        if args.use_xla and XLA_AVAILABLE:
            self.device = xm.xla_device()
        elif args.use_ddp and args.local_rank >= 0:
            self.device = torch.device(f"cuda:{args.local_rank}")
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)

        # Wrap model for distributed training
        if args.use_fsdp and FSDP_AVAILABLE and dist.is_initialized():
            # Import block types for auto-wrapping
            from .modeling.rca_model import MambaMixBlock, GLABlock, ReasoningBlock
            auto_wrap = transformer_auto_wrap_policy(
                transformer_layer_cls={MambaMixBlock, GLABlock, ReasoningBlock}
            )
            mp = None
            if args.bf16:
                mp = MixedPrecision(
                    param_dtype=torch.bfloat16,
                    reduce_dtype=torch.bfloat16,
                    buffer_dtype=torch.bfloat16,
                )
            elif args.fp16:
                mp = MixedPrecision(
                    param_dtype=torch.float16,
                    reduce_dtype=torch.float16,
                    buffer_dtype=torch.float16,
                )
            self.model = FSDP(
                self.model,
                auto_wrap_policy=auto_wrap,
                mixed_precision=mp,
                sharding_strategy=ShardingStrategy.FULL_SHARD,
            )
        elif args.use_ddp and DDP_AVAILABLE and dist.is_initialized():
            self.model = DDP(self.model, device_ids=[args.local_rank])

        # torch.compile
        if args.use_torch_compile and hasattr(torch, "compile"):
            self.model = torch.compile(self.model, mode=args.compile_mode)

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )

        self.global_step = 0
        self.epoch = 0
        self.best_eval_loss = float("inf")

    def _get_lr(self, step: int, total_steps: int) -> float:
        """Cosine schedule with warmup."""
        if step < self.args.warmup_steps:
            return self.args.learning_rate * step / max(1, self.args.warmup_steps)
        progress = (step - self.args.warmup_steps) / max(
            1, total_steps - self.args.warmup_steps
        )
        return self.args.learning_rate * 0.5 * (1 + math.cos(math.pi * progress))

    def _is_main_process(self) -> bool:
        """Returns True if this is the main process (for logging/saving)."""
        if self.args.use_xla and XLA_AVAILABLE:
            return xm.is_master_ordinal()
        if dist.is_initialized():
            return dist.get_rank() == 0
        return True

    def train(self) -> Dict[str, float]:
        """Run training loop."""
        assert self.train_dataset is not None, "train_dataset required"

        # Build data loader (with distributed sampler if needed)
        sampler = None
        shuffle = True
        if self.args.use_ddp and dist.is_initialized():
            sampler = DistributedSampler(self.train_dataset)
            shuffle = False

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=shuffle,
            sampler=sampler,
            collate_fn=self.collate_fn,
        )

        # XLA parallel loader
        if self.args.use_xla and XLA_AVAILABLE:
            train_loader = pl.MpDeviceLoader(train_loader, self.device)

        total_steps = (
            len(train_loader)
            * self.args.num_train_epochs
            // self.args.gradient_accumulation_steps
        )

        os.makedirs(self.args.output_dir, exist_ok=True)

        # Mixed precision scaler (GPU only, not for bf16 or XLA)
        scaler = None
        if self.args.fp16 and self.device.type == "cuda" and not self.args.use_fsdp:
            scaler = torch.amp.GradScaler("cuda")

        self.model.train()
        total_loss = 0.0
        log_loss = 0.0
        t0 = time.time()

        for epoch in range(self.args.num_train_epochs):
            self.epoch = epoch
            if sampler is not None:
                sampler.set_epoch(epoch)

            for step, batch in enumerate(train_loader):
                # Move to device
                if isinstance(batch, dict):
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    input_ids = batch.get("input_ids")
                    labels = batch.get("labels", input_ids)
                elif isinstance(batch, (list, tuple)):
                    input_ids = batch[0].to(self.device)
                    labels = batch[1].to(self.device) if len(batch) > 1 else input_ids
                else:
                    input_ids = batch.to(self.device)
                    labels = input_ids

                # Update LR
                lr = self._get_lr(self.global_step, total_steps)
                for pg in self.optimizer.param_groups:
                    pg["lr"] = lr

                # Forward
                use_amp = self.args.fp16 or self.args.bf16
                dtype = torch.float16 if self.args.fp16 else torch.bfloat16
                if use_amp and self.device.type == "cuda":
                    with torch.amp.autocast("cuda", dtype=dtype):
                        out = self.model(
                            input_ids,
                            labels=labels,
                            use_cuda=self.args.use_cuda_scan,
                        )
                        loss = out.loss / self.args.gradient_accumulation_steps
                else:
                    out = self.model(
                        input_ids,
                        labels=labels,
                        use_cuda=self.args.use_cuda_scan,
                    )
                    loss = out.loss / self.args.gradient_accumulation_steps

                # Backward
                if scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                total_loss += loss.item()
                log_loss += loss.item()

                # Step
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    if scaler:
                        scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.args.max_grad_norm
                        )
                        scaler.step(self.optimizer)
                        scaler.update()
                    elif self.args.use_xla and XLA_AVAILABLE:
                        # XLA optimizer step
                        nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.args.max_grad_norm
                        )
                        xm.optimizer_step(self.optimizer)
                    else:
                        nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.args.max_grad_norm
                        )
                        self.optimizer.step()

                    # set_to_none=True is faster than zero_grad()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.global_step += 1

                    # Logging (main process only)
                    if (
                        self.global_step % self.args.logging_steps == 0
                        and self._is_main_process()
                    ):
                        avg = log_loss / self.args.logging_steps
                        elapsed = time.time() - t0
                        tokens_per_sec = (
                            self.args.logging_steps
                            * self.args.per_device_train_batch_size
                            * self.args.gradient_accumulation_steps
                            * input_ids.shape[1]
                            / elapsed
                        )
                        print(
                            f"Step {self.global_step} | "
                            f"Loss {avg:.4f} | "
                            f"LR {lr:.2e} | "
                            f"Tok/s {tokens_per_sec:.0f} | "
                            f"Time {elapsed:.1f}s"
                        )
                        log_loss = 0.0
                        t0 = time.time()

                    # Save
                    if (
                        self.args.save_steps > 0
                        and self.global_step % self.args.save_steps == 0
                        and self._is_main_process()
                    ):
                        self._save_checkpoint()

                    # Eval
                    if (
                        self.args.eval_strategy == "steps"
                        and self.args.eval_steps > 0
                        and self.global_step % self.args.eval_steps == 0
                        and self.eval_dataset is not None
                    ):
                        self.evaluate()
                        self.model.train()

            # End-of-epoch eval
            if self.args.eval_strategy == "epoch" and self.eval_dataset is not None:
                self.evaluate()
                self.model.train()

        # Final save
        if self._is_main_process():
            self._save_checkpoint(final=True)

        return {"loss": total_loss / max(1, self.global_step)}

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Run evaluation."""
        assert self.eval_dataset is not None

        eval_loader = DataLoader(
            self.eval_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            collate_fn=self.collate_fn,
        )

        if self.args.use_xla and XLA_AVAILABLE:
            eval_loader = pl.MpDeviceLoader(eval_loader, self.device)

        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for batch in eval_loader:
            if isinstance(batch, dict):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                input_ids = batch.get("input_ids")
                labels = batch.get("labels", input_ids)
            elif isinstance(batch, (list, tuple)):
                input_ids = batch[0].to(self.device)
                labels = batch[1].to(self.device) if len(batch) > 1 else input_ids
            else:
                input_ids = batch.to(self.device)
                labels = input_ids

            out = self.model(input_ids, labels=labels)
            total_loss += out.loss.item()
            n_batches += 1

        avg_loss = total_loss / max(1, n_batches)
        perplexity = math.exp(min(avg_loss, 100))

        if self._is_main_process():
            print(f"Eval | Loss {avg_loss:.4f} | Perplexity {perplexity:.2f}")

        if avg_loss < self.best_eval_loss:
            self.best_eval_loss = avg_loss
            if self._is_main_process():
                self._save_checkpoint(best=True)

        return {"eval_loss": avg_loss, "perplexity": perplexity}

    def _save_checkpoint(self, best: bool = False, final: bool = False):
        """Save model checkpoint."""
        if best:
            path = os.path.join(self.args.output_dir, "best")
        elif final:
            path = os.path.join(self.args.output_dir, "final")
        else:
            path = os.path.join(
                self.args.output_dir, f"checkpoint-{self.global_step}"
            )

        # Unwrap DDP/FSDP/compiled model
        model = self.model
        if hasattr(model, "module"):
            model = model.module
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod

        if hasattr(model, "save_pretrained"):
            model.save_pretrained(path)
        else:
            os.makedirs(path, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(path, "model.pt"))

        if self._is_main_process():
            print(f"Saved checkpoint to {path}")
