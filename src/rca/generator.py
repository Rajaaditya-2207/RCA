"""
RCA Generator
=============

High-level text generation interface.

Author: Rajaaditya.R
"""

import torch
import torch.nn.functional as F
from typing import Optional, Iterator


class RCAGenerator:
    """
    High-level generation wrapper for RCA models.

    Usage:
        generator = RCAGenerator(model, tokenizer)
        text = generator.generate("Hello, world!")
    """

    def __init__(self, model, tokenizer=None, device: Optional[str] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)
        self.model.eval()

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> str:
        """Generate text from a prompt."""
        assert self.tokenizer is not None, "Tokenizer required for string input"

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        output_ids = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def generate_from_ids(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        """Generate token IDs from input IDs."""
        input_ids = input_ids.to(self.device)
        return self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

    @torch.no_grad()
    def stream(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> Iterator[str]:
        """Stream tokens one at a time."""
        assert self.tokenizer is not None, "Tokenizer required for streaming"

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        out = self.model(input_ids, use_cache=True, use_cuda=True)
        logits = out.logits[:, -1, :]
        ssm_states = out.ssm_states

        for _ in range(max_new_tokens):
            logits = logits / max(temperature, 1e-8)

            if top_k > 0:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, -1:]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)

            token_str = self.tokenizer.decode(next_token[0])
            yield token_str

            if next_token.item() == self.model.config.eos_token_id:
                break

            out = self.model(next_token, ssm_states=ssm_states, use_cache=True)
            logits = out.logits[:, -1, :]
            ssm_states = out.ssm_states
