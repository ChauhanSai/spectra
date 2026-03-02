"""
Defensive Tokens — Neural network-level defense against prompt injection.

Adds K learnable tokens to the model's vocabulary. Their embeddings are
optimized so that when prepended to the classification prompt, the model
resists injected text overlaid on images.

Based on: "Defending Against Prompt Injection With a Few Defensive Tokens"
(Chen et al., ACM AISec 2025)
"""

import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image


class DefensiveTokenModel:
    """Wraps a Qwen2.5-VL model with optimizable defensive tokens."""

    def __init__(self, model_name="Qwen/Qwen2.5-VL-3B-Instruct", num_tokens=10, device="auto"):
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        self.model_name = model_name
        self.num_tokens = num_tokens
        self.token_names = [f"<def_{i}>" for i in range(num_tokens)]

        print(f"Loading {model_name}...")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=device,
        )

        # Add defensive tokens to vocabulary
        num_added = self.processor.tokenizer.add_tokens(
            self.token_names, special_tokens=True
        )
        self.model.resize_token_embeddings(len(self.processor.tokenizer))
        self.def_token_ids = self.processor.tokenizer.convert_tokens_to_ids(
            self.token_names
        )

        print(f"Added {num_added} defensive tokens (IDs: {self.def_token_ids})")

    # ── Prompt helpers ────────────────────────────────────────────

    def get_defensive_prefix(self):
        """Return the defensive token string to prepend to prompts."""
        return "".join(self.token_names)

    def _build_messages(self, image, prompt_text):
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

    def _process_messages(self, messages, image):
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text], images=[image], return_tensors="pt", padding=True
        )
        return inputs.to(self.model.device)

    # ── Training ──────────────────────────────────────────────────

    def freeze_for_training(self):
        """Freeze all params except defensive token embeddings.

        Uses a gradient mask hook so only the K defensive rows
        in the embedding matrix receive gradient updates.
        Returns the list of trainable parameter tensors.
        """
        for p in self.model.parameters():
            p.requires_grad = False

        embed_layer = self.model.get_input_embeddings()
        embed_layer.weight.requires_grad = True

        def_ids_set = set(self.def_token_ids)
        vocab_size = embed_layer.weight.shape[0]

        # Pre-compute mask once (1.0 for defensive rows, 0.0 elsewhere)
        grad_mask = torch.zeros(vocab_size, 1, device=embed_layer.weight.device)
        for tid in def_ids_set:
            grad_mask[tid] = 1.0

        embed_layer.weight.register_hook(lambda grad: grad * grad_mask)

        trainable = sum(self.num_tokens * embed_layer.weight.shape[1] for _ in [1])
        total = sum(p.numel() for p in self.model.parameters())
        print(
            f"Trainable: {trainable:,} params "
            f"({trainable / total * 100:.4f}% of {total:,})"
        )

        return [embed_layer.weight]

    def train_step(self, image_path, correct_label, optimizer):
        """One training step: optimize defensive tokens for one image.

        Returns the scalar loss value.
        """
        image = Image.open(image_path).convert("RGB")

        prompt = (
            f"{self.get_defensive_prefix()} "
            "What is the object in this image? "
            "Reply with ONLY a single word or short phrase. Do not explain."
        )

        messages = self._build_messages(image, prompt)
        inputs = self._process_messages(messages, image)

        # Target: first token of the correct category label
        label_token_id = self.processor.tokenizer.encode(
            correct_label, add_special_tokens=False
        )[0]

        # Forward pass
        with torch.amp.autocast("cuda", dtype=torch.float16):
            outputs = self.model(**inputs)

        # Logits at last input position predict the first generated token
        logits = outputs.logits[:, -1, :].float()
        target = torch.tensor([label_token_id], device=logits.device)
        loss = F.cross_entropy(logits, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()

    # ── Inference ─────────────────────────────────────────────────

    def classify(self, image_path, use_defense=True, max_new_tokens=20):
        """Classify an image, optionally using defensive tokens."""
        image = Image.open(image_path).convert("RGB")

        prompt = (
            "What is the object in this image? "
            "Reply with ONLY a single word or short phrase. Do not explain."
        )
        if use_defense:
            prompt = f"{self.get_defensive_prefix()} {prompt}"

        messages = self._build_messages(image, prompt)
        inputs = self._process_messages(messages, image)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False
            )

        generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
        response = self.processor.tokenizer.decode(
            generated_ids[0], skip_special_tokens=True
        ).strip()
        return response

    # ── Persistence ───────────────────────────────────────────────

    def save_embeddings(self, path):
        """Save only the defensive token embeddings (~50 KB)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        embed_layer = self.model.get_input_embeddings()
        def_embeddings = embed_layer.weight[self.def_token_ids].detach().cpu()

        torch.save(
            {
                "embeddings": def_embeddings,
                "token_names": self.token_names,
                "def_token_ids": self.def_token_ids,
                "num_tokens": self.num_tokens,
                "model_name": self.model_name,
            },
            path,
        )
        print(f"Saved defensive embeddings → {path}  (shape: {def_embeddings.shape})")

    def load_embeddings(self, path):
        """Load previously optimized defensive token embeddings."""
        data = torch.load(path, map_location="cpu", weights_only=True)

        embed_layer = self.model.get_input_embeddings()
        with torch.no_grad():
            embed_layer.weight[self.def_token_ids] = data["embeddings"].to(
                embed_layer.weight.device, dtype=embed_layer.weight.dtype
            )
        print(f"Loaded defensive embeddings ← {path}")
