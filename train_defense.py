"""
Train defensive token embeddings against prompt injection attacks.

Uses injected images from the experiment as training data. The correct
category labels are known, so we optimize the defensive tokens to make
the model output the correct label despite injection.

Usage:
    python train_defense.py --epochs 50 --sample-size 3
    python train_defense.py --epochs 100 --num-tokens 15 --lr 5e-4
"""

import argparse
import random
from pathlib import Path

import torch

import config
from defend import DefensiveTokenModel
from inject_prompt import generate_injected_dataset


def create_training_data(sample_size=3):
    """Generate injected images and return (path, label) pairs."""
    manifest = generate_injected_dataset(sample_size)
    train_data = [(m["injected_path"], m["category"]) for m in manifest]
    random.shuffle(train_data)
    return train_data


def train(
    num_tokens=None,
    epochs=50,
    lr=1e-3,
    sample_size=3,
):
    num_tokens = num_tokens or config.DEFENSE_NUM_TOKENS

    # ── Create training data ──────────────────────────────────────
    print("=" * 60)
    print("  DEFENSIVE TOKEN TRAINING")
    print("=" * 60)
    print(f"\n  Model:          {config.DEFENSE_MODEL}")
    print(f"  Def. tokens:    {num_tokens}")
    print(f"  Epochs:         {epochs}")
    print(f"  Learning rate:  {lr}")
    print(f"  Sample size:    {sample_size} images/category")
    print()

    print("─" * 60)
    print("STEP 1: Generating injected training images")
    print("─" * 60)
    train_data = create_training_data(sample_size)
    print(f"  Training samples: {len(train_data)}\n")

    # ── Load model ────────────────────────────────────────────────
    print("─" * 60)
    print("STEP 2: Loading model + adding defensive tokens")
    print("─" * 60)
    model = DefensiveTokenModel(
        model_name=config.DEFENSE_MODEL,
        num_tokens=num_tokens,
    )

    # ── Setup training ────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("STEP 3: Training")
    print("─" * 60)
    trainable_params = model.freeze_for_training()
    optimizer = torch.optim.AdamW(trainable_params, lr=lr)

    best_loss = float("inf")

    for epoch in range(epochs):
        random.shuffle(train_data)
        total_loss = 0.0
        num_samples = 0

        for img_path, label in train_data:
            try:
                loss = model.train_step(img_path, label, optimizer)
                total_loss += loss
                num_samples += 1
            except Exception as e:
                print(f"    [SKIP] {Path(img_path).name}: {e}")

        avg_loss = total_loss / max(num_samples, 1)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            marker = " ★" if avg_loss < best_loss else ""
            print(f"  Epoch {epoch + 1:>3}/{epochs}  loss={avg_loss:.4f}{marker}")

        if avg_loss < best_loss:
            best_loss = avg_loss

    # ── Save ──────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("STEP 4: Saving optimized embeddings")
    print("─" * 60)
    save_path = config.DEFENSE_DIR / "optimized_embeddings.pt"
    model.save_embeddings(save_path)

    print(f"\n{'=' * 60}")
    print(f"  TRAINING COMPLETE — best loss: {best_loss:.4f}")
    print(f"{'=' * 60}\n")

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train defensive token embeddings against prompt injection",
    )
    parser.add_argument(
        "--num-tokens", type=int, default=None,
        help=f"Number of defensive tokens (default: {config.DEFENSE_NUM_TOKENS})",
    )
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="Training epochs (default: 50)",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="Learning rate (default: 1e-3)",
    )
    parser.add_argument(
        "--sample-size", type=int, default=3,
        help="Images per category for training data (default: 3)",
    )
    args = parser.parse_args()

    random.seed(42)
    torch.manual_seed(42)

    train(
        num_tokens=args.num_tokens,
        epochs=args.epochs,
        lr=args.lr,
        sample_size=args.sample_size,
    )
