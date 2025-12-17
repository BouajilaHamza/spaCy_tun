#!/usr/bin/env python3
"""Train a linguistic quality scoring encoder for Tunisian Derja.

This script trains a transformer-based encoder that predicts:
1. Authenticity score (regression)
2. Quality tier (classification: high/medium/low)
3. Feature presence (multi-label classification)

Usage:
    python scripts/train_quality_encoder.py --data-path data/processed/tunisian_derja_encoder_data.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass


@dataclass
class EncoderConfig:
    """Configuration for the quality encoder."""
    
    vocab_size: int = 50000
    hidden_size: int = 256
    num_layers: int = 4
    num_heads: int = 4
    dropout: float = 0.1
    max_length: int = 512
    
    # Output heads
    num_quality_tiers: int = 3  # high, medium, low
    num_binary_features: int = 7  # Various linguistic features


class SimpleTokenizer:
    """Simple character-level tokenizer for Arabic text."""
    
    def __init__(self, max_length: int = 512):
        self.max_length = max_length
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.bos_token_id = 2
        self.eos_token_id = 3
        
        # Build vocab from Arabic unicode range + common chars
        self.char_to_id = {
            "<pad>": 0,
            "<unk>": 1,
            "<bos>": 2,
            "<eos>": 3,
        }
        
        # Arabic letters and diacritics
        for i, c in enumerate(range(0x0600, 0x06FF)):
            self.char_to_id[chr(c)] = i + 4
        
        # Latin letters (for Arabizi and code-switching)
        for i, c in enumerate("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"):
            self.char_to_id[c] = len(self.char_to_id)
        
        # Numbers and punctuation
        for c in "0123456789.,!?;:'\"()-/ \n\t":
            if c not in self.char_to_id:
                self.char_to_id[c] = len(self.char_to_id)
        
        self.id_to_char = {v: k for k, v in self.char_to_id.items()}
        self.vocab_size = len(self.char_to_id)
    
    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs."""
        ids = [self.bos_token_id]
        for c in text[:self.max_length - 2]:
            ids.append(self.char_to_id.get(c, self.unk_token_id))
        ids.append(self.eos_token_id)
        return ids
    
    def pad(self, ids: list[int]) -> list[int]:
        """Pad sequence to max_length."""
        if len(ids) >= self.max_length:
            return ids[:self.max_length]
        return ids + [self.pad_token_id] * (self.max_length - len(ids))


class TunisianQualityDataset(Dataset):
    """Dataset for training the quality encoder."""
    
    QUALITY_TIER_MAP = {"high": 0, "medium": 1, "low": 2}
    
    BINARY_FEATURES = [
        "is_authentic",
        "has_negation_circumfix",
        "has_discourse_markers",
        "has_interrogatives",
        "has_wh_in_situ",
        "has_arabizi",
        "has_msa_markers",
    ]
    
    def __init__(self, data_path: Path, tokenizer: SimpleTokenizer):
        self.tokenizer = tokenizer
        self.samples = []
        
        with data_path.open("r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                self.samples.append(record)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        text = sample["text"]
        features = sample["features"]
        
        # Tokenize
        input_ids = self.tokenizer.pad(self.tokenizer.encode(text))
        
        # Targets
        authenticity_score = features["authenticity_score"]
        quality_tier = self.QUALITY_TIER_MAP[features["quality_tier"]]
        
        # Binary features
        binary_features = [
            1.0 if features.get(feat, False) else 0.0
            for feat in self.BINARY_FEATURES
        ]
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "authenticity_score": torch.tensor(authenticity_score, dtype=torch.float),
            "quality_tier": torch.tensor(quality_tier, dtype=torch.long),
            "binary_features": torch.tensor(binary_features, dtype=torch.float),
        }


class TunisianQualityEncoder(nn.Module):
    """Transformer encoder for Tunisian text quality scoring."""
    
    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config
        
        # Embedding
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.pos_embedding = nn.Embedding(config.max_length, config.hidden_size)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_size * 4,
            dropout=config.dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        
        # Pooling
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # Output heads
        self.score_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size // 2, 1),
        )
        
        self.tier_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size // 2, config.num_quality_tiers),
        )
        
        self.feature_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size // 2, config.num_binary_features),
            nn.Sigmoid(),
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Returns:
            - score: authenticity score (batch_size, 1)
            - tier_logits: quality tier logits (batch_size, num_tiers)
            - feature_probs: binary feature probabilities (batch_size, num_features)
        """
        batch_size, seq_len = input_ids.shape
        
        # Create attention mask from padding
        if attention_mask is None:
            attention_mask = (input_ids != 0).float()
        
        # Embeddings
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        x = self.embedding(input_ids) + self.pos_embedding(positions)
        
        # Create transformer mask (True = ignore)
        src_key_padding_mask = (attention_mask == 0)
        
        # Transformer encoding
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        
        # Pool to single vector (mean pooling over non-padded tokens)
        x = x * attention_mask.unsqueeze(-1)
        x = x.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True).clamp(min=1)
        
        # Output heads
        score = self.score_head(x)
        tier_logits = self.tier_head(x)
        feature_probs = self.feature_head(x)
        
        return score, tier_logits, feature_probs
    
    def encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get pooled representation for downstream tasks."""
        batch_size, seq_len = input_ids.shape
        attention_mask = (input_ids != 0).float()
        
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        x = self.embedding(input_ids) + self.pos_embedding(positions)
        
        src_key_padding_mask = (attention_mask == 0)
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        
        x = x * attention_mask.unsqueeze(-1)
        x = x.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True).clamp(min=1)
        
        return x


class QualityEncoderTrainer:
    """Trainer for the quality encoder."""
    
    def __init__(
        self,
        model: TunisianQualityEncoder,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        lr: float = 1e-4,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        
        # Loss functions
        self.score_loss = nn.MSELoss()
        self.tier_loss = nn.CrossEntropyLoss()
        self.feature_loss = nn.BCELoss()
    
    def train_epoch(self) -> dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        score_loss_sum = 0.0
        tier_loss_sum = 0.0
        feature_loss_sum = 0.0
        n_batches = 0
        
        for batch in self.train_loader:
            input_ids = batch["input_ids"].to(self.device)
            target_score = batch["authenticity_score"].to(self.device)
            target_tier = batch["quality_tier"].to(self.device)
            target_features = batch["binary_features"].to(self.device)
            
            self.optimizer.zero_grad()
            
            pred_score, pred_tier, pred_features = self.model(input_ids)
            
            # Compute losses
            loss_score = self.score_loss(pred_score.squeeze(-1), target_score)
            loss_tier = self.tier_loss(pred_tier, target_tier)
            loss_features = self.feature_loss(pred_features, target_features)
            
            # Combined loss (weighted)
            loss = loss_score + 0.5 * loss_tier + 0.3 * loss_features
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            score_loss_sum += loss_score.item()
            tier_loss_sum += loss_tier.item()
            feature_loss_sum += loss_features.item()
            n_batches += 1
        
        return {
            "total_loss": total_loss / n_batches,
            "score_loss": score_loss_sum / n_batches,
            "tier_loss": tier_loss_sum / n_batches,
            "feature_loss": feature_loss_sum / n_batches,
        }
    
    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        """Evaluate on validation set."""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        
        total_loss = 0.0
        correct_tier = 0
        total_samples = 0
        score_mae = 0.0
        
        for batch in self.val_loader:
            input_ids = batch["input_ids"].to(self.device)
            target_score = batch["authenticity_score"].to(self.device)
            target_tier = batch["quality_tier"].to(self.device)
            target_features = batch["binary_features"].to(self.device)
            
            pred_score, pred_tier, pred_features = self.model(input_ids)
            
            # Losses
            loss_score = self.score_loss(pred_score.squeeze(-1), target_score)
            loss_tier = self.tier_loss(pred_tier, target_tier)
            loss_features = self.feature_loss(pred_features, target_features)
            loss = loss_score + 0.5 * loss_tier + 0.3 * loss_features
            
            total_loss += loss.item() * input_ids.size(0)
            
            # Accuracy metrics
            pred_tier_labels = pred_tier.argmax(dim=-1)
            correct_tier += (pred_tier_labels == target_tier).sum().item()
            total_samples += input_ids.size(0)
            
            # MAE for score
            score_mae += (pred_score.squeeze(-1) - target_score).abs().sum().item()
        
        return {
            "val_loss": total_loss / total_samples,
            "val_tier_accuracy": correct_tier / total_samples,
            "val_score_mae": score_mae / total_samples,
        }


def main():
    parser = argparse.ArgumentParser(description="Train Tunisian quality encoder")
    parser.add_argument("--data-path", type=str, required=True,
                        help="Path to encoder training data (JSONL)")
    parser.add_argument("--output-dir", type=str, default="models/quality_encoder",
                        help="Output directory for model")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--hidden-size", type=int, default=256,
                        help="Hidden size")
    parser.add_argument("--num-layers", type=int, default=4,
                        help="Number of transformer layers")
    parser.add_argument("--val-split", type=float, default=0.1,
                        help="Validation split ratio")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use")
    
    args = parser.parse_args()
    
    data_path = Path(args.data_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading data from {data_path}...")
    
    # Initialize tokenizer
    tokenizer = SimpleTokenizer()
    
    # Create dataset
    full_dataset = TunisianQualityDataset(data_path, tokenizer)
    print(f"Loaded {len(full_dataset)} samples")
    
    # Split into train/val
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    # Initialize model
    config = EncoderConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
    )
    model = TunisianQualityEncoder(config)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    trainer = QualityEncoderTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=args.lr,
        device=args.device,
    )
    
    print(f"\nTraining on {args.device}...")
    best_val_loss = float("inf")
    
    for epoch in range(args.epochs):
        train_metrics = trainer.train_epoch()
        val_metrics = trainer.evaluate()
        
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print(f"  Train - Loss: {train_metrics['total_loss']:.4f}, "
              f"Score: {train_metrics['score_loss']:.4f}, "
              f"Tier: {train_metrics['tier_loss']:.4f}")
        
        if val_metrics:
            print(f"  Val   - Loss: {val_metrics['val_loss']:.4f}, "
                  f"Tier Acc: {val_metrics['val_tier_accuracy']:.4f}, "
                  f"Score MAE: {val_metrics['val_score_mae']:.4f}")
            
            if val_metrics["val_loss"] < best_val_loss:
                best_val_loss = val_metrics["val_loss"]
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "tokenizer_vocab_size": tokenizer.vocab_size,
                }, output_dir / "best_model.pt")
                print("  -> Saved best model")
    
    # Save final model
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config,
        "tokenizer_vocab_size": tokenizer.vocab_size,
    }, output_dir / "final_model.pt")
    
    print(f"\nTraining complete. Models saved to {output_dir}")


if __name__ == "__main__":
    main()
