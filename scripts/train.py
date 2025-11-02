"""
EchoStream Training Script

Train EchoStream model for simultaneous speech-to-speech translation.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import argparse
import logging
from pathlib import Path
import yaml
import sys
import os

# Add models directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))

from echostream_model import build_echostream_model, EchoStreamConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)


class S2STDataset(Dataset):
    """
    Speech-to-Speech Translation Dataset.
    
    In production, replace with actual data loading from TSV files.
    """
    
    def __init__(self, manifest_path: str, config: dict):
        self.manifest_path = manifest_path
        self.config = config
        
        # Dummy data for demonstration
        self.data = self._load_data()
    
    def _load_data(self):
        """Load data from manifest (TSV format in production)."""
        # Dummy data
        return [
            {
                'speech': torch.randn(100, 80),  # [T, F]
                'speech_length': 100,
                'target_text': torch.randint(0, 6000, (20,)),  # [T_text]
                'target_length': 20,
            }
            for _ in range(100)  # 100 samples
        ]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    """Collate batch for training."""
    # Sort by length
    batch = sorted(batch, key=lambda x: x['speech_length'], reverse=True)
    
    # Get max lengths
    max_speech_len = max(x['speech_length'] for x in batch)
    max_text_len = max(x['target_length'] for x in batch)
    
    # Pad sequences
    speech_batch = []
    speech_lengths = []
    text_batch = []
    text_lengths = []
    
    for item in batch:
        # Pad speech
        speech = item['speech']
        pad_len = max_speech_len - speech.size(0)
        if pad_len > 0:
            speech = torch.cat([speech, torch.zeros(pad_len, 80)], dim=0)
        speech_batch.append(speech)
        speech_lengths.append(item['speech_length'])
        
        # Pad text
        text = item['target_text']
        pad_len = max_text_len - text.size(0)
        if pad_len > 0:
            text = torch.cat([text, torch.zeros(pad_len, dtype=torch.long)], dim=0)
        text_batch.append(text)
        text_lengths.append(item['target_length'])
    
    return {
        'speech': torch.stack(speech_batch),  # [B, T, F]
        'speech_lengths': torch.tensor(speech_lengths),
        'target_text': torch.stack(text_batch),  # [B, T_text]
        'target_lengths': torch.tensor(text_lengths),
    }


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss for EchoStream.
    
    Combines:
    - ASR CTC loss
    - ST CTC loss
    - MT cross-entropy loss
    - Unit CTC loss
    """
    
    def __init__(
        self,
        asr_weight: float = 0.3,
        st_weight: float = 0.3,
        mt_weight: float = 0.2,
        unit_weight: float = 0.2,
        ctc_weight: float = 0.5,
    ):
        super().__init__()
        
        self.asr_weight = asr_weight
        self.st_weight = st_weight
        self.mt_weight = mt_weight
        self.unit_weight = unit_weight
        self.ctc_weight = ctc_weight
        
        # Loss functions
        self.ctc_loss = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=0, reduction='mean')
    
    def forward(self, model_output, target):
        """
        Compute multi-task loss.
        
        Args:
            model_output: Model output dict
            target: Target dict with text/units
        
        Returns:
            total_loss: Scalar loss
            loss_dict: Dict of individual losses
        """
        losses = {}
        
        # ASR CTC Loss (if available)
        if 'asr_log_probs' in model_output and model_output['asr_log_probs'] is not None:
            asr_log_probs = model_output['asr_log_probs']  # [T, B, V]
            # Dummy target for demonstration
            asr_target = torch.randint(1, 6000, (asr_log_probs.size(1), 10))
            asr_target_lengths = torch.full((asr_log_probs.size(1),), 10)
            input_lengths = torch.full((asr_log_probs.size(1),), asr_log_probs.size(0))
            
            losses['asr'] = self.ctc_loss(
                asr_log_probs,
                asr_target,
                input_lengths,
                asr_target_lengths
            ) * self.asr_weight
        
        # ST CTC Loss
        if 'st_log_probs' in model_output and model_output['st_log_probs'] is not None:
            st_log_probs = model_output['st_log_probs']  # [T, B, V]
            st_target = torch.randint(1, 6000, (st_log_probs.size(1), 10))
            st_target_lengths = torch.full((st_log_probs.size(1),), 10)
            input_lengths = torch.full((st_log_probs.size(1),), st_log_probs.size(0))
            
            losses['st'] = self.ctc_loss(
                st_log_probs,
                st_target,
                input_lengths,
                st_target_lengths
            ) * self.st_weight
        
        # MT Loss (if MT decoder was used)
        if 'mt_logits' in model_output and model_output['mt_logits'] is not None:
            mt_logits = model_output['mt_logits']  # [B, T, V]
            mt_target = target['target_text']  # [B, T]
            
            losses['mt'] = self.ce_loss(
                mt_logits.view(-1, mt_logits.size(-1)),
                mt_target.view(-1)
            ) * self.mt_weight
        
        # Unit Loss
        if 'unit_log_probs' in model_output:
            unit_log_probs = model_output['unit_log_probs']  # [B, T_unit, V_unit]
            # Transpose for CTC: [T_unit, B, V_unit]
            unit_log_probs = unit_log_probs.transpose(0, 1)
            
            unit_target = torch.randint(1, 1000, (unit_log_probs.size(1), 50))
            unit_target_lengths = torch.full((unit_log_probs.size(1),), 50)
            input_lengths = torch.full((unit_log_probs.size(1),), unit_log_probs.size(0))
            
            losses['unit'] = self.ctc_loss(
                unit_log_probs,
                unit_target,
                input_lengths,
                unit_target_lengths
            ) * self.unit_weight
        
        # Total loss
        total_loss = sum(losses.values())
        
        return total_loss, losses


def train_epoch(model, dataloader, optimizer, criterion, device, epoch):
    """Train for one epoch."""
    model.train()
    
    total_loss = 0
    total_losses = {}
    
    for batch_idx, batch in enumerate(dataloader):
        # Move to device
        speech = batch['speech'].to(device)
        speech_lengths = batch['speech_lengths'].to(device)
        target_text = batch['target_text'].to(device)
        
        # Forward
        output = model(
            src_tokens=speech,
            src_lengths=speech_lengths,
            prev_output_tokens=target_text[:, :-1],  # Teacher forcing
        )
        
        # Compute loss
        loss, loss_dict = criterion(output, batch)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        
        # Update
        optimizer.step()
        
        # Accumulate
        total_loss += loss.item()
        for k, v in loss_dict.items():
            total_losses[k] = total_losses.get(k, 0) + v.item()
        
        # Log
        if (batch_idx + 1) % 10 == 0:
            logger.info(
                f"Epoch {epoch} | Batch {batch_idx+1}/{len(dataloader)} | "
                f"Loss: {loss.item():.4f}"
            )
    
    # Average
    avg_loss = total_loss / len(dataloader)
    avg_losses = {k: v / len(dataloader) for k, v in total_losses.items()}
    
    return avg_loss, avg_losses


def main(args):
    """Main training function."""
    logger.info("="*70)
    logger.info("EchoStream Training")
    logger.info("="*70)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    
    # Load config
    if args.config:
        with open(args.config) as f:
            config_dict = yaml.safe_load(f)
    else:
        config_dict = {}
    
    # Model config
    config = EchoStreamConfig()
    logger.info(f"Model: {config.encoder_layers}L Emformer + Decoders")
    
    # Build model
    model = build_echostream_model(config)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Dataset
    train_dataset = S2STDataset(args.train_manifest, config_dict)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )
    
    logger.info(f"Training samples: {len(train_dataset)}")
    
    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.98),
        eps=1e-8,
    )
    
    # Loss
    criterion = MultiTaskLoss()
    
    # Training loop
    logger.info("\nStarting training...")
    for epoch in range(1, args.epochs + 1):
        logger.info(f"\nEpoch {epoch}/{args.epochs}")
        
        avg_loss, avg_losses = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )
        
        logger.info(
            f"Epoch {epoch} completed | "
            f"Avg Loss: {avg_loss:.4f} | "
            f"ASR: {avg_losses.get('asr', 0):.4f} | "
            f"ST: {avg_losses.get('st', 0):.4f} | "
            f"MT: {avg_losses.get('mt', 0):.4f} | "
            f"Unit: {avg_losses.get('unit', 0):.4f}"
        )
        
        # Save checkpoint
        if epoch % args.save_interval == 0:
            save_path = Path(args.save_dir) / f"checkpoint_epoch_{epoch}.pt"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': avg_loss,
            }, save_path)
            
            logger.info(f"Checkpoint saved: {save_path}")
    
    logger.info("\n" + "="*70)
    logger.info("Training completed!")
    logger.info("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EchoStream Training")
    
    # Data
    parser.add_argument("--train-manifest", type=str, default="data/train.tsv")
    parser.add_argument("--valid-manifest", type=str, default="data/valid.tsv")
    parser.add_argument("--config", type=str, default=None)
    
    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--num-workers", type=int, default=4)
    
    # Checkpointing
    parser.add_argument("--save-dir", type=str, default="checkpoints/")
    parser.add_argument("--save-interval", type=int, default=10)
    
    args = parser.parse_args()
    
    main(args)

