"""
EchoStream Training Script

Train EchoStream model for simultaneous speech-to-speech translation.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import pickle
import argparse
import logging
from pathlib import Path
import yaml
import sys
import os
from typing import Optional

# Add project directories to path
ROOT_DIR = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.abspath(ROOT_DIR))
sys.path.insert(0, os.path.join(os.path.abspath(ROOT_DIR), 'models'))

from echostream_model import build_echostream_model, EchoStreamConfig
from datasets import S2STManifestDataset, collate_s2st_batches

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)


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

    @staticmethod
    def _flatten_tokens(tokens: torch.Tensor, lengths: torch.Tensor, *, exclude_last: bool = False) -> Optional[torch.Tensor]:
        sequences = []
        for seq, length in zip(tokens, lengths):
            length_int = int(length.item())
            if exclude_last and length_int > 0:
                length_int -= 1
            if length_int <= 0:
                continue
            sequences.append(seq[:length_int])
        if not sequences:
            return None
        return torch.cat(sequences, dim=0)
    
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
            asr_target = target.get('src_tokens')
            asr_target_lengths = target.get('src_lengths')
            if asr_target is not None and asr_target_lengths is not None:
                flat_targets = self._flatten_tokens(asr_target, asr_target_lengths, exclude_last=False)
                if flat_targets is not None and flat_targets.numel() > 0:
                    input_lengths = torch.full(
                        (asr_log_probs.size(1),),
                        asr_log_probs.size(0),
                        dtype=torch.long,
                        device=asr_log_probs.device,
                    )
                    losses['asr'] = self.ctc_loss(
                        asr_log_probs,
                        flat_targets,
                        input_lengths,
                        asr_target_lengths.to(asr_log_probs.device),
                    ) * self.asr_weight
        
        # ST CTC Loss
        if 'st_log_probs' in model_output and model_output['st_log_probs'] is not None:
            st_log_probs = model_output['st_log_probs']  # [T, B, V]
            tgt_tokens = target.get('target_text')
            tgt_lengths = target.get('target_lengths')
            if tgt_tokens is not None and tgt_lengths is not None:
                st_lengths = torch.clamp(tgt_lengths - 1, min=0)
                flat_targets = self._flatten_tokens(tgt_tokens, tgt_lengths, exclude_last=True)
                if flat_targets is not None and flat_targets.numel() > 0:
                    input_lengths = torch.full(
                        (st_log_probs.size(1),),
                        st_log_probs.size(0),
                        dtype=torch.long,
                        device=st_log_probs.device,
                    )
                    losses['st'] = self.ctc_loss(
                        st_log_probs,
                        flat_targets,
                        input_lengths,
                        st_lengths.to(st_log_probs.device),
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
        if 'unit_log_probs' in model_output and model_output['unit_log_probs'] is not None:
            unit_log_probs = model_output['unit_log_probs']  # [B, T_unit, V_unit]
            tgt_units = target.get('tgt_units')
            tgt_unit_lengths = target.get('tgt_unit_lengths')
            if tgt_units is not None and tgt_unit_lengths is not None:
                # Transpose for CTC: [T_unit, B, V_unit]
                unit_log_probs = unit_log_probs.transpose(0, 1)
                flat_units = self._flatten_tokens(tgt_units, tgt_unit_lengths, exclude_last=False)
                if flat_units is not None and flat_units.numel() > 0:
                    input_lengths = torch.full(
                        (unit_log_probs.size(1),),
                        unit_log_probs.size(0),
                        dtype=torch.long,
                        device=unit_log_probs.device,
                    )
                    losses['unit'] = self.ctc_loss(
                        unit_log_probs,
                        flat_units.to(unit_log_probs.device),
                        input_lengths,
                        tgt_unit_lengths.to(unit_log_probs.device),
                    ) * self.unit_weight
        
        # Total loss
        total_loss = sum(losses.values())
        
        return total_loss, losses


def train_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    device,
    epoch,
    *,
    scaler: Optional[GradScaler] = None,
    use_amp: bool = False,
    retain_graph: bool = False,
):
    """Train for one epoch."""
    model.train()
    
    total_loss = 0
    total_losses = {}
    
    for batch_idx, batch in enumerate(dataloader):
        # Move to device
        speech = batch['speech'].to(device)
        speech_lengths = batch['speech_lengths'].to(device)
        prev_output_tokens = batch['prev_output_tokens'].to(device)
        target_text = batch['target_text'].to(device)
        target_lengths = batch['target_lengths'].to(device)
        src_tokens = batch['src_tokens'].to(device)
        src_lengths = batch['src_lengths'].to(device)
        tgt_units = batch.get('tgt_units')
        tgt_unit_lengths = batch.get('tgt_unit_lengths')
        
        # Reset streaming cache per batch to avoid stale graph references
        if hasattr(model, "reset_cache"):
            model.reset_cache()
        
        optimizer.zero_grad(set_to_none=True)
        
        with autocast(enabled=use_amp):
            output = model(
                src_tokens=speech,
                src_lengths=speech_lengths,
                prev_output_tokens=prev_output_tokens,
                target_lengths=target_lengths,
            )
            
            target_dict = {
                'src_tokens': src_tokens,
                'src_lengths': src_lengths,
                'target_text': target_text,
                'target_lengths': target_lengths,
            }
            if tgt_units is not None and tgt_unit_lengths is not None:
                target_dict['tgt_units'] = tgt_units.to(device)
                target_dict['tgt_unit_lengths'] = tgt_unit_lengths.to(device)
            
            loss, loss_dict = criterion(output, target_dict)
        
        if use_amp and scaler is not None:
            scaler.scale(loss).backward(retain_graph=retain_graph)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward(retain_graph=retain_graph)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
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


@torch.no_grad()
def evaluate(model, dataloader, criterion, device, *, use_amp: bool = False):
    """Evaluate model on validation/test set."""
    model.eval()
    
    if dataloader is None or len(dataloader) == 0:
        return None, {}
    
    total_loss = 0.0
    total_losses = {}
    
    for batch in dataloader:
        speech = batch['speech'].to(device)
        speech_lengths = batch['speech_lengths'].to(device)
        prev_output_tokens = batch['prev_output_tokens'].to(device)
        target_text = batch['target_text'].to(device)
        target_lengths = batch['target_lengths'].to(device)
        src_tokens = batch['src_tokens'].to(device)
        src_lengths = batch['src_lengths'].to(device)
        tgt_units = batch.get('tgt_units')
        tgt_unit_lengths = batch.get('tgt_unit_lengths')
        
        if hasattr(model, "reset_cache"):
            model.reset_cache()
        
        with autocast(enabled=use_amp):
            output = model(
                src_tokens=speech,
                src_lengths=speech_lengths,
                prev_output_tokens=prev_output_tokens,
                target_lengths=target_lengths,
            )
            
            target_dict = {
                'src_tokens': src_tokens,
                'src_lengths': src_lengths,
                'target_text': target_text,
                'target_lengths': target_lengths,
            }
            if tgt_units is not None and tgt_unit_lengths is not None:
                target_dict['tgt_units'] = tgt_units.to(device)
                target_dict['tgt_unit_lengths'] = tgt_unit_lengths.to(device)
            
            loss, loss_dict = criterion(output, target_dict)
        
        total_loss += loss.item()
        for k, v in loss_dict.items():
            total_losses[k] = total_losses.get(k, 0) + v.item()
    
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
    if getattr(args, "detect_anomaly", False):
        torch.autograd.set_detect_anomaly(True)
        logger.warning("Autograd anomaly detection enabled. This will slow down training.")
    
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
    
    hardware_cfg = config_dict.get('hardware', {})
    use_amp = hardware_cfg.get('fp16', False) or getattr(args, "fp16", False)
    if device.type != "cuda":
        use_amp = False
    if use_amp:
        logger.info("Mixed precision training enabled (AMP).")
    scaler = GradScaler(enabled=use_amp)
    
    data_cfg = config_dict.get('data', {})
    config_dir = Path(args.config).resolve().parent if args.config else ROOT_DIR

    def resolve_optional_path(value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        path = Path(value)
        if not path.is_absolute():
            path = (config_dir / path).resolve()
        return str(path)

    train_manifest = args.train_manifest or data_cfg.get('train_manifest')
    train_manifest_path = resolve_optional_path(train_manifest)
    if train_manifest_path is None:
        raise ValueError("Training manifest must be specified via --train-manifest or config[data.train_manifest].")
    dev_manifest = args.dev_manifest or data_cfg.get('valid_manifest')
    dev_manifest_path = resolve_optional_path(dev_manifest) if dev_manifest else None
    test_manifest = args.test_manifest or data_cfg.get('test_manifest')
    test_manifest_path = resolve_optional_path(test_manifest) if test_manifest else None

    data_root = resolve_optional_path(data_cfg.get('data_root'))
    global_cmvn = resolve_optional_path(data_cfg.get('global_cmvn_stats_npz'))
    units_root = resolve_optional_path(data_cfg.get('units_root'))
    src_vocab_path = resolve_optional_path(data_cfg.get('src_dict'))
    tgt_vocab_path = resolve_optional_path(data_cfg.get('tgt_dict'))

    streaming_cfg = config_dict.get('streaming', {})
    streaming_chunk_ms = streaming_cfg.get('chunk_size_ms', streaming_cfg.get('chunk_size'))
    streaming_hop_ms = streaming_cfg.get('chunk_hop_ms', streaming_cfg.get('chunk_hop'))

    train_dataset = S2STManifestDataset(
        manifest_path=train_manifest_path,
        data_root=data_root,
        units_root=units_root,
        sample_rate=data_cfg.get('sample_rate', 16000),
        num_mel_bins=data_cfg.get('num_mel_bins', 80),
        src_vocab_path=src_vocab_path,
        tgt_vocab_path=tgt_vocab_path,
        text_level=data_cfg.get('tokenize_level', 'word'),
        global_cmvn_stats=global_cmvn,
        load_waveform=data_cfg.get('load_waveform', False),
        load_tgt_audio=data_cfg.get('load_tgt_audio', False),
        load_tgt_units=data_cfg.get('load_tgt_units', False),
        streaming_chunk_ms=streaming_chunk_ms,
        streaming_hop_ms=streaming_hop_ms,
        min_duration=data_cfg.get('min_duration'),
        max_duration=data_cfg.get('max_duration'),
        pad_value=data_cfg.get('pad_value', 0.0),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_s2st_batches,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    
    logger.info(f"Training samples: {len(train_dataset)}")
    
    def build_eval_loader(manifest_path: Optional[str]):
        if manifest_path is None:
            return None
        dataset = S2STManifestDataset(
            manifest_path=manifest_path,
            data_root=data_root,
            units_root=units_root,
            sample_rate=data_cfg.get('sample_rate', 16000),
            num_mel_bins=data_cfg.get('num_mel_bins', 80),
            src_vocab_path=src_vocab_path,
            tgt_vocab_path=tgt_vocab_path,
            text_level=data_cfg.get('tokenize_level', 'word'),
            global_cmvn_stats=global_cmvn,
            load_waveform=data_cfg.get('load_waveform', False),
            load_tgt_audio=data_cfg.get('load_tgt_audio', False),
            load_tgt_units=data_cfg.get('load_tgt_units', False),
            streaming_chunk_ms=streaming_chunk_ms,
            streaming_hop_ms=streaming_hop_ms,
            min_duration=data_cfg.get('min_duration'),
            max_duration=data_cfg.get('max_duration'),
            pad_value=data_cfg.get('pad_value', 0.0),
        )
        return DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_s2st_batches,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
    
    dev_loader = build_eval_loader(dev_manifest_path)
    test_loader = build_eval_loader(test_manifest_path)
    
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
    metrics_history = []
    for epoch in range(1, args.epochs + 1):
        logger.info(f"\nEpoch {epoch}/{args.epochs}")
        
        avg_loss, avg_losses = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            epoch,
            scaler=scaler,
            use_amp=use_amp,
            retain_graph=getattr(args, "retain_graph", False),
        )
        
        logger.info(
            f"Epoch {epoch} completed | "
            f"Avg Loss: {avg_loss:.4f} | "
            f"ASR: {avg_losses.get('asr', 0):.4f} | "
            f"ST: {avg_losses.get('st', 0):.4f} | "
            f"MT: {avg_losses.get('mt', 0):.4f} | "
            f"Unit: {avg_losses.get('unit', 0):.4f}"
        )
        
        epoch_metrics = {
            "epoch": epoch,
            "train": {
                "loss": avg_loss,
                "components": avg_losses,
            },
        }
        
        if dev_loader is not None:
            dev_loss, dev_losses = evaluate(
                model,
                dev_loader,
                criterion,
                device,
                use_amp=use_amp,
            )
            logger.info(
                f"[Dev] Loss: {dev_loss:.4f} | "
                f"ASR: {dev_losses.get('asr', 0):.4f} | "
                f"ST: {dev_losses.get('st', 0):.4f} | "
                f"MT: {dev_losses.get('mt', 0):.4f} | "
                f"Unit: {dev_losses.get('unit', 0):.4f}"
            )
            epoch_metrics["dev"] = {"loss": dev_loss, "components": dev_losses}
        
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
            epoch_metrics["checkpoint"] = str(save_path)
        
        metrics_history.append(epoch_metrics)
    
    test_metrics = None
    if test_loader is not None:
        test_loss, test_losses = evaluate(
            model,
            test_loader,
            criterion,
            device,
            use_amp=use_amp,
        )
        logger.info(
            f"\n[Test] Loss: {test_loss:.4f} | "
            f"ASR: {test_losses.get('asr', 0):.4f} | "
            f"ST: {test_losses.get('st', 0):.4f} | "
            f"MT: {test_losses.get('mt', 0):.4f} | "
            f"Unit: {test_losses.get('unit', 0):.4f}"
        )
        test_metrics = {"loss": test_loss, "components": test_losses}
    
    if args.metrics_output:
        metrics_output_path = Path(args.metrics_output)
        metrics_output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": args.config,
            "epochs": args.epochs,
            "metrics": metrics_history,
            "test": test_metrics,
        }
        with open(metrics_output_path, "wb") as f:
            pickle.dump(payload, f)
        logger.info(f"Metrics saved to {metrics_output_path}")
    
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
    parser.add_argument("--fp16", action="store_true", help="Enable mixed precision training (AMP).")
    parser.add_argument("--detect-anomaly", action="store_true", help="Enable torch autograd anomaly detection.")
    parser.add_argument("--retain-graph", action="store_true", help="Call backward(retain_graph=True) for debugging.")
    
    # Evaluation / checkpointing
    parser.add_argument("--dev-manifest", type=str, default=None)
    parser.add_argument("--test-manifest", type=str, default=None)
    parser.add_argument("--save-dir", type=str, default="checkpoints/")
    parser.add_argument("--save-interval", type=int, default=10)
    parser.add_argument("--metrics-output", type=str, default=None)
    
    args = parser.parse_args()
    
    main(args)

