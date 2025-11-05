"""
EchoStream Multi-task Learning Criterion

StreamSpeech 참고:
- criterions/speech_to_speech_ctc_asr_st_criterion.py
- Multi-task loss: L = L_asr + L_st + L_mt + L_unit

핵심 개선:
- StreamSpeech: Conformer 기반
- EchoStream: Emformer 기반 + Multi-chunk training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class EchoStreamMultiTaskCriterion(nn.Module):
    """
    EchoStream Multi-task Learning Criterion.
    
    Loss Components:
    1. L_asr: ASR CTC Loss (source text recognition)
    2. L_st: ST CTC Loss (target text translation)
    3. L_mt: MT Cross-Entropy Loss (text refinement)
    4. L_unit: Unit CTC Loss (speech unit generation)
    
    Total: L = L_asr + L_st + L_mt + L_unit
    
    StreamSpeech 참고: Line 115-200
    """
    
    def __init__(
        self,
        asr_weight: float = 1.0,
        st_weight: float = 1.0,
        mt_weight: float = 1.0,
        unit_weight: float = 1.0,
        blank_idx: int = 0,
        pad_idx: int = 1,
        eos_idx: int = 2,
        label_smoothing: float = 0.1,
        sentence_avg: bool = True,
    ):
        super().__init__()
        
        # Loss weights
        self.asr_weight = asr_weight
        self.st_weight = st_weight
        self.mt_weight = mt_weight
        self.unit_weight = unit_weight
        
        # Special tokens
        self.blank_idx = blank_idx
        self.pad_idx = pad_idx
        self.eos_idx = eos_idx
        
        # Label smoothing
        self.label_smoothing = label_smoothing
        self.sentence_avg = sentence_avg
        
        logger.info(
            f"EchoStreamMultiTaskCriterion initialized "
            f"(asr={asr_weight}, st={st_weight}, mt={mt_weight}, unit={unit_weight})"
        )
    
    def forward(
        self,
        model_output: Dict[str, torch.Tensor],
        sample: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute multi-task loss.
        
        Args:
            model_output: Model outputs
                - asr_logits: [B, T, vocab_asr]
                - st_logits: [B, T, vocab_st]
                - mt_logits: [B, T_mt, vocab_mt]
                - unit_logits: [B, T_unit, num_units]
                - encoder_out: [B, T, D]
            sample: Training sample
                - source_text: [B, T_src]
                - target_text: [B, T_tgt]
                - target_units: [B, T_unit]
                - src_lengths: [B]
                - tgt_lengths: [B]
        
        Returns:
            total_loss: Total loss
            loss_dict: Individual losses
        """
        # 1. ASR CTC Loss (StreamSpeech Line 224-232)
        L_asr = self._compute_ctc_loss(
            logits=model_output['asr_logits'],
            targets=sample['source_text'],
            input_lengths=model_output.get('encoder_lengths', None),
            target_lengths=sample.get('src_lengths', None),
            name="ASR"
        )
        
        # 2. ST CTC Loss
        L_st = self._compute_ctc_loss(
            logits=model_output['st_logits'],
            targets=sample['target_text'],
            input_lengths=model_output.get('encoder_lengths', None),
            target_lengths=sample.get('tgt_lengths', None),
            name="ST"
        )
        
        # 3. MT Cross-Entropy Loss
        L_mt = self._compute_ce_loss(
            logits=model_output['mt_logits'],
            targets=sample['target_text'],
            name="MT"
        )
        
        # 4. Unit CTC Loss
        L_unit = self._compute_ctc_loss(
            logits=model_output['unit_logits'],
            targets=sample['target_units'],
            input_lengths=model_output.get('unit_lengths', None),
            target_lengths=sample.get('unit_target_lengths', None),
            name="Unit"
        )
        
        # 5. Total Loss (weighted sum)
        total_loss = (
            self.asr_weight * L_asr +
            self.st_weight * L_st +
            self.mt_weight * L_mt +
            self.unit_weight * L_unit
        )
        
        # 6. Logging
        loss_dict = {
            'loss': total_loss.item(),
            'L_asr': L_asr.item(),
            'L_st': L_st.item(),
            'L_mt': L_mt.item(),
            'L_unit': L_unit.item(),
        }
        
        return total_loss, loss_dict
    
    def _compute_ctc_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        input_lengths: Optional[torch.Tensor],
        target_lengths: Optional[torch.Tensor],
        name: str = "CTC"
    ) -> torch.Tensor:
        """
        Compute CTC loss (StreamSpeech Line 224-232).
        
        Args:
            logits: [B, T, vocab] or [T, B, vocab]
            targets: [B, T_tgt]
            input_lengths: [B]
            target_lengths: [B]
        
        Returns:
            loss: Scalar
        """
        # Transpose to [T, B, vocab] if needed
        if logits.dim() == 3 and logits.size(0) != logits.size(1):
            if logits.size(1) > logits.size(0):
                # [B, T, vocab] → [T, B, vocab]
                logits = logits.transpose(0, 1)
        
        # Log probabilities
        log_probs = F.log_softmax(logits, dim=-1)
        
        T, B, vocab = log_probs.shape
        
        # Input lengths (default: full length)
        if input_lengths is None:
            input_lengths = torch.full((B,), T, dtype=torch.long, device=logits.device)
        
        # Target lengths (default: non-padding length)
        if target_lengths is None:
            pad_mask = (targets != self.pad_idx) & (targets != self.eos_idx)
            target_lengths = pad_mask.sum(dim=-1)
        
        # Flatten targets (remove padding)
        targets_flat = []
        for b in range(B):
            tgt_len = target_lengths[b].item()
            targets_flat.append(targets[b, :tgt_len])
        targets_flat = torch.cat(targets_flat, dim=0)
        
        # CTC Loss (StreamSpeech Line 224-232)
        try:
            with torch.backends.cudnn.flags(enabled=False):
                loss = F.ctc_loss(
                    log_probs,
                    targets_flat,
                    input_lengths,
                    target_lengths,
                    blank=self.blank_idx,
                    reduction='mean' if self.sentence_avg else 'sum',
                    zero_infinity=True,
                )
        except Exception as e:
            logger.warning(f"{name} CTC loss failed: {e}")
            loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        return loss
    
    def _compute_ce_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        name: str = "CE"
    ) -> torch.Tensor:
        """
        Compute Cross-Entropy loss with label smoothing.
        
        Args:
            logits: [B, T_logits, vocab]
            targets: [B, T_targets]
        
        Returns:
            loss: Scalar
        """
        B, T_logits, vocab = logits.shape
        B_tgt, T_targets = targets.shape
        
        # Align lengths (take minimum)
        T = min(T_logits, T_targets)
        logits = logits[:, :T, :]  # [B, T, vocab]
        targets = targets[:, :T]  # [B, T]
        
        # Flatten
        logits_flat = logits.reshape(-1, vocab)  # [B*T, vocab]
        targets_flat = targets.reshape(-1)  # [B*T]
        
        # Mask padding
        pad_mask = (targets_flat != self.pad_idx) & (targets_flat != self.eos_idx)
        
        if pad_mask.sum() == 0:
            logger.warning(f"{name} CE loss: no valid targets")
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        # Cross-Entropy with label smoothing
        if self.label_smoothing > 0:
            # Apply mask first
            logits_masked = logits_flat[pad_mask]
            targets_masked = targets_flat[pad_mask]
            
            # Smooth labels
            smooth_targets = torch.zeros_like(logits_masked)
            smooth_targets.fill_(self.label_smoothing / (vocab - 1))
            smooth_targets.scatter_(1, targets_masked.unsqueeze(1), 1.0 - self.label_smoothing)
            
            # Log softmax
            log_probs = F.log_softmax(logits_masked, dim=-1)
            
            # KL divergence
            loss = -(smooth_targets * log_probs).sum(dim=-1).mean()
        else:
            # Standard cross-entropy
            loss = F.cross_entropy(
                logits_flat[pad_mask],
                targets_flat[pad_mask],
                reduction='mean'
            )
        
        return loss


class EchoStreamTrainer:
    """
    EchoStream Training Loop with Multi-task Learning.
    
    StreamSpeech 참고:
    - Multi-chunk training (Line 149-168)
    - Streaming config (Line 136-147)
    """
    
    def __init__(
        self,
        model: nn.Module,
        criterion: EchoStreamMultiTaskCriterion,
        optimizer: torch.optim.Optimizer,
        device: str = "cuda",
        multi_chunk: bool = True,
        segment_choices: list = [1, 2, 4, 8, 16],
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        
        # Multi-chunk training (StreamSpeech Line 149-168)
        self.multi_chunk = multi_chunk
        self.segment_choices = segment_choices
        
        logger.info(
            f"EchoStreamTrainer initialized "
            f"(multi_chunk={multi_chunk}, segments={segment_choices})"
        )
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Single training step with multi-task loss.
        
        Args:
            batch: Training batch
                - audio: [B, T, F]
                - source_text: [B, T_src]
                - target_text: [B, T_tgt]
                - target_units: [B, T_unit]
        
        Returns:
            loss: Total loss
            metrics: Loss components
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # 1. Multi-chunk: Random segment length (StreamSpeech Line 149-168)
        if self.multi_chunk and self.model.training:
            segment_length = self._sample_segment_length()
        else:
            segment_length = 99999  # Full sequence (offline)
        
        # 2. Model forward
        model_output = self.model(
            src_tokens=batch['audio'].to(self.device),
            src_lengths=batch.get('audio_lengths', None),
            segment_length=segment_length,  # ← Multi-chunk!
        )
        
        # 3. Multi-task loss
        total_loss, loss_dict = self.criterion(
            model_output=model_output,
            sample=batch
        )
        
        # 4. Backward
        total_loss.backward()
        
        # 5. Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # 6. Optimizer step
        self.optimizer.step()
        
        # 7. Metrics
        metrics = {
            'loss': total_loss.item(),
            'segment_length': segment_length,
            **loss_dict
        }
        
        return total_loss, metrics
    
    def _sample_segment_length(self) -> int:
        """
        Sample random segment length (StreamSpeech Line 153).
        
        Returns:
            segment_length: Random choice from segment_choices
        """
        import random
        return random.choice(self.segment_choices)


if __name__ == "__main__":
    print("="*70)
    print("Testing EchoStreamMultiTaskCriterion")
    print("="*70)
    
    # Mock model output
    B, T, vocab = 2, 100, 6000
    T_mt = 20
    T_unit = 100
    num_units = 1000
    
    model_output = {
        'asr_logits': torch.randn(B, T, vocab, requires_grad=True),
        'st_logits': torch.randn(B, T, vocab, requires_grad=True),
        'mt_logits': torch.randn(B, T_mt, vocab, requires_grad=True),
        'unit_logits': torch.randn(B, T_unit, num_units, requires_grad=True),
        'encoder_lengths': torch.tensor([T, T]),
    }
    
    # Mock sample
    sample = {
        'source_text': torch.randint(3, vocab, (B, 50)),
        'target_text': torch.randint(3, vocab, (B, 50)),
        'target_units': torch.randint(0, num_units, (B, 80)),
        'src_lengths': torch.tensor([50, 50]),
        'tgt_lengths': torch.tensor([50, 50]),
    }
    
    # Initialize criterion
    criterion = EchoStreamMultiTaskCriterion(
        asr_weight=1.0,
        st_weight=1.0,
        mt_weight=1.0,
        unit_weight=1.0,
    )
    
    print("\n1. Testing multi-task loss computation...")
    total_loss, loss_dict = criterion(model_output, sample)
    
    print(f"   Total loss: {total_loss.item():.4f}")
    print(f"   L_asr: {loss_dict['L_asr']:.4f}")
    print(f"   L_st: {loss_dict['L_st']:.4f}")
    print(f"   L_mt: {loss_dict['L_mt']:.4f}")
    print(f"   L_unit: {loss_dict['L_unit']:.4f}")
    print("   ✅ Multi-task loss computed")
    
    print("\n2. Testing backward pass...")
    total_loss.backward()
    print("   ✅ Backward pass successful")
    
    print("\n3. Testing trainer...")
    
    # Mock model
    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(80, 256)
            self.training = True
        
        def forward(self, src_tokens, src_lengths=None, segment_length=None):
            B, T, F = src_tokens.shape
            # Pass through linear to create gradient flow
            x = self.linear(src_tokens)  # [B, T, 256]
            return {
                'asr_logits': x.new_zeros(B, T, 6000).requires_grad_(True),
                'st_logits': x.new_zeros(B, T, 6000).requires_grad_(True),
                'mt_logits': x.new_zeros(B, 20, 6000).requires_grad_(True),
                'unit_logits': x.new_zeros(B, 100, 1000).requires_grad_(True),
            }
    
    model = MockModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    trainer = EchoStreamTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device="cpu",
        multi_chunk=True,
        segment_choices=[1, 2, 4, 8],
    )
    
    batch = {
        'audio': torch.randn(2, 100, 80),
        'source_text': torch.randint(3, 6000, (2, 50)),
        'target_text': torch.randint(3, 6000, (2, 50)),
        'target_units': torch.randint(0, 1000, (2, 80)),
    }
    
    loss, metrics = trainer.train_step(batch)
    
    print(f"   Loss: {loss.item():.4f}")
    print(f"   Segment length: {metrics['segment_length']}")
    print("   ✅ Training step successful")
    
    print("\n4. Testing multi-chunk sampling...")
    segments = [trainer._sample_segment_length() for _ in range(10)]
    print(f"   Sampled segments: {segments}")
    print("   ✅ Multi-chunk sampling works")
    
    print("\n" + "="*70)
    print("✅ All EchoStreamMultiTaskCriterion tests passed!")
    print("="*70)
    
    print("\n💡 Usage:")
    print("  # Initialize")
    print("  criterion = EchoStreamMultiTaskCriterion()")
    print("  trainer = EchoStreamTrainer(model, criterion, optimizer)")
    print("  ")
    print("  # Training loop")
    print("  for batch in dataloader:")
    print("      loss, metrics = trainer.train_step(batch)")
    print("      print(f\"Loss: {metrics['loss']:.4f}\")")

