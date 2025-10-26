##########################################
# Simultaneous Speech-to-Speech Translation Agent with CT-Transformer Punctuation
#
# StreamSpeech + CT-Transformer Integration for Sentence Boundary Detection
# and Re-composition Triggering
##########################################

from agent.speech_to_speech.streamspeech.agent import (
    StreamSpeechS2STAgent,
    OnlineFeatureExtractor,
    SHIFT_SIZE,
    WINDOW_SIZE,
    ORG_SAMPLE_RATE,
    SAMPLE_RATE,
    FEATURE_DIM,
    BOW_PREFIX,
    DEFAULT_EOS,
)
from simuleval.utils import entrypoint
from simuleval.data.segments import SpeechSegment
from simuleval.agents.actions import WriteAction, ReadAction
from pathlib import Path
import logging
import torch
import numpy as np
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


@entrypoint
class StreamSpeechWithPunctuationAgent(StreamSpeechS2STAgent):
    """
    StreamSpeech agent enhanced with CT-Transformer punctuation predictor
    for sentence boundary detection and re-composition triggering.
    """

    def __init__(self, args):
        # 기존 StreamSpeech 초기화
        super().__init__(args)
        
        # CT-Transformer Punctuation 초기화
        self.init_punctuation_predictor(args)
        
        # 재조합 버퍼
        self.unit_buffer = []  # 유닛 시퀀스 버퍼
        self.text_buffer = []  # 텍스트 버퍼
        self.wav_buffer = []   # 음성 버퍼
        
        # 재조합 설정
        self.enable_recomposition = getattr(args, "enable_recomposition", True)
        self.recomposition_delay = getattr(args, "recomposition_delay", 0.2)
        
        logger.info(
            f"StreamSpeech with Punctuation initialized "
            f"(recomposition={'enabled' if self.enable_recomposition else 'disabled'})"
        )
    
    def init_punctuation_predictor(self, args):
        """Initialize CT-Transformer punctuation predictor."""
        try:
            from agent.ct_transformer_punctuator import (
                CTTransformerPunctuator,
                SentenceBoundaryDetector
            )
            
            # CT-Transformer 모델 경로
            punc_model_path = getattr(
                args, 
                "punctuation_model_path",
                "models/ct_transformer/punc.bin"
            )
            
            # Punctuator 초기화
            self.punctuator = CTTransformerPunctuator(
                model_path=punc_model_path,
                mode="online"
            )
            
            # Sentence Boundary Detector 초기화
            self.sentence_detector = SentenceBoundaryDetector(
                punctuator=self.punctuator,
                buffer_size=getattr(args, "punc_buffer_size", 50),
                min_trigger_length=getattr(args, "punc_min_length", 5)
            )
            
            self.use_punctuation = True
            logger.info("CT-Transformer punctuation predictor initialized successfully.")
            
        except Exception as e:
            logger.warning(
                f"Failed to initialize CT-Transformer: {e}. "
                f"Running without punctuation-based re-composition."
            )
            self.use_punctuation = False
            self.sentence_detector = None
    
    @staticmethod
    def add_args(parser):
        # 기존 StreamSpeech 인자
        StreamSpeechS2STAgent.add_args(parser)
        
        # CT-Transformer 관련 인자
        parser.add_argument(
            "--punctuation-model-path",
            type=str,
            default="models/ct_transformer/punc.bin",
            help="Path to CT-Transformer ONNX model"
        )
        parser.add_argument(
            "--enable-recomposition",
            action="store_true",
            default=True,
            help="Enable re-composition based on sentence boundaries"
        )
        parser.add_argument(
            "--punc-buffer-size",
            type=int,
            default=50,
            help="Maximum buffer size for punctuation prediction"
        )
        parser.add_argument(
            "--punc-min-length",
            type=int,
            default=5,
            help="Minimum text length to trigger punctuation prediction"
        )
        parser.add_argument(
            "--recomposition-delay",
            type=float,
            default=0.2,
            help="Delay (seconds) before re-composition after sentence boundary"
        )
    
    def reset(self):
        """Reset agent state."""
        super().reset()
        
        # 재조합 버퍼 초기화
        self.unit_buffer = []
        self.text_buffer = []
        self.wav_buffer = []
        
        # Sentence detector 초기화
        if self.use_punctuation and self.sentence_detector:
            self.sentence_detector.reset()
    
    @torch.inference_mode()
    def policy(self):
        """
        Main policy with CT-Transformer sentence boundary detection.
        """
        # 기존 StreamSpeech policy 실행
        feature = self.feature_extractor(self.states.source)
        
        if feature.size(0) == 0 and not self.states.source_finished:
            return ReadAction()
        
        src_indices = feature.unsqueeze(0)
        src_lengths = torch.tensor([feature.size(0)], device=self.device).long()
        
        # Encoder forward
        self.encoder_outs = self.generator.model.forward_encoder(
            {"src_tokens": src_indices, "src_lengths": src_lengths}
        )
        
        # ========================================
        # ASR CTC 디코더 (CT-Transformer 입력용)
        # ========================================
        finalized_asr = self.asr_ctc_generator.generate(
            self.encoder_outs[0], aux_task_name="source_unigram"
        )
        
        # ASR 텍스트 추출
        asr_text = self._extract_text_from_hypo(
            finalized_asr, 
            self.dict["source_unigram"]
        )
        
        # CT-Transformer로 문장 경계 탐지
        sentence_boundary_detected = False
        complete_sentence = ""
        
        if self.use_punctuation and self.sentence_detector and asr_text:
            trigger, complete_sentence, remaining = self.sentence_detector.add_text(
                asr_text
            )
            
            if trigger:
                sentence_boundary_detected = True
                logger.info(f"[CT-Transformer] Sentence boundary detected: {complete_sentence}")
        
        # ========================================
        # ST CTC 디코더
        # ========================================
        finalized_st = self.st_ctc_generator.generate(
            self.encoder_outs[0], aux_task_name="ctc_target_unigram"
        )
        st_probs = torch.exp(finalized_st[0][0]["lprobs"])
        
        # 기존 wait-k 정책 체크
        if not self.states.source_finished:
            src_ctc_indices = finalized_asr[0][0]["tokens"].int()
            tgt_ctc_indices = finalized_st[0][0]["tokens"].int()
            
            src_ctc_prefix_length = src_ctc_indices.size(-1)
            tgt_ctc_prefix_length = tgt_ctc_indices.size(-1)
            
            self.src_ctc_indices = src_ctc_indices
            
            if (
                src_ctc_prefix_length < self.src_ctc_prefix_length + self.stride_n
                or tgt_ctc_prefix_length < self.tgt_ctc_prefix_length + self.stride_n
            ):
                # 문장 경계가 감지되지 않았으면 계속 READ
                if not sentence_boundary_detected:
                    return ReadAction()
            
            self.src_ctc_prefix_length = max(
                src_ctc_prefix_length, self.src_ctc_prefix_length
            )
            self.tgt_ctc_prefix_length = max(
                tgt_ctc_prefix_length, self.tgt_ctc_prefix_length
            )
        
        # ========================================
        # MT Decoder + Unit Decoder
        # ========================================
        single_model = self.generator.model.single_model
        mt_decoder = getattr(single_model, f"{single_model.mt_task_name}_decoder")
        
        # MT decoder 처리 (기존 코드와 동일)
        new_subword_tokens = self._compute_new_subword_tokens()
        
        finalized_mt = self.generator_mt.generate_decoder(
            self.encoder_outs,
            src_indices,
            src_lengths,
            {
                "id": 1,
                "net_input": {"src_tokens": src_indices, "src_lengths": src_lengths},
            },
            self.tgt_subwords_indices,
            None,
            None,
            aux_task_name=single_model.mt_task_name,
            max_new_tokens=new_subword_tokens,
        )
        
        # Unit decoder 처리
        units, wav = self._generate_units_and_wav(finalized_mt, mt_decoder, single_model)
        
        # ========================================
        # 재조합 트리거 처리
        # ========================================
        if sentence_boundary_detected and self.enable_recomposition:
            # 버퍼에 누적된 유닛/텍스트/음성을 재조합
            return self._trigger_recomposition(
                complete_sentence,
                units,
                wav
            )
        
        # 기존 방식대로 출력
        return self._generate_output(units, wav)
    
    def _extract_text_from_hypo(self, hypo_list, dictionary):
        """Extract text from CTC hypothesis."""
        if not hypo_list or not hypo_list[0]:
            return ""
        
        tokens = hypo_list[0][0]["tokens"].int()
        text = "".join([dictionary[c] for c in tokens])
        text = text.replace("_", " ").replace("▁", " ").replace("<unk>", " ")
        text = text.replace("<s>", "").replace("</s>", "")
        return text.strip()
    
    def _compute_new_subword_tokens(self):
        """Compute number of new subword tokens to generate."""
        if not self.states.source_finished:
            subword_tokens = (
                (self.tgt_ctc_prefix_length - self.lagging_k1) // self.stride_n
            ) * self.stride_n
            
            if self.whole_word:
                subword_tokens += 1
            
            new_subword_tokens = (
                (subword_tokens - self.tgt_subwords_indices.size(-1))
                if self.tgt_subwords_indices is not None
                else subword_tokens
            )
            
            return max(0, int(new_subword_tokens))
        else:
            return -1
    
    def _generate_units_and_wav(self, finalized_mt, mt_decoder, single_model):
        """Generate speech units and waveform from MT decoder output."""
        # MT decoder forward (기존 코드 참조)
        # ... (StreamSpeech 기존 로직 사용)
        
        # Simplified version - actual implementation uses full logic from original agent
        return [], torch.tensor([])
    
    def _trigger_recomposition(
        self,
        complete_sentence: str,
        units: List[int],
        wav: torch.Tensor
    ):
        """
        Trigger re-composition when sentence boundary is detected.
        
        Args:
            complete_sentence: Complete sentence with punctuation
            units: Generated speech units
            wav: Generated waveform
        
        Returns:
            WriteAction with re-composed output
        """
        logger.info(f"[Re-composition] Triggered for: '{complete_sentence}'")
        
        # 재조합 모듈 호출 (TODO: 실제 재조합 로직 구현)
        # 현재는 버퍼의 내용을 그대로 출력
        recomposed_wav = wav
        
        # 버퍼 초기화
        self.unit_buffer = []
        self.text_buffer = []
        self.wav_buffer = []
        
        # 통계 업데이트
        if hasattr(self, 'sentence_detector'):
            stats = self.sentence_detector.get_stats()
            logger.info(f"[CT-Transformer Stats] {stats}")
        
        return WriteAction(
            SpeechSegment(
                content=recomposed_wav.tolist(),
                sample_rate=SAMPLE_RATE,
                finished=self.states.source_finished,
            ),
            finished=self.states.target_finished,
        )
    
    def _generate_output(self, units, wav):
        """Generate normal output (without re-composition)."""
        # 기존 StreamSpeech 출력 로직
        return WriteAction(
            SpeechSegment(
                content=wav.tolist() if len(wav) > 0 else [],
                sample_rate=SAMPLE_RATE,
                finished=self.states.source_finished,
            ),
            finished=self.states.target_finished,
        )

