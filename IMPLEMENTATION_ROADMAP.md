# EchoStream 구현 로드맵

**목표**: StreamSpeech 분석 기반 EchoStream 완성 구현

**기간**: 6 Phase (체계적 단계별 구현)

**핵심 개선사항**:
1. Conformer → Emformer (O(T²) → O(1))
2. Wait-k → Word-Level Streaming (1200ms → 100ms)
3. 단일 청크 → Multi-chunk Training (유연한 레이턴시)
4. 단순 학습 → Multi-task Learning (품질 향상)

---

## 📁 StreamSpeech 구조 분석

### 핵심 디렉토리

```
StreamSpeech_analysis/
├── researches/ctc_unity/          # StreamSpeech 메인 구현
│   ├── models/
│   │   ├── streamspeech_model.py  # 메인 모델
│   │   ├── s2s_conformer.py       # Conformer 기반
│   │   └── s2t_conformer.py       # Encoder
│   ├── modules/
│   │   ├── conformer_layer.py     # Conformer 레이어
│   │   ├── ctc_decoder_with_transformer_layer.py  # ST CTC
│   │   ├── ctc_transformer_unit_decoder.py        # Unit Decoder
│   │   ├── transformer_decoder.py                 # MT Decoder
│   │   └── transformer_encoder.py                 # T2U Encoder
│   ├── criterions/
│   │   └── speech_to_speech_ctc_asr_st_criterion.py  # Multi-task Loss
│   └── tasks/
│       └── speech_to_speech_ctc.py                # Task 정의
├── agent/
│   ├── speech_to_speech.streamspeech.agent.py    # S2ST Agent
│   ├── ctc_decoder.py                             # CTC 디코더
│   └── tts/codehifigan.py                        # Vocoder
└── fairseq/                                       # 기반 프레임워크
```

---

## 🎯 Phase 1: StreamSpeech 핵심 구조 분석 [IN PROGRESS]

### 목표
StreamSpeech의 핵심 컴포넌트 이해 및 매핑

### 작업

#### 1.1 모델 구조 분석 ✅

**파일**: `researches/ctc_unity/models/streamspeech_model.py`

**핵심 컴포넌트**:
```python
class StreamSpeechModel(ChunkS2UTConformerModel):
    def __init__(self, encoder, multitask_decoders, args):
        # 1. Speech Encoder (Chunk-based Conformer)
        self.encoder
        
        # 2. Multi-task Decoders
        self.multitask_decoders = {
            'source_unigram': ASR CTC Decoder,
            'ctc_target_unigram': ST CTC Decoder,
            'target_unigram': MT Decoder,
        }
        
        # 3. T2U Encoder (Optional)
        self.synthesizer_encoder
        
        # 4. Unit Decoder
        self.decoder  # CTCTransformerUnitDecoder
```

**EchoStream 매핑**:
```python
class EchoStreamModel(nn.Module):
    def __init__(self, ...):
        # 1. Emformer Encoder (대체!)
        self.encoder = EchoStreamSpeechEncoder(...)
        
        # 2. Multi-task Decoders (동일)
        self.asr_ctc_decoder = CTCDecoder(...)
        self.st_ctc_decoder = CTCDecoderWithTransformerLayer(...)
        self.mt_decoder = TransformerMTDecoder(...)
        
        # 3. T2U Encoder (동일, 0 layers)
        # 생략 (직접 연결)
        
        # 4. Unit Decoder (동일)
        self.unit_decoder = CTCTransformerUnitDecoder(...)
        
        # 5. Vocoder (동일)
        self.vocoder = CodeHiFiGANVocoder(...)
```

---

#### 1.2 디코더 구조 분석

**ASR CTC Decoder**:
```python
# modules/ctc_decoder.py (fairseq)
class CTCDecoder(nn.Module):
    def __init__(self, embed_dim, vocab_size):
        self.output_projection = nn.Linear(embed_dim, vocab_size)
    
    def forward(self, encoder_out):
        logits = self.output_projection(encoder_out)
        return logits
```

**ST CTC Decoder**:
```python
# modules/ctc_decoder_with_transformer_layer.py
class CTCDecoderWithTransformerLayer(nn.Module):
    def __init__(self, ..., num_layers=2, unidirectional=True):
        # Transformer layers (2L)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(...) for _ in range(num_layers)
        ])
        
        # CTC projection
        self.output_projection = nn.Linear(embed_dim, vocab_size)
    
    def forward(self, encoder_out, incremental_state=None):
        x = encoder_out
        
        # Transformer layers with causal mask
        for layer in self.layers:
            x = layer(x, self_attn_mask=future_mask if unidirectional else None)
        
        # CTC projection
        logits = self.output_projection(x)
        return logits
```

**MT Decoder**:
```python
# modules/transformer_decoder.py
class TransformerDecoder(TransformerDecoderBase):
    def __init__(self, ..., num_layers=4):
        # Standard Transformer Decoder
        # - Self-attention (causal)
        # - Cross-attention (encoder_out)
        # - Feed-forward
        pass
```

**Unit Decoder**:
```python
# modules/ctc_transformer_unit_decoder.py
class CTCTransformerUnitDecoder(TransformerUnitDecoder):
    def __init__(self, ..., num_layers=6, ctc_upsample_ratio=5):
        # CTC upsampling
        self.ctc_upsample_ratio = 5
        
        # Transformer layers (6L)
        self.layers = ...
        
        # Multi-frame prediction
        self.output_projection = nn.Linear(embed_dim, num_units * n_frames_per_step)
```

---

#### 1.3 정책 분석

**Agent 파일**: `agent/speech_to_speech.streamspeech.agent.py`

**핵심 정책**:
```python
def policy(self):
    # 1. Feature extraction
    feature = self.feature_extractor(self.states.source)
    
    # 2. Encoder forward
    encoder_outs = self.model.forward_encoder(...)
    
    # 3. ASR CTC
    finalized_asr = self.asr_ctc_generator.generate(encoder_outs)
    src_ctc_length = asr_tokens.size(-1)
    
    # 4. ST CTC
    finalized_st = self.st_ctc_generator.generate(encoder_outs)
    tgt_ctc_length = st_tokens.size(-1)
    
    # 5. READ/WRITE 정책
    if (
        src_ctc_length < self.src_ctc_prefix_length + self.stride_n
        or tgt_ctc_length < self.tgt_ctc_prefix_length + self.stride_n
    ):
        return ReadAction()  # READ
    
    # 6. MT Decoder
    new_subword_tokens = (
        (tgt_ctc_length - self.lagging_k1) // self.stride_n
    ) * self.stride_n
    
    finalized_mt = self.generator_mt.generate_decoder(
        ...,
        max_new_tokens=new_subword_tokens  # Alignment-guided!
    )
    
    # 7. Unit Decoder
    finalized = self.ctc_generator.generate(mt_output)
    
    # 8. Vocoder
    wav = self.vocoder(units)
    
    # 9. WRITE
    return WriteAction(SpeechSegment(content=wav))
```

---

## 🚀 Phase 2: Word-Level Streaming 모듈 구현 [PENDING]

### 목표
StreamSpeech 정책을 Word-Level로 개선

### 작업

#### 2.1 WordBoundaryDetector 구현

**파일**: `models/word_boundary_detector.py`

```python
class WordBoundaryDetector:
    """
    ASR CTC + SentencePiece를 사용한 단어 경계 탐지.
    
    StreamSpeech 개선:
    - StreamSpeech: stride_n 기반 (고정 토큰 수)
    - EchoStream: 단어 경계 기반 (동적)
    """
    
    def __init__(self, emformer_encoder, asr_ctc_decoder, tokenizer):
        self.encoder = emformer_encoder
        self.asr_ctc = asr_ctc_decoder
        self.tokenizer = tokenizer
        
        # Cache (Emformer)
        self.encoder_cache = {}
        self.partial_word = ""
        
    def process_segment(self, audio_segment):
        """
        세그먼트 처리 및 단어 경계 탐지.
        
        Returns:
            None: 단어 미완성
            Dict: 완성된 단어
        """
        # 1. Emformer encoding (with cache)
        encoder_out, self.encoder_cache = self.encoder(
            audio_segment,
            cache=self.encoder_cache
        )
        
        # 2. ASR CTC decoding
        asr_logits = self.asr_ctc(encoder_out)
        asr_tokens = asr_logits.argmax(dim=-1)
        
        # 3. CTC collapse
        collapsed_tokens = self._ctc_collapse(asr_tokens)
        
        # 4. Decode
        new_text = self.tokenizer.decode(collapsed_tokens)
        
        # 5. Word boundary check
        if self._is_word_boundary(new_text):
            word = self.partial_word + new_text.rstrip("▁ ")
            self.partial_word = ""
            
            return {
                'word': word,
                'encoder_out': encoder_out,
                'is_complete': True
            }
        else:
            self.partial_word += new_text
            return None
    
    def _is_word_boundary(self, text):
        """SentencePiece ▁ 토큰 체크"""
        return (
            text.endswith("▁") or
            text.endswith(" ") or
            text.endswith((".", ",", "!", "?"))
        )
```

---

#### 2.2 WordLevelTranslator 구현

**파일**: `models/word_level_translator.py`

```python
class WordLevelTranslator:
    """
    단어 단위 번역.
    
    StreamSpeech와 차이:
    - StreamSpeech: 청크 단위 batch 생성
    - EchoStream: 단어 단위 incremental 생성
    """
    
    def __init__(self, st_ctc, mt_decoder, unit_decoder, vocoder):
        self.st_ctc = st_ctc
        self.mt_decoder = mt_decoder
        self.unit_decoder = unit_decoder
        self.vocoder = vocoder
        
        # Incremental state (StreamSpeech와 동일)
        self.mt_incremental_state = {}
        self.prev_mt_tokens = None
    
    def translate_word(self, encoder_out, source_word):
        """
        단어 번역 (StreamSpeech 정책 활용).
        
        StreamSpeech의 MT Decoder 로직 차용:
        - max_new_tokens 계산
        - Incremental state 관리
        """
        # 1. ST CTC
        st_logits = self.st_ctc(encoder_out)
        st_tokens = st_logits.argmax(dim=-1)
        st_tokens = self._ctc_collapse(st_tokens)
        
        # 2. MT Decoder (incremental)
        # StreamSpeech의 alignment-guided token calculation 차용
        max_new_tokens = len(st_tokens) + 2
        
        mt_output = self.mt_decoder(
            prev_output_tokens=self.prev_mt_tokens,
            encoder_out=encoder_out,
            incremental_state=self.mt_incremental_state,
            max_new_tokens=max_new_tokens
        )
        
        # 3. Extract new tokens
        if self.prev_mt_tokens is not None:
            new_mt_tokens = mt_output['tokens'][len(self.prev_mt_tokens):]
        else:
            new_mt_tokens = mt_output['tokens']
        
        self.prev_mt_tokens = mt_output['tokens']
        
        # 4. Unit Decoder
        unit_output = self.unit_decoder(mt_output['decoder_out'])
        units = unit_output['units']
        
        # 5. Vocoder
        waveform = self.vocoder(units.unsqueeze(0))
        
        return {
            'translation': self.tokenizer.decode(new_mt_tokens),
            'units': units,
            'waveform': waveform
        }
```

---

## 📚 Phase 3: Multi-task 학습 구현 [PENDING]

### 목표
StreamSpeech의 Multi-task Learning 차용

### 작업

#### 3.1 Multi-task Criterion 구현

**참고**: `criterions/speech_to_speech_ctc_asr_st_criterion.py`

```python
class EchoStreamMultiTaskCriterion(nn.Module):
    """
    StreamSpeech의 Multi-task Learning 차용.
    
    L = L_asr + L_st + L_mt + L_unit
    """
    
    def forward(self, model, sample):
        # 1. Forward pass
        output = model(
            src_tokens=sample['net_input']['src_tokens'],
            src_lengths=sample['net_input']['src_lengths'],
            prev_output_tokens=sample['prev_output_tokens']
        )
        
        # 2. ASR Loss (CTC)
        L_asr = self.compute_ctc_loss(
            output['asr_logits'],
            sample['source_text']
        )
        
        # 3. ST Loss (CTC)
        L_st = self.compute_ctc_loss(
            output['st_logits'],
            sample['target_text']
        )
        
        # 4. MT Loss (Cross-Entropy)
        L_mt = self.compute_ce_loss(
            output['mt_logits'],
            sample['target_text']
        )
        
        # 5. Unit Loss (CTC)
        L_unit = self.compute_ctc_loss(
            output['unit_logits'],
            sample['target_units']
        )
        
        # 6. Total Loss
        L_total = L_asr + L_st + L_mt + L_unit
        
        return L_total, {
            'loss': L_total,
            'L_asr': L_asr,
            'L_st': L_st,
            'L_mt': L_mt,
            'L_unit': L_unit
        }
```

---

## 🎲 Phase 4: Alignment-based 정책 구현 [PENDING]

### 목표
StreamSpeech의 정렬 기반 READ/WRITE 정책 통합

### 작업

#### 4.1 Policy Module 구현

**참고**: `agent/speech_to_speech.streamspeech.agent.py:480-509`

```python
class AlignmentBasedPolicy:
    """
    StreamSpeech의 alignment-based policy 차용.
    
    조건:
    - |Â| > |A|: 새 ASR 토큰 인식
    - |Ŷ| > |Y|: 새 ST 토큰 예측
    """
    
    def __init__(self, stride_n=1):
        self.stride_n = stride_n
        
        # Previous lengths
        self.src_ctc_prefix_length = 0
        self.tgt_ctc_prefix_length = 0
    
    def should_write(self, asr_tokens, st_tokens):
        """
        WRITE 여부 결정 (StreamSpeech 로직).
        
        Returns:
            (should_write, max_new_tokens)
        """
        src_ctc_length = asr_tokens.size(-1)
        tgt_ctc_length = st_tokens.size(-1)
        
        # StreamSpeech 정책 (Line 485-489)
        if (
            src_ctc_length < self.src_ctc_prefix_length + self.stride_n
            or tgt_ctc_length < self.tgt_ctc_prefix_length + self.stride_n
        ):
            return False, 0  # READ
        
        # Update lengths
        self.src_ctc_prefix_length = max(src_ctc_length, self.src_ctc_prefix_length)
        self.tgt_ctc_prefix_length = max(tgt_ctc_length, self.tgt_ctc_prefix_length)
        
        # Calculate max_new_tokens (Line 496-498)
        max_new_tokens = (tgt_ctc_length // self.stride_n) * self.stride_n
        
        return True, max_new_tokens  # WRITE
```

---

## 🔀 Phase 5: Multi-chunk 학습 구현 [PENDING]

### 목표
StreamSpeech의 Multi-chunk Training 차용

### 작업

#### 5.1 Multi-chunk Sampler 구현

```python
class MultiChunkSampler:
    """
    StreamSpeech의 Multi-chunk Training.
    
    C ~ U(1, |X|)
    """
    
    def __init__(self, min_segment=1, max_segment=None):
        self.min_segment = min_segment
        self.max_segment = max_segment
    
    def sample_segment_length(self, audio_length):
        """
        랜덤 세그먼트 크기 샘플링.
        
        Returns:
            segment_length: 1 ~ audio_length
        """
        max_len = self.max_segment or audio_length
        segment_length = random.randint(self.min_segment, max_len)
        
        return segment_length
```

#### 5.2 Training Loop 수정

```python
class EchoStreamTrainer:
    def train_step(self, batch):
        # Multi-chunk: 랜덤 세그먼트 크기
        segment_length = self.sampler.sample_segment_length(
            batch['audio_length']
        )
        
        # Model forward with random segment length
        output = self.model(
            batch['audio'],
            segment_length=segment_length  # ← 동적!
        )
        
        # Multi-task loss
        loss, loss_dict = self.criterion(output, batch)
        
        return loss
```

---

## 🎭 Phase 6: 통합 Agent 구현 및 테스트 [PENDING]

### 목표
모든 모듈 통합 및 StreamSpeech와 비교

### 작업

#### 6.1 EchoStream Word-Level Agent

```python
@entrypoint
class EchoStreamWordLevelAgent(SpeechToSpeechAgent):
    """
    StreamSpeech 정책 + Emformer + Word-Level.
    
    개선사항:
    1. Conformer → Emformer (O(1))
    2. Chunk 320ms → Segment 40ms
    3. Wait-k → Word boundary
    4. 정렬 기반 정책 유지
    """
    
    def __init__(self, args):
        super().__init__(args)
        
        # Components
        self.word_detector = WordBoundaryDetector(...)
        self.translator = WordLevelTranslator(...)
        self.recomposer = SentenceRecomposer(...)
        self.policy = AlignmentBasedPolicy(...)
    
    def policy(self):
        # 1. 세그먼트 읽기 (40ms)
        segment = self.states.source
        
        # 2. 단어 경계 탐지
        word_data = self.word_detector.process_segment(segment)
        
        if not word_data:
            return ReadAction()  # 단어 미완성
        
        # 3. StreamSpeech 정책 체크
        should_write, max_new_tokens = self.policy.should_write(
            asr_tokens=word_data['asr_tokens'],
            st_tokens=word_data['st_tokens']
        )
        
        if not should_write:
            return ReadAction()  # 정책: READ
        
        # 4. 단어 번역
        word_result = self.translator.translate_word(
            encoder_out=word_data['encoder_out'],
            source_word=word_data['word'],
            max_new_tokens=max_new_tokens  # StreamSpeech alignment!
        )
        
        # 5. 문장 재조합 체크
        sentence_result = self.recomposer.add_word(word_result)
        
        # 6. WRITE
        return WriteAction(
            SpeechSegment(
                content=sentence_result['waveform'],
                sample_rate=16000
            )
        )
```

---

## 📊 최종 비교 목표

| 메트릭 | StreamSpeech | EchoStream (목표) |
|--------|-------------|-------------------|
| Encoder | Conformer O(T²) | Emformer O(1) |
| 첫 응답 | 800ms | 100ms |
| 정책 | Wait-k (고정) | Word-Level (동적) |
| 메모리 (60s) | 150MB | 6MB |
| RTF (60s) | 3.0 ❌ | 0.4 ✅ |
| 학습 | Multi-task ✅ | Multi-task ✅ |
| 청크 크기 | 고정 320ms | Multi-chunk ✅ |

---

## 📅 일정

- **Week 1**: Phase 1-2 (구조 분석 + Word-Level)
- **Week 2**: Phase 3-4 (Multi-task + Policy)
- **Week 3**: Phase 5-6 (Multi-chunk + Integration)
- **Week 4**: Testing & Benchmarking

---

**현재 진행**: Phase 1 (구조 분석 중) ✅

**다음 단계**: StreamSpeech 모듈 상세 분석 및 EchoStream 매핑

