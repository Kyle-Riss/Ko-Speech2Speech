# StreamSpeech 호환성 가이드

**중요**: EchoStream은 StreamSpeech의 디코더 구조를 그대로 사용하므로, **반드시 StreamSpeech/Fairseq 형식을 따라야 합니다**. 형식을 맞추지 않으면 오류가 발생합니다.

---

## 🔴 필수 구조: Encoder 출력 형식

### 올바른 형식 (StreamSpeech/Fairseq 호환)

```python
encoder_out = {
    'encoder_out': [tensor],              # ⚠️ List 형태! [T, B, D]
    'encoder_padding_mask': [tensor],    # ⚠️ List 형태! [B, T] 또는 []
    'encoder_embedding': [],              # 빈 리스트
    'encoder_states': [],                 # 빈 리스트
    'src_tokens': [],                     # 빈 리스트
    'src_lengths': [],                   # 빈 리스트
}
```

### ❌ 잘못된 형식 (오류 발생!)

```python
# 잘못된 예 1: List가 아닌 직접 텐서
encoder_out = {
    'encoder_out': tensor,  # ❌ List가 아님!
    ...
}

# 잘못된 예 2: 차원 순서가 다름
encoder_out = {
    'encoder_out': [tensor],  # ❌ [B, T, D] (time-first가 아님!)
    ...
}

# 잘못된 예 3: 키 이름이 다름
encoder_out = {
    'output': [tensor],  # ❌ 'encoder_out'이 아님!
    ...
}
```

---

## 📐 텐서 차원 순서

### Encoder 출력

```python
encoder_out['encoder_out'][0]  # [T, B, D]
# T: 시간 프레임 (다운샘플링 후)
# B: 배치 크기
# D: 임베딩 차원 (예: 256)
```

**중요**: 
- ✅ **Time-first**: `[T, B, D]` 형식
- ❌ Batch-first: `[B, T, D]` 형식 (오류!)

### Padding Mask

```python
encoder_out['encoder_padding_mask'][0]  # [B, T]
# B: 배치 크기
# T: 시간 프레임
# 값: True = padding, False = valid
```

---

## 🔧 디코더가 기대하는 형식

### MT Decoder (TransformerMTDecoder)

```python
# models/decoders/transformer_decoder.py
def forward(self, prev_output_tokens, encoder_out):
    # 디코더는 이렇게 접근합니다:
    encoder_hidden = encoder_out['encoder_out'][0]  # [T', B, D]
    if encoder_out['encoder_padding_mask']:
        encoder_padding_mask = encoder_out['encoder_padding_mask'][0]  # [B, T']
```

**오류 예시**:
```python
# ❌ 이렇게 하면 오류!
encoder_hidden = encoder_out['encoder_out']  # List 전체를 전달
# → TypeError: expected Tensor, got list

# ❌ 이렇게 해도 오류!
encoder_hidden = encoder_out['output']  # 키 이름이 다름
# → KeyError: 'output'
```

### CTC Decoder

```python
# models/decoders/ctc_decoder.py
def forward(self, encoder_out, encoder_padding_mask):
    # encoder_out: [T, B, D] 텐서 (List가 아님!)
    # encoder_padding_mask: [B, T] 텐서 또는 None
```

**사용 예시**:
```python
# ✅ 올바른 사용
encoder_out_dict = encoder(src_tokens, src_lengths)
encoder_hidden = encoder_out_dict['encoder_out'][0]  # [T, B, D]
encoder_padding_mask = encoder_out_dict['encoder_padding_mask'][0] if encoder_out_dict['encoder_padding_mask'] else None

ctc_output = ctc_decoder(
    encoder_out=encoder_hidden,  # 텐서 직접 전달
    encoder_padding_mask=encoder_padding_mask,
)
```

---

## ✅ 올바른 구현 예시

### EchoStreamSpeechEncoder

```python
# models/echostream_encoder.py
class EchoStreamSpeechEncoder(nn.Module):
    def forward(self, src_tokens, src_lengths):
        # ... 인코딩 로직 ...
        
        # ✅ 올바른 형식으로 반환
        return {
            'encoder_out': emformer_out['encoder_out'],  # List of [T, B, D]
            'encoder_padding_mask': emformer_out['encoder_padding_mask'],  # List of [B, T]
            'encoder_embedding': [],
            'encoder_states': [],
            'src_tokens': [],
            'src_lengths': [],
        }
```

### EchoStreamModel에서 사용

```python
# models/echostream_model.py
def forward(self, src_tokens, src_lengths):
    # 1. Encoder 호출
    encoder_out = self.encoder(src_tokens, src_lengths)
    
    # 2. ✅ List에서 첫 번째 요소 추출
    encoder_hidden = encoder_out['encoder_out'][0]  # [T', B, D]
    
    # 3. ✅ Padding mask 추출 (있을 경우)
    encoder_padding_mask = (
        encoder_out['encoder_padding_mask'][0] 
        if encoder_out['encoder_padding_mask'] 
        else None
    )
    
    # 4. 디코더에 전달
    asr_out = self.asr_ctc_decoder(
        encoder_out=encoder_hidden,  # 텐서 직접 전달
        encoder_padding_mask=encoder_padding_mask,
    )
```

---

## 🚨 자주 발생하는 오류

### 오류 1: List 인덱싱 누락

```python
# ❌ 잘못된 코드
encoder_hidden = encoder_out['encoder_out']  # List 전체
ctc_decoder(encoder_out=encoder_hidden)  # TypeError!

# ✅ 올바른 코드
encoder_hidden = encoder_out['encoder_out'][0]  # 첫 번째 요소
ctc_decoder(encoder_out=encoder_hidden)
```

### 오류 2: 차원 순서 불일치

```python
# ❌ 잘못된 코드 (Batch-first)
encoder_out = {
    'encoder_out': [tensor],  # [B, T, D] 형식
}

# ✅ 올바른 코드 (Time-first)
encoder_out = {
    'encoder_out': [tensor],  # [T, B, D] 형식
}
```

### 오류 3: Padding mask 처리

```python
# ❌ 잘못된 코드
encoder_padding_mask = encoder_out['encoder_padding_mask']  # List 전체

# ✅ 올바른 코드
if encoder_out['encoder_padding_mask']:
    encoder_padding_mask = encoder_out['encoder_padding_mask'][0]
else:
    encoder_padding_mask = None
```

### 오류 4: 키 이름 불일치

```python
# ❌ 잘못된 코드
encoder_out = {
    'output': [tensor],  # 'encoder_out'이 아님!
}

# ✅ 올바른 코드
encoder_out = {
    'encoder_out': [tensor],  # 정확한 키 이름
}
```

---

## 📋 체크리스트

Encoder를 구현할 때 다음을 확인하세요:

- [ ] `encoder_out`은 **List** 형태인가? (`[tensor]` 형식)
- [ ] 텐서 차원이 **[T, B, D]** (time-first)인가?
- [ ] `encoder_padding_mask`도 **List** 형태인가?
- [ ] Padding mask 차원이 **[B, T]**인가?
- [ ] 필수 키들이 모두 있는가?
  - `encoder_out`
  - `encoder_padding_mask`
  - `encoder_embedding` (빈 리스트 가능)
  - `encoder_states` (빈 리스트 가능)
  - `src_tokens` (빈 리스트 가능)
  - `src_lengths` (빈 리스트 가능)

---

## 🔍 디버깅 팁

### 1. Encoder 출력 확인

```python
encoder_out = encoder(src_tokens, src_lengths)

# 타입 확인
print(type(encoder_out['encoder_out']))  # <class 'list'>
print(type(encoder_out['encoder_out'][0]))  # <class 'torch.Tensor'>

# 차원 확인
print(encoder_out['encoder_out'][0].shape)  # [T, B, D]
```

### 2. 디코더 입력 확인

```python
# 디코더에 전달하기 전에 확인
encoder_hidden = encoder_out['encoder_out'][0]
print(f"Encoder hidden shape: {encoder_hidden.shape}")  # [T, B, D]

# Padding mask 확인
if encoder_out['encoder_padding_mask']:
    mask = encoder_out['encoder_padding_mask'][0]
    print(f"Padding mask shape: {mask.shape}")  # [B, T]
```

### 3. 오류 메시지 해석

```
TypeError: expected Tensor, got list
→ encoder_out['encoder_out']를 [0]으로 인덱싱하지 않았음

KeyError: 'encoder_out'
→ 키 이름이 잘못되었거나 딕셔너리 구조가 다름

RuntimeError: Expected 3D tensor, got 2D
→ 차원 순서가 잘못되었거나 transpose가 필요함
```

---

## 📚 참고: StreamSpeech 원본 코드

StreamSpeech의 Conformer 인코더도 동일한 형식을 사용합니다:

```python
# StreamSpeech_analysis/researches/ctc_unity/models/s2t_conformer.py
def _forward(self, src_tokens, src_lengths):
    # ... 인코딩 로직 ...
    
    return {
        "encoder_out": [x],  # List of [T, B, C]
        "encoder_padding_mask": (
            [encoder_padding_mask] if encoder_padding_mask.any() else []
        ),
        "encoder_embedding": [],
        "encoder_states": encoder_states,
        "src_tokens": [],
        "src_lengths": [],
    }
```

---

## ✅ 결론

**핵심 원칙**:
1. Encoder 출력은 **반드시 List 형태**로 반환
2. 텐서는 **Time-first** 형식 `[T, B, D]`
3. 디코더에 전달할 때는 **`[0]`으로 인덱싱**하여 텐서 추출
4. Padding mask도 **List 형태**이며, 비어있을 수 있음

이 형식을 정확히 따르지 않으면 디코더에서 오류가 발생합니다!

---

**마지막 업데이트**: 2025-01-XX  
**버전**: 1.0

