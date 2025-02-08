import torch
import json
import whisper
import evaluate
import librosa
import numpy as np
from scipy.stats import mode
from datasets import load_dataset
from transformers import (
    WhisperForConditionalGeneration, WhisperProcessor, WhisperFeatureExtractor,
    TrainingArguments, Trainer, DataCollatorForSeq2Seq
)
from jiwer import wer
import os
import random
from datetime import datetime
from huggingface_hub import login

# ✅ Hugging Face 로그인
HF_TOKEN = os.getenv("HF_TOKEN") or input("🔑 Hugging Face Access Token 입력: ")
login(token=HF_TOKEN)

# ✅ 결과 저장 폴더 설정
SAVE_PATH = "./whisper-ko-finetuned/"
os.makedirs(SAVE_PATH, exist_ok=True)

# ✅ 데이터셋 다운로드 및 일부 데이터 선택
dataset = load_dataset("jp1924/KsponSpeech", split="train")

# ⚠️ 전처리 전에 원본 오디오 길이가 최소 1초(16000 샘플) 이상인 샘플만 남깁니다.
min_length = 1 * 16000
dataset = dataset.filter(lambda x: len(x["audio"]["array"]) >= min_length)

# 테스트용으로 10개 샘플만 선택 (원하는 개수로 조정 가능)
small_dataset = dataset.select(range(10))

# ✅ Pretrained Whisper 모델 및 프로세서 로드
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
processor = WhisperProcessor.from_pretrained("openai/whisper-small")
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")

# --- 요약 통계 정보를 출력하는 함수 ---
def summarize_audio(audio_array, num_samples=5):
    """오디오 배열에 대한 요약 통계 정보를 반환합니다."""
    summary = {
        'length': len(audio_array),
        'mean': float(np.mean(audio_array)),
        'std': float(np.std(audio_array)),
        'min': float(np.min(audio_array)),
        'max': float(np.max(audio_array)),
        'head': audio_array[:num_samples].tolist(),
        'tail': audio_array[-num_samples:].tolist()
    }
    return summary

# ✅ 반복된 값 제거 (강화된 필터링 적용)
def remove_repeated_values(audio_array, window_size=20, threshold_factor=0.02, mode_threshold=0.5, enable_mode_filter=True):
    """
    반복되는 특정 값과 일정 구간 동안 변화가 거의 없는 부분을 제거합니다.
    기존 방식에서는 무음(0 근처의 값)도 제거될 수 있는 문제가 있었으므로,
    이를 개선하여 무음 부분은 유지하도록 합니다.
    """
    if len(audio_array) < window_size:
        return audio_array  # 데이터가 너무 짧으면 처리하지 않음

    # ✅ 모드(최빈값) 기반 필터링 (무음에 해당하는 값은 제거하지 않음)
    if enable_mode_filter:
        mode_result = mode(audio_array, keepdims=False)
        most_common_value, count = mode_result.mode, mode_result.count
        # 만약 최빈값이 numpy 배열이면 첫번째 요소를 가져옴, 아니라면 그대로 사용
        if not np.isscalar(most_common_value):
            most_common_value = most_common_value[0]
            count = count[0]
        # 최빈값이 0(또는 0에 매우 가까운 값)이 아니라면 제거 진행
        if not np.isclose(most_common_value, 0, atol=1e-3) and (count / len(audio_array)) >= mode_threshold:
            audio_array = audio_array[audio_array != most_common_value]

    # ✅ Sliding Window로 변화량 계산
    diff = np.abs(np.diff(audio_array))
    avg_diff = np.convolve(diff, np.ones(window_size) / window_size, mode='valid')
    # ✅ Adaptive Threshold: 전체 데이터의 변화량을 기반으로 동적으로 설정
    threshold = np.mean(avg_diff) * threshold_factor

    # ✅ 변화량이 임계치 이하인 구간은 안정적이라고 판단하여 해당 인덱스는 제거
    # (+1: diff 연산으로 길이가 1 줄어듦)
    stable_indices = np.where(avg_diff > threshold)[0] + 1  
    filtered_audio = audio_array[stable_indices] if len(stable_indices) > 0 else audio_array

    return np.array(filtered_audio, dtype="float32")

# ✅ 데이터 전처리 함수 정의 (요약 통계 정보를 출력)
def preprocess_function(batch):
    print(f"=== Processing sample id: {batch['id']} ===")
    
    # 원본 오디오 정보 요약 출력
    audio_array = batch["audio"]["array"].astype("float32")
    orig_sr = batch["audio"]["sampling_rate"]
    original_summary = summarize_audio(audio_array)
    print(f"Original audio summary: {original_summary}")
    
    # 샘플링 레이트 강제 변환 (16kHz로 변환)
    if orig_sr != 16000:
        audio_array = librosa.resample(audio_array, orig_sr=orig_sr, target_sr=16000)
        print("Resampled audio to 16000 Hz")
    
    # 무음 제거 (librosa.effects.trim 사용) → 동적 감도 조절
    audio_array, _ = librosa.effects.trim(
        audio_array, top_db=np.clip(np.std(audio_array) * 10, 15, 30)
    )
    print(f"After trim, audio summary: {summarize_audio(audio_array)}")
    
    # 연속된 값 제거 (최빈값 + Sliding Window 방식 적용)
    audio_array = remove_repeated_values(
        audio_array, window_size=20, threshold_factor=0.02, mode_threshold=0.5
    )
    print(f"After remove_repeated_values, audio summary: {summarize_audio(audio_array)}")
    
    # 처리 후 길이가 1초 미만이면 드롭 (None 대신 플래그 사용)
    if len(audio_array) < min_length:
        print(f"Sample id {batch['id']} dropped: processed length {len(audio_array)} is below minimum {min_length}")
        return {
            "id": batch["id"],
            "input_values": None,
            "labels": None,
            "drop": True
        }
    
    # 너무 긴 오디오는 랜덤 샘플링하여 30초로 제한
    max_length = 30 * 16000  # 30초
    if len(audio_array) > max_length:
        start_idx = random.randint(0, len(audio_array) - max_length)
        audio_array = audio_array[start_idx:start_idx + max_length]
        print(f"Trimmed long audio to max_length: {max_length}")
    
    # Feature Extractor 적용
    processed = feature_extractor(audio_array, sampling_rate=16000)
    batch["input_values"] = processed["input_features"][0]
    
    # labels 처리  
    # → tokenizer.pad()가 올바르게 동작하도록, tokenizer의 전체 인코딩(dict)을 반환합니다.
    labels_encoding = processor.tokenizer(batch["sentence"], truncation=True, padding="longest")
    if not labels_encoding["input_ids"]:
        labels_encoding["input_ids"] = [processor.tokenizer.pad_token_id]
    batch["labels"] = labels_encoding
    # (모델 학습 시, Trainer 내부에서 labels["input_ids"]를 tensor로 변환합니다.)
    
    # 드롭 플래그 False 설정
    batch["drop"] = False
    print(f"Sample id {batch['id']} processed successfully.\n")
    return batch

# ✅ 데이터셋 전처리 적용 (원래의 "audio"와 "sentence" 컬럼은 제거)
small_dataset = small_dataset.map(preprocess_function, remove_columns=["audio", "sentence"])

# ✅ 드롭 플래그가 True인 샘플 제거
small_dataset = small_dataset.filter(lambda x: not x["drop"])
# 사용 후 drop 플래그 컬럼 제거 (필요 시)
small_dataset = small_dataset.remove_columns("drop")

# ✅ 변환된 데이터 확인 (디버깅)
print("✅ 데이터셋 샘플 확인")
print(small_dataset[0])
print("🎧 변환된 오디오 길이:", len(small_dataset[0]["input_values"]))

# -----------------------------------------------------------------------------
# 학습 (fine-tuning) 부분 시작
# -----------------------------------------------------------------------------

# TrainingArguments 설정 (환경에 맞게 조정)
training_args = TrainingArguments(
    output_dir=SAVE_PATH,
    per_device_train_batch_size=1,    # GPU 메모리에 따라 조절
    num_train_epochs=3,               # 에폭 수 조절
    logging_steps=10,
    save_steps=50,
    evaluation_strategy="no",         # FutureWarning: eval_strategy 사용 권장 (당장은 무시)
    fp16=False,                       # GPU가 FP16을 지원한다면 True로 변경 가능
)

# DataCollator 설정: Whisper 모델은 입력 시퀀스와 라벨 모두 패딩이 필요합니다.
data_collator = DataCollatorForSeq2Seq(
    tokenizer=processor.tokenizer,    # 수정: processor.tokenizer를 사용
    model=model,
    padding=True,
)

# Trainer 설정: train_dataset으로 small_dataset 사용
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_dataset,
    data_collator=data_collator,
    tokenizer=processor.tokenizer,    # 수정: processor.tokenizer를 사용 (FutureWarning: processing_class 사용 권장)
)

# 모델 학습 시작
trainer.train()
