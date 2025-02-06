import torch
import json
import whisper
import evaluate
import librosa
import numpy as np
from scipy.stats import mode
from datasets import load_dataset
from transformers import WhisperForConditionalGeneration, WhisperProcessor, WhisperFeatureExtractor, TrainingArguments, Trainer, DataCollatorForSeq2Seq
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
small_dataset = dataset.select(range(10))  # ✅ 10개 샘플만 선택 (테스트용)

# ✅ Pretrained Whisper 모델 및 프로세서 로드
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
processor = WhisperProcessor.from_pretrained("openai/whisper-small")
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")

# ✅ 반복된 값 제거 (강화된 필터링 적용)
def remove_repeated_values(audio_array, window_size=20, threshold_factor=0.02, mode_threshold=0.5, enable_mode_filter=True):
    """
    반복되는 특정 값과 일정 구간 동안 변화가 거의 없는 부분을 제거.
    """
    if len(audio_array) < window_size:
        return audio_array  # 데이터가 너무 짧으면 처리하지 않음

    # ✅ 최빈값(Mode) 기반 필터링 (선택적 적용)
    if enable_mode_filter:
        most_common_value, count = mode(audio_array, keepdims=False)
        most_common_value = most_common_value[0]
        if count / len(audio_array) >= mode_threshold:  # 특정 값이 50% 이상 차지하면 삭제
            audio_array = audio_array[audio_array != most_common_value]

    # ✅ Sliding Window로 평균 변화량 계산
    diff = np.abs(np.diff(audio_array))
    avg_diff = np.convolve(diff, np.ones(window_size) / window_size, mode='valid')

    # ✅ Adaptive Threshold: 전체 데이터의 변화량을 기반으로 동적으로 설정
    threshold = np.mean(avg_diff) * threshold_factor

    # ✅ 특정 임계값 이하의 변화만 있는 구간을 제거
    stable_indices = np.where(avg_diff > threshold)[0] + 1  # 변화가 있는 인덱스만 유지
    filtered_audio = audio_array[stable_indices] if len(stable_indices) > 0 else audio_array

    return np.array(filtered_audio, dtype="float32")

# ✅ 데이터 전처리 함수 정의
def preprocess_function(batch):
    audio_array = batch["audio"]["array"].astype("float32")
    orig_sr = batch["audio"]["sampling_rate"]

    # ✅ 샘플링 레이트 강제 변환 (16kHz로 변환)
    if orig_sr != 16000:
        audio_array = librosa.resample(audio_array, orig_sr=orig_sr, target_sr=16000)

    # ✅ 무음 제거 (librosa.effects.trim 사용) → 동적 감도 조절
    audio_array, _ = librosa.effects.trim(audio_array, top_db=np.clip(np.std(audio_array) * 10, 15, 30))  

    # ✅ 연속된 값 제거 (최빈값 + Sliding Window 방식 적용)
    audio_array = remove_repeated_values(audio_array, window_size=20, threshold_factor=0.02, mode_threshold=0.5)

    # ✅ 너무 짧은 오디오는 필터링
    min_length = 1 * 16000  # 최소 1초 이상 유지
    if len(audio_array) < min_length:
        return None  # None을 반환하면 자동으로 필터링됨

    # ✅ 너무 긴 오디오는 랜덤 샘플링하여 30초로 제한
    max_length = 30 * 16000  # 30초 * 16000Hz
    if len(audio_array) > max_length:
        start_idx = random.randint(0, len(audio_array) - max_length)
        audio_array = audio_array[start_idx:start_idx + max_length]

    # ✅ Feature Extractor 적용
    processed = feature_extractor(audio_array, sampling_rate=16000)
    batch["input_values"] = processed["input_features"][0]

    # ✅ labels 처리
    labels = processor.tokenizer(batch["sentence"], truncation=True, padding="longest").input_ids
    if not labels:
        labels = [processor.tokenizer.pad_token_id]  # 빈 경우 패딩 토큰 추가

    batch["labels"] = labels
    return batch

# ✅ 데이터셋 전처리 적용 (None 값 제거 포함)
small_dataset = small_dataset.map(preprocess_function, remove_columns=["audio", "sentence"])
small_dataset = small_dataset.filter(lambda x: x is not None)  # None 값 제거

# ✅ 변환된 데이터 확인 (디버깅)
print("✅ 데이터셋 샘플 확인")
print(small_dataset[0])  # 첫 번째 데이터 확인
print("🎧 변환된 오디오 길이:", len(small_dataset[0]["input_values"]))

# ✅ 이제 불필요한 반복된 값이 확실히 제거될 것입니다! 🚀
