import torch
from datasets import load_dataset
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Union

# 프로세서 로드
processor = WhisperProcessor.from_pretrained("openai/whisper-small")

# 데이터셋 로드
dataset = load_dataset("jp1924/KsponSpeech", split="train")

# 데이터 전처리를 위한 함수
def prepare_dataset(batch):
    # 오디오 처리
    audio = batch["audio"]
    input_features = processor(
        audio["array"],
        sampling_rate=audio["sampling_rate"],
        return_tensors="pt"
    ).input_features[0]

    # 텍스트 처리
    batch["labels"] = processor.tokenizer(batch["text"]).input_ids
    batch["input_features"] = input_features

    return batch

# 데이터셋 전처리
processed_dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names)

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # 입력 피처와 라벨 가져오기
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # 라벨 패딩
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # -100으로 패딩된 부분 replace
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # 모든 텐서를 배치에 추가
        batch["labels"] = labels

        return batch

# 데이터 콜레이터 인스턴스 생성
data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# 평가 메트릭 계산 함수
def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # -100을 무시
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # ids를 텍스트로 디코딩
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    # WER 계산
    wer = 100 * sum(pred != label for pred, label in zip(pred_str, label_str)) / len(pred_str)
    
    return {"wer": wer}

# 학습 인자 설정
training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-kspon",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=4000,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    eval_steps=1000,
    save_strategy="steps",
    save_steps=1000,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
)

# 트레이너 초기화
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=processed_dataset,
    eval_dataset=processed_dataset.select(range(100)),  # 평가용 데이터셋 (예시로 100개만 사용)
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# 학습 시작
trainer.train()

# 모델 저장
trainer.save_model("./whisper-kspon-final")