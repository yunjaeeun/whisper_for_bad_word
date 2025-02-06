import torch
import json
import whisper
import evaluate
from datasets import load_dataset
from transformers import WhisperForConditionalGeneration, WhisperProcessor, TrainingArguments, Trainer
from jiwer import wer
import os
from datetime import datetime
from huggingface_hub import login

login(token="")


# 1️⃣ 결과 저장 폴더 설정
save_path = "./whisper-ko-finetuned/"
os.makedirs(save_path, exist_ok=True)  # 폴더가 없으면 자동 생성

# 2️⃣ Common Voice 한국어 데이터셋 다운로드
dataset = load_dataset("mozilla-foundation/common_voice_12_0", "Korean", split="train", trust_remote_code=True)
# dataset = load_dataset("mozilla-foundation/common_voice_10_0", "ko", split="train", trust_remote_code=True)


# 3️⃣ Pretrained Whisper 모델 및 프로세서 로드
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
processor = WhisperProcessor.from_pretrained("openai/whisper-small")

# 4️⃣ 데이터 전처리 함수 정의 (음성 → 텍스트)
def preprocess_function(batch):
    audio = batch["audio"]
    batch["input_values"] = processor(audio["array"], sampling_rate=16000).input_values[0]
    batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
    return batch

dataset = dataset.map(preprocess_function, remove_columns=["audio", "sentence"])

# 5️⃣ WER (Word Error Rate) 평가 지표 설정
wer_metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
    wer_result = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer_result}

# 6️⃣ 학습 설정 (TrainingArguments)
training_args = TrainingArguments(
    output_dir=save_path,  # 모델 저장 경로
    evaluation_strategy="epoch",  # 매 에포크마다 평가
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=1e-5,
    num_train_epochs=5,  # 5 에포크 학습
    weight_decay=0.01,
    save_total_limit=2,
    logging_dir="./logs",
    fp16=torch.cuda.is_available(),  # GPU 사용 시 FP16 적용
    save_steps=500,  # 500 스텝마다 모델 저장
)

# 7️⃣ Trainer 객체 생성 및 학습 시작
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=dataset.select(range(1000)),  # 일부 데이터로 평가 진행
    tokenizer=processor,  # tokenizer 수정
    compute_metrics=compute_metrics,
)

trainer.train()

# 8️⃣ 학습된 모델 저장
model.save_pretrained(save_path)
processor.save_pretrained(save_path)

# 9️⃣ 학습 후 정확도(WER) 평가
test_audio = "audio.m4a"  # 평가용 오디오 파일 (사용자가 지정)
ground_truth = "내가그린기린그림"  # 해당 오디오의 정답 텍스트

# Whisper 기본 모델 불러오기 및 변환 수행
model = whisper.load_model("small")  # Fine-Tuned 모델이 아닌 기본 Whisper 사용
result = model.transcribe(test_audio)
whisper_output = result["text"]

# 10️⃣ WER 계산
wer_result = wer(ground_truth, whisper_output)
accuracy = (1 - wer_result) * 100

# 11️⃣ 결과 JSON 파일로 저장 (파일명에 타임스탬프 추가)
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
json_filename = f"{save_path}/wer_results_{timestamp}.json"

wer_data = {
    "model": "Whisper (Fine-Tuned)",
    "WER": round(wer_result, 4),
    "Accuracy (%)": round(accuracy, 2),
    "Ground Truth": ground_truth,
    "STT Output": whisper_output
}

with open(json_filename, "w", encoding="utf-8") as f:
    json.dump(wer_data, f, ensure_ascii=False, indent=4)

print("✅ Whisper 한국어 학습 완료!")
print(f"✅ 모델이 {save_path} 폴더에 저장되었습니다.")
print(f"✅ 정확도 체크 결과가 {json_filename} 파일에 저장되었습니다.")
