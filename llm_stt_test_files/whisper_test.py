# import whisper

# model = whisper.load_model("small")

# result = model.transcribe("audio.mp3", fp16=False)

# print(result["text"])

from datasets import load_dataset
from huggingface_hub import login

login(token="")
dataset = load_dataset("jp1924/KsponSpeech", split="train")
small_dataset = dataset.select(range(100))  # ✅ 100개 샘플만 선택
print(small_dataset)
print(next(iter(small_dataset)))
