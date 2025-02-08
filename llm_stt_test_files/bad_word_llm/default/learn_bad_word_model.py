import pandas as pd
import re
import pickle
import os
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ✅ 데이터 파일 경로 설정
train_file_path = "train_data.txt"
test_file_path = "test_data.txt"
model_path = "badword_model.pkl"
vectorizer_path = "vectorizer.pkl"
log_file_path = "model_accuracy_log.csv"

# ✅ TXT 파일에서 데이터 불러오기
def load_data(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split("|")  # '|' 기준으로 분리
            if len(parts) < 2:
                continue  
            text = "|".join(parts[:-1])  # 마지막 요소(label)를 제외한 나머지를 합침
            label = parts[-1]  # 마지막 요소를 라벨로 저장
            try:
                label = int(label)  # 숫자로 변환
                data.append((text, label))
            except ValueError:
                continue  
    return pd.DataFrame(data, columns=["text", "label"])

# ✅ 학습 및 평가 데이터 로드
train_df = load_data(train_file_path)
test_df = load_data(test_file_path)

# ✅ TF-IDF 벡터화 (텍스트 → 숫자로 변환)
vectorizer = TfidfVectorizer(max_features=5000)

# ✅ 기존 모델 불러오기 (있는 경우)
if os.path.exists(model_path) and os.path.exists(vectorizer_path):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)

    # ✅ 기존 모델의 데이터와 새로운 데이터를 결합하여 학습
    X_train = vectorizer.transform(train_df["text"])  # 학습 데이터 벡터화
    y_train = train_df["label"]

    # ✅ 모델에 새로운 데이터 추가 학습
    model.fit(X_train, y_train)
    print("🔄 기존 모델에 새로운 데이터를 추가 학습 완료!")
else:
    # ✅ 기존 모델이 없으면 새로 학습
    X_train = vectorizer.fit_transform(train_df["text"])
    y_train = train_df["label"]
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    print("🆕 새 모델 학습 완료!")

# ✅ 평가 데이터 벡터화
X_test = vectorizer.transform(test_df["text"])
y_test = test_df["label"]

# ✅ 모델 평가
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred) * 100
print(f"🎯 모델 정확도: {accuracy:.2f}%")

# ✅ 학습된 모델 저장
with open(model_path, "wb") as f:
    pickle.dump(model, f)

# ✅ 벡터화 모델 저장
with open(vectorizer_path, "wb") as f:
    pickle.dump(vectorizer, f)

print("💾 모델 저장 완료!")

# ✅ 학습 정확도 로그 저장
log_entry = pd.DataFrame({
    "timestamp": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
    "accuracy": [accuracy]
})

# 기존 로그가 존재하는지 확인하고 이어서 저장 (첫 실행 시 헤더 추가)
if os.path.exists(log_file_path):
    log_entry.to_csv(log_file_path, mode="a", header=False, index=False, encoding="utf-8-sig")
else:
    log_entry.to_csv(log_file_path, mode="w", header=True, index=False, encoding="utf-8-sig")

print(f"📄 학습 로그 저장 완료: {log_file_path}")
