import pandas as pd
import re
import pickle
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

# ✅ 한국어 불용어 리스트 추가
korean_stopwords = [
    "이", "그", "저", "것", "및", "의", "를", "은", "는", "이랑", "과", "와", "으로", "에서", "한",
    "들", "다", "자", "좀", "을", "하다", "등", "보다", "라고", "였다", "합니다", "이다", "했다"
]

# ✅ 텍스트 데이터 전처리 함수 (불용어 제거 포함)
def clean_text(text):
    text = text.lower()  # 소문자 변환
    text = re.sub(r'\W', ' ', text)  # 특수문자 제거
    text = ' '.join(word for word in text.split() if word not in korean_stopwords)  # 불용어 제거
    return text

# ✅ TXT 파일에서 데이터 불러오기
data = []
with open("badwords.txt", "r", encoding="utf-8") as file:
    for line in file:
        parts = line.strip().split("|")  # | 기준으로 분리
        if len(parts) < 2:  # 오류 방지
            continue  
        text = "|".join(parts[:-1])  # 마지막 값(label)을 제외한 모든 부분을 텍스트로 합침
        label = parts[-1]  # 마지막 값이 label
        try:
            label = int(label)  # 숫자로 변환
            data.append((text, label))
        except ValueError:
            print(f"⚠️ 잘못된 라인 스킵: {line.strip()}")  # 오류 발생 시 출력

df = pd.DataFrame(data, columns=["text", "label"])

# ✅ 데이터 불균형 해결 (비속어와 정상 문장 개수 균형 맞추기)
label_counts = df["label"].value_counts()
min_class = label_counts.min()  # 가장 적은 클래스 개수

df_balanced = pd.concat([
    df[df["label"] == 0].sample(n=min_class, random_state=42),  # 정상 문장 랜덤 샘플링
    df[df["label"] == 1].sample(n=min_class, random_state=42)   # 비속어 랜덤 샘플링
])

# ✅ 전처리 적용
df_balanced["clean_text"] = df_balanced["text"].apply(clean_text)

# ✅ TF-IDF 벡터화 (텍스트 → 숫자로 변환)
vectorizer = TfidfVectorizer(max_features=5000)

# ✅ 기존 모델 불러오기 (있는 경우)
if os.path.exists("badword_model.pkl") and os.path.exists("vectorizer.pkl"):
    with open("badword_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    # ✅ 기존 모델의 데이터와 새로운 데이터를 결합하여 학습
    X = vectorizer.transform(df_balanced["clean_text"])  # 기존 데이터 벡터화
    y = df_balanced["label"]

    # ✅ 모델에 새로운 데이터 추가 학습
    model.fit(X, y)
    print("🔄 기존 모델에 새로운 데이터를 추가 학습 완료!")
else:
    # ✅ 기존 모델이 없으면 새로 학습
    X = vectorizer.fit_transform(df_balanced["clean_text"])
    y = df_balanced["label"]
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X, y)
    print("🆕 새 모델 학습 완료!")

# ✅ 데이터 분할 (X, y가 항상 존재하도록 수정됨!)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ 모델 평가
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred) * 100
print(f"🎯 모델 정확도: {accuracy:.2f}%")

# ✅ 학습된 모델 저장
with open("badword_model.pkl", "wb") as f:
    pickle.dump(model, f)

# ✅ 벡터화 모델 저장
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("💾 모델 저장 완료!")
