import pickle
from gtts import gTTS
import whisper

def text_to_speech(text, filename="output.mp3", lang="ko"):
    """
    텍스트를 음성으로 변환하여 파일로 저장합니다.
    
    Parameters:
        text (str): 변환할 텍스트
        filename (str): 저장할 오디오 파일 이름 (기본: output.mp3)
        lang (str): 언어 코드 (예: 한국어는 "ko")
    """
    tts = gTTS(text=text, lang=lang)
    tts.save(filename)
    print(f"음성 파일 '{filename}' 이(가) 생성되었습니다.")

def transcribe_audio(audio_path, model_size="medium"):
    """
    Whisper 모델을 사용하여 오디오 파일을 텍스트로 변환합니다.
    
    Parameters:
        audio_path (str): 오디오 파일 경로
        model_size (str): 사용할 Whisper 모델 사이즈 (예: "medium")
    
    Returns:
        transcribed_text (str): 변환된 텍스트
    """
    print(f"Whisper '{model_size}' 모델을 불러오는 중...")
    model = whisper.load_model(model_size)
    print("모델 로딩 완료. 음성을 텍스트로 변환합니다...")
    result = model.transcribe(audio_path)
    transcribed_text = result["text"]
    print("음성 -> 텍스트 변환 결과:")
    print(transcribed_text)
    return transcribed_text

def load_classifier(model_path):
    """
    pickle 파일로 저장된 분류 모델을 로드합니다.
    
    Parameters:
        model_path (str): 모델 파일 경로 (예: "badword_model.pkl")
    
    Returns:
        classifier: 로드된 분류 모델 객체
    """
    with open(model_path, "rb") as f:
        classifier = pickle.load(f)
    print(f"분류 모델 '{model_path}' 로드 완료.")
    return classifier

def load_vectorizer(vectorizer_path):
    """
    pickle 파일로 저장된 벡터라이저를 로드합니다.
    
    Parameters:
        vectorizer_path (str): 벡터라이저 파일 경로 (예: "vectorizer.pkl")
    
    Returns:
        vectorizer: 로드된 벡터라이저 객체
    """
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)
    print(f"벡터라이저 '{vectorizer_path}' 로드 완료.")
    return vectorizer

def classify_text(text, classifier, vectorizer=None):
    """
    입력 텍스트가 욕설인지 판별합니다.
    
    Parameters:
        text (str): 판별할 텍스트
        classifier: 미리 로드한 분류 모델
        vectorizer: 텍스트를 수치형 데이터로 변환하기 위한 벡터라이저 
                    (Pipeline이 아닌 경우 필요하며, Pipeline 모델인 경우 None으로 사용)
    
    Returns:
        result: 분류 결과 (예: 1이면 욕설, 0이면 욕설 아님)
    """
    # Pipeline일 경우 vectorizer가 내장되어 있으므로 바로 사용 가능
    if vectorizer is not None:
        text_vector = vectorizer.transform([text])
        prediction = classifier.predict(text_vector)
    else:
        prediction = classifier.predict([text])
    return prediction[0]

if __name__ == "__main__":
    # 1. 사용자로부터 텍스트 입력 받기
    input_text = input("텍스트를 입력하세요: ")

    # 2. 텍스트 -> 음성 (TTS)
    audio_filename = "output.mp3"
    text_to_speech(input_text, filename=audio_filename, lang="ko")

    # 3. 음성 -> 텍스트 (ASR: Whisper Medium 모델 사용)
    transcribed_text = transcribe_audio(audio_filename, model_size="medium")

    # 4. 텍스트가 욕설인지 판별 (사용자 제작 .pkl 모델 사용)
    classifier_model_path = "./badword_model.pkl"  # 본인이 만든 분류 모델 파일 경로
    classifier = load_classifier(classifier_model_path)
    
    # [옵션] 만약 분류 모델이 Pipeline이 아니라면 벡터라이저를 별도로 로드합니다.
    # Pipeline 모델을 사용 중이면 아래 부분은 주석 처리하거나 vectorizer를 None으로 설정하세요.
    vectorizer_model_path = "./vectorizer.pkl"  # 본인이 만든 벡터라이저 파일 경로
    try:
        vectorizer = load_vectorizer(vectorizer_model_path)
    except FileNotFoundError:
        print(f"벡터라이저 파일 '{vectorizer_model_path}'을(를) 찾을 수 없습니다. Pipeline 모델이라면 이 부분은 무시하세요.")
        vectorizer = None

    result = classify_text(transcribed_text, classifier, vectorizer=vectorizer)

    # 5. 결과 출력
    if result == 1:
        print("욕설이 감지되었습니다.")
    else:
        print("욕설이 감지되지 않았습니다.")
