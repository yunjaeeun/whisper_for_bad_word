import os
import sys
import time
import pickle
import requests
import subprocess
import torch
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import whisper
import whisper.audio as whisper_audio

# ---------------------------------------------------------------------
# ffmpeg 실행 파일 경로 지정
if getattr(sys, 'frozen', False):
    base_path = os.path.dirname(sys.executable)  # 배포된 실행 파일 위치
    ffmpeg_executable = os.path.join(base_path, "ffmpeg", "ffmpeg.exe")
    print(f"[DEBUG] 배포 환경의 ffmpeg 경로: {ffmpeg_executable}")
else:
    ffmpeg_executable = r"C:\meeple\ffmpeg\ffmpeg.exe"

# Whisper에서 ffmpeg 실행 경로 명시적으로 설정
os.environ["PATH"] += os.pathsep + os.path.dirname(ffmpeg_executable)
whisper_audio.FFMPEG = ffmpeg_executable  # Whisper가 ffmpeg 실행하도록 설정

# ---------------------------------------------------------------------

# 전역 변수에 모델 캐싱 (한 번만 로딩)
CACHED_MODEL = None

def get_whisper_model(model_size="medium"):
    global CACHED_MODEL
    if CACHED_MODEL is None:
        print(f"Whisper '{model_size}' 모델을 처음 로딩 중...")
        CACHED_MODEL = whisper.load_model(model_size, device="cuda" if torch.cuda.is_available() else "cpu")
        print("모델 로딩 완료.")
    else:
        print("캐시된 Whisper 모델 사용.")
    return CACHED_MODEL

def transcribe_audio(audio_path, model_size="medium"):
    """
    Whisper 모델을 사용하여 오디오 파일을 텍스트로 변환합니다.
    """
    model = get_whisper_model(model_size)
    print("음성을 텍스트로 변환합니다...")

    try:
        result = model.transcribe(audio_path)
        transcribed_text = result["text"]
        print("음성 -> 텍스트 변환 결과:")
        print(transcribed_text)
        return transcribed_text
    except Exception as e:
        print(f"[ERROR] Whisper 변환 중 오류 발생: {e}")
        return ""

def load_classifier(model_path):
    """
    pickle 파일로 저장된 분류 모델을 로드합니다.
    """
    with open(model_path, "rb") as f:
        classifier = pickle.load(f)
    print(f"분류 모델 '{model_path}' 로드 완료.")
    return classifier

def load_vectorizer(vectorizer_path):
    """
    pickle 파일로 저장된 벡터라이저를 로드합니다.
    """
    try:
        with open(vectorizer_path, "rb") as f:
            vectorizer = pickle.load(f)
        print(f"벡터라이저 '{vectorizer_path}' 로드 완료.")
        return vectorizer
    except FileNotFoundError:
        print(f"[WARNING] 벡터라이저 '{vectorizer_path}'을(를) 찾을 수 없습니다. Pipeline 모델이라면 이 부분은 무시하세요.")
        return None

def classify_text(text, classifier, vectorizer=None):
    """
    입력 텍스트가 욕설인지 판별합니다.
    """
    if vectorizer is not None:
        text_vector = vectorizer.transform([text])
        prediction = classifier.predict(text_vector)
    else:
        prediction = classifier.predict([text])
    return prediction[0]

def extract_user_nickname_from_filename(file_path):
    """
    파일명이 'userNickname_audio_timestamp.webm' 형식일 때,
    첫 번째 언더바 앞의 문자열을 닉네임으로 추출합니다.
    """
    base = os.path.basename(file_path)
    parts = base.split('_')
    return parts[0] if len(parts) >= 2 else ""

def process_audio_file(audio_file_path):
    """
    음성 파일을 Whisper로 전사하고 욕설 여부를 판별합니다.
    욕설이 감지되면 SpringBoot 서버에 파일과 전사된 텍스트를 전송하고,
    욕설이 아니면 해당 파일을 삭제합니다.
    """
    print("처리할 파일:", audio_file_path)

    if not os.path.exists(audio_file_path):
        print(f"[ERROR] 파일이 존재하지 않습니다: {audio_file_path}")
        return

    user_nickname = extract_user_nickname_from_filename(audio_file_path)
    print("파일명에서 추출된 userNickname:", user_nickname)

    transcribed_text = transcribe_audio(audio_file_path, model_size="medium")

    classifier_model_path = r"C:\meeple\models\badword_model.pkl"
    classifier = load_classifier(classifier_model_path)

    vectorizer_model_path = r"C:\meeple\models\vectorizer.pkl"
    vectorizer = load_vectorizer(vectorizer_model_path)

    result = classify_text(transcribed_text, classifier, vectorizer=vectorizer)

    if result == 1:
        print("욕설이 감지되었습니다.")
        springboot_api_url = "https://letsmeeple.duckdns.org/api/ai/voice-log"
        try:
            with open(audio_file_path, "rb") as f:
                files = {"audio": f}
                data = {"voiceLog": transcribed_text, "userNickname": user_nickname}
                response = requests.post(springboot_api_url, files=files, data=data)
            print("SpringBoot 서버 응답:", response.json())
        except Exception as e:
            print("SpringBoot 서버 전송 오류:", e)
    else:
        print("욕설이 감지되지 않았습니다. 파일을 삭제합니다.")
        try:
            os.remove(audio_file_path)
            print("파일 삭제 완료.")
        except Exception as e:
            print("[ERROR] 파일 삭제 중 오류:", e)

class AudioFileHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return
        ext = os.path.splitext(event.src_path)[1].lower()
        if ext in [".mp3", ".wav", ".webm", ".m4a"]:
            print("새 음성 파일 감지:", event.src_path)
            process_audio_file(event.src_path)

if __name__ == "__main__":
    folder_to_watch = r"C:\meeple\meeple_audio"
    event_handler = AudioFileHandler()
    observer = Observer()
    observer.schedule(event_handler, folder_to_watch, recursive=False)
    observer.start()
    print("폴더 모니터링 시작:", folder_to_watch)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("모니터링 중지...")
        observer.stop()
    observer.join()
