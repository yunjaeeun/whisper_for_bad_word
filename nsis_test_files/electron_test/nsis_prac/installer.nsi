; 애플리케이션 정보
Name "MyElectronApp Installer"
OutFile "MyElectronApp Setup 1.0.0.exe"
InstallDir "C:\meeple"                   ; 고정 설치 경로: C:\meeple
RequestExecutionLevel admin              ; 관리자 권한 요구

; 설치 진행 화면 페이지 구성 (사용자가 경로를 변경할 수 없도록)
Page instfiles

Section "Install"

  ; 설치 전에 별도의 디렉토리 생성: C:\meeple 및 C:\meeple\meeple_audio
  CreateDirectory "C:\meeple"
  CreateDirectory "C:\meeple\meeple_audio"

  ; 실제 설치 경로 설정 (C:\meeple)
  SetOutPath "$INSTDIR"
  
  ; 1. Electron 앱 실행 파일 복사  
  ;    (아래 경로는 Electron 빌드 결과물의 실제 경로로 수정)
  File "C:\Users\User\AppData\Local\Programs\my-electron-app\MyElectronApp.exe"
  
  ; 2. 필요한 추가 리소스 복사 (예: resources 폴더 등)
  CreateDirectory "$INSTDIR\resources"
  File /r "C:\Users\User\AppData\Local\Programs\my-electron-app\*.*"
  
  ; 3. Python 모델 파일 복사  
  ;    모델 파일들을 설치 경로 내의 models 폴더에 복사
  CreateDirectory "$INSTDIR\models"
  File /oname=models\badword_model.pkl "C:\Users\User\Desktop\electron_test\python_model\badword_model.pkl"
  File /oname=models\vectorizer.pkl "C:\Users\User\Desktop\electron_test\python_model\vectorizer.pkl"
  
  ; 4. 바탕화면에 바로가기 생성  
  CreateShortCut "$DESKTOP\MyElectronApp.lnk" "$INSTDIR\MyElectronApp.exe"
  
SectionEnd

;--------------------------------
; (선택 사항) PATH 환경 변수 업데이트
;--------------------------------
Section "Update PATH"
  ReadRegStr $0 HKLM "SYSTEM\CurrentControlSet\Control\Session Manager\Environment" "Path"
  StrCpy $1 "$INSTDIR"
  StrCpy $0 "$0;$1"
  WriteRegExpandStr HKLM "SYSTEM\CurrentControlSet\Control\Session Manager\Environment" "Path" "$0"
  SendMessage ${HWND_BROADCAST} ${WM_SETTINGCHANGE} 0 "STR:Environment" /TIMEOUT=5000
SectionEnd
