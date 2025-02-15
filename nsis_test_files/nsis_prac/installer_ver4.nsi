Name "Meeple Installer"
OutFile "Meeple Setup 1.4.5.exe"
InstallDir "C:\meeple"
RequestExecutionLevel admin

Page instfiles


Section "Install"
  CreateDirectory "$INSTDIR"
  CreateDirectory "$INSTDIR\meeple_audio"
  
  SetOutPath "$INSTDIR"
  File "C:\Users\User\AppData\Local\Programs\meeple-app\MeepleApp.exe"
  
  CreateDirectory "$INSTDIR\resources"
  File /r "C:\Users\User\AppData\Local\Programs\meeple-app\*.*"
  
  CreateDirectory "$INSTDIR\models"
  File /oname=models\badword_model.pkl "C:\Users\User\Desktop\pjt\whisper_for_bad_word\llm_stt_test_files\bad_word_llm\test_with_whisper\badword_model.pkl"
  File /oname=models\vectorizer.pkl "C:\Users\User\Desktop\pjt\whisper_for_bad_word\llm_stt_test_files\bad_word_llm\test_with_whisper\vectorizer.pkl"
  
  CreateShortCut "$DESKTOP\MeepleApp.lnk" "$INSTDIR\MeepleApp.exe"
SectionEnd


Section "Install ffmpeg"
  CreateDirectory "$INSTDIR\ffmpeg"
  SetOutPath "$INSTDIR\ffmpeg"
  
  File /r "C:\env\ffmpeg\bin\*.*"
SectionEnd

Section "Update PATH"
  ReadRegStr $0 HKLM "SYSTEM\CurrentControlSet\Control\Session Manager\Environment" "Path"
  
  StrCpy $1 "$INSTDIR\ffmpeg"
  
  StrCpy $0 "$0;$1"
  WriteRegExpandStr HKLM "SYSTEM\CurrentControlSet\Control\Session Manager\Environment" "Path" "$0"
  
  SendMessage ${HWND_BROADCAST} ${WM_SETTINGCHANGE} 0 "STR:Environment" /TIMEOUT=5000
SectionEnd
