Name "Meeple Installer"
OutFile "Meeple Setup 1.3.0.exe"
InstallDir "C:\meeple"
RequestExecutionLevel admin

Page instfiles


Section "Install"
  CreateDirectory "$INSTDIR"
  CreateDirectory "$INSTDIR\meeple_audio"
  
  SetOutPath "$INSTDIR"
  File "C:\Users\SSAFY\AppData\Local\Programs\meeple-app\MeepleApp.exe"
  
  CreateDirectory "$INSTDIR\resources"
  File /r "C:\Users\SSAFY\AppData\Local\Programs\meeple-app\*.*"
  
  CreateDirectory "$INSTDIR\models"
  File /oname=models\badword_model.pkl "C:\Users\SSAFY\Desktop\whisper_for_bad_word\nsis_test_files\python_model\badword_model.pkl"
  File /oname=models\vectorizer.pkl "C:\Users\SSAFY\Desktop\whisper_for_bad_word\nsis_test_files\python_model\vectorizer.pkl"
  
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
