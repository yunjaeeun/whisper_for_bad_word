Name "Meeple Installer"
OutFile "Meeple Setup 1.0.0.exe"
InstallDir "C:\meeple"                   
RequestExecutionLevel admin           

Page instfiles

Section "Install"

  CreateDirectory "C:\meeple"
  CreateDirectory "C:\meeple\meeple_audio"

  SetOutPath "$INSTDIR"
  
  File "C:\Users\User\AppData\Local\Programs\my-electron-app\MyElectronApp.exe"
  
  CreateDirectory "$INSTDIR\resources"
  File /r "C:\Users\User\AppData\Local\Programs\my-electron-app\*.*"
  
  CreateDirectory "$INSTDIR\models"
  File /oname=models\badword_model.pkl "C:\Users\User\Desktop\electron_test\python_model\badword_model.pkl"
  File /oname=models\vectorizer.pkl "C:\Users\User\Desktop\electron_test\python_model\vectorizer.pkl"
  
  CreateShortCut "$DESKTOP\MyElectronApp.lnk" "$INSTDIR\MyElectronApp.exe"
  
SectionEnd


Section "Update PATH"
  ReadRegStr $0 HKLM "SYSTEM\CurrentControlSet\Control\Session Manager\Environment" "Path"
  StrCpy $1 "$INSTDIR"
  StrCpy $0 "$0;$1"
  WriteRegExpandStr HKLM "SYSTEM\CurrentControlSet\Control\Session Manager\Environment" "Path" "$0"
  SendMessage ${HWND_BROADCAST} ${WM_SETTINGCHANGE} 0 "STR:Environment" /TIMEOUT=5000
SectionEnd
