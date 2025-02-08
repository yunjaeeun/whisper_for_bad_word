// main.js
const { app, BrowserWindow } = require('electron');
const { spawn } = require('child_process');
const path = require('path');

function createWindow() {
  const win = new BrowserWindow({
    width: 800,
    height: 600,
    webPreferences: {
      nodeIntegration: true,    // 개발 중 편의를 위해 활성화 (production에서는 보안에 주의)
      contextIsolation: false,
      sandbox: false
    }
  });
  win.loadFile('index.html');
}

app.whenReady().then(() => {
    createWindow() 
    const pythonExePath = path.join("C:\\meeple", "find_badword.exe");
    

  const pythonProcess = spawn(pythonExePath, [], {shell: true});

  // 표준 출력 데이터 처리
  pythonProcess.stdout.on('data', (data) => {
    console.log(`Python stdout: ${data}`);
  });

  // 에러 출력 데이터 처리
  pythonProcess.stderr.on('data', (data) => {
    console.error(`Python stderr: ${data}`);
  });

  pythonProcess.on('close', (code) => {
    console.log(`Python 프로세스 종료 (종료 코드: ${code})`);
  });
});

// macOS에서는 창이 모두 닫혀도 앱이 종료되지 않도록 함
app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});
app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow();
  }
});
