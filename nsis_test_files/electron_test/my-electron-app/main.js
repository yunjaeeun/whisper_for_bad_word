const { app, BrowserWindow } = require('electron');
const { spawn } = require('child_process');
const path = require('path');

let mainWindow;

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 800,
    height: 600,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false,
      sandbox: false
    }
  });

  mainWindow.loadFile('index.html');

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

app.whenReady().then(() => {
  createWindow();

  // 실행할 Python 파일 경로 설정
  const pythonExePath = path.resolve("C:\\meeple", "find_badword.exe");

  console.log(`🔍 실행할 Python 파일: ${pythonExePath}`);

  function runPythonProcess() {
    // 새로운 cmd 창에서 실행하도록 `start` 명령 사용
    const pythonProcess = spawn('cmd.exe', ['/c', 'start', 'cmd.exe', '/k', pythonExePath], {
      shell: true
    });

    pythonProcess.stdout.on('data', (data) => {
      console.log(`🐍 Python stdout: ${data.toString().trim()}`);
    });

    pythonProcess.stderr.on('data', (data) => {
      console.error(`❌ Python stderr: ${data.toString().trim()}`);
    });

    pythonProcess.on('error', (err) => {
      console.error(`⚠️ 실행 실패: ${err.message}`);
    });

    pythonProcess.on('close', (code) => {
      console.log(`✅ Python 프로세스 종료 (코드: ${code})`);
    });
  }

  // 앱이 준비된 후 Python 실행
  setTimeout(runPythonProcess, 1000);
});

// macOS에서는 창이 모두 닫혀도 앱이 종료되지 않도록 설정
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
