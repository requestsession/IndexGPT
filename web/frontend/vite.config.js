import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { execSync } from 'child_process';
import path from 'path';

// 动态从 Python 的 config.py 中提取端口号
let backendPort = 8176; // 默认值
try {
  // 执行一段 python 代码直接打印出 BACKEND_PORT
  // 需要先将根目录添加到 PYTHONPATH 以便导入 config
  const pythonCmd = 'python -c "import sys; sys.path.append(\'../../\'); from config import BACKEND_PORT; print(BACKEND_PORT)"';
  const output = execSync(pythonCmd).toString().trim();
  if (output && !isNaN(parseInt(output))) {
    backendPort = parseInt(output, 10);
  }
} catch (e) {
  console.warn("Could not read BACKEND_PORT from config.py, using default 8176");
}

export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',
    port: 5173,
    proxy: {
      '/api': {
        target: `http://127.0.0.1:${backendPort}`,
        changeOrigin: true,
      },
    },
  },
});
