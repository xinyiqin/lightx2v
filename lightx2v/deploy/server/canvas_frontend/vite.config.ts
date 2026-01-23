import path from 'path';
import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';
import qiankun from 'vite-plugin-qiankun';

export default defineConfig(({ mode }) => {
    // 优先使用系统环境变量，如果没有则从 .env 文件读取
    const env = loadEnv(mode, '.', '');
    // 合并系统环境变量（如果存在）
    const mergedEnv = {
      ...env,
      // 如果系统环境变量存在，优先使用系统环境变量
      LIGHTX2V_URL: process.env.LIGHTX2V_URL || env.LIGHTX2V_URL || '',
      LIGHTX2V_TOKEN: process.env.LIGHTX2V_TOKEN || env.LIGHTX2V_TOKEN || '',
      GEMINI_API_KEY: process.env.GEMINI_API_KEY || env.GEMINI_API_KEY || '',
      DEEPSEEK_API_KEY: process.env.DEEPSEEK_API_KEY || env.DEEPSEEK_API_KEY || '',
      PPCHAT_API_KEY: process.env.PPCHAT_API_KEY || env.PPCHAT_API_KEY || '',
    };
    return {
      server: {
        port: 3000,
        host: '0.0.0.0',
        cors: true, // 允许跨域，支持微前端
        headers: {
          'Access-Control-Allow-Origin': '*',
        },
      },
      plugins: [
        react(),
        // qiankun 插件，自动处理微前端生命周期
        qiankun('react-canvas', {
          // 开发模式：true，生产模式：false
          useDevMode: mode === 'development'
        })
      ],
      define: {
        'process.env.API_KEY': JSON.stringify(mergedEnv.GEMINI_API_KEY),
        'process.env.GEMINI_API_KEY': JSON.stringify(mergedEnv.GEMINI_API_KEY),
        'process.env.LIGHTX2V_TOKEN': JSON.stringify(mergedEnv.LIGHTX2V_TOKEN || ''),
        // 优先使用系统环境变量，如果没有则从 .env 文件读取
        // 如果都不设置，则 process.env.LIGHTX2V_URL 为空字符串
        // App.tsx 会检测到空字符串并根据是否有 apiClient 决定使用相对路径或默认 URL
        'process.env.LIGHTX2V_URL': JSON.stringify(mergedEnv.LIGHTX2V_URL || ''),
        'process.env.DEEPSEEK_API_KEY': JSON.stringify(mergedEnv.DEEPSEEK_API_KEY || ''),
        'process.env.PPCHAT_API_KEY': JSON.stringify(mergedEnv.PPCHAT_API_KEY || '')
      },
      resolve: {
        alias: {
          '@': path.resolve(__dirname, '.'),
        }
      },
      optimizeDeps: {
        esbuildOptions: {
          // 忽略 source map 错误
          logOverride: { 'this-is-undefined-in-esm': 'silent' },
          // 跳过 source map 处理
          sourcemap: false
        }
      },
      build: {
        sourcemap: false,
        // 确保资源路径使用相对路径，支持微前端
        rollupOptions: {
          output: {
            entryFileNames: 'assets/[name].[hash].js',
            chunkFileNames: 'assets/[name].[hash].js',
            assetFileNames: 'assets/[name].[hash].[ext]',
          },
        },
      },
      // 构建时使用绝对路径 /canvas/，确保在 qiankun 环境中能正确加载
      base: mode === 'production' ? '/canvas/' : '/',
    };
});
