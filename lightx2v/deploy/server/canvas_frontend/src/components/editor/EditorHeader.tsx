import React, { useState, useEffect } from 'react';
import {
  ChevronLeft,
  ZoomIn,
  ZoomOut,
  Maximize,
  Languages,
  BookOpen,
  Timer,
  X,
  Save,
  Play,
  Pause,
  RefreshCw,
  Zap,
  Undo,
  Redo,
  LayoutGrid,
  LogOut,
  LogIn
} from 'lucide-react';
import { WorkflowState } from '../../../types';
import { useTranslation, Language } from '../../i18n/useTranslation';
import { formatTime } from '../../utils/format';

interface ViewState {
  x: number;
  y: number;
  zoom: number;
}

interface EditorHeaderProps {
  lang: Language;
  workflow: WorkflowState;
  view: ViewState;
  selectedRunId: string | null;
  isPaused: boolean;
  isRunning: boolean;
  canvasRef: React.RefObject<HTMLDivElement>;
  onBack: () => void;
  onWorkflowNameChange: (name: string) => void;
  onZoomIn: () => void;
  onZoomOut: () => void;
  onResetView: () => void;
  onToggleLang: () => void;
  onClearSnapshot: () => void;
  onSave: () => void;
  onPause: () => void;
  onRun: () => void;
  canUndo: boolean;
  canRedo: boolean;
  onUndo: () => void;
  onRedo: () => void;
}

export const EditorHeader: React.FC<EditorHeaderProps> = ({
  lang,
  workflow,
  view,
  selectedRunId,
  isPaused,
  isRunning,
  canvasRef,
  onBack,
  onWorkflowNameChange,
  onZoomIn,
  onZoomOut,
  onResetView,
  onToggleLang,
  onClearSnapshot,
  onSave,
  onPause,
  onRun,
  canUndo,
  canRedo,
  onUndo,
  onRedo
}) => {
  const { t } = useTranslation(lang);
  const [user, setUser] = useState<any>(null);

  // 获取用户信息
  useEffect(() => {
    const sharedStore = (window as any).__SHARED_STORE__;
    
    // 初始化用户信息
    if (sharedStore) {
      const currentUser = sharedStore.getState('user');
      setUser(currentUser);
      
      // 订阅用户状态变化
      const unsubscribe = sharedStore.subscribe((state: any) => {
        setUser(state.user);
      });
      
      return () => {
        if (unsubscribe) unsubscribe();
      };
    } else {
      // 如果没有 sharedStore，从 localStorage 读取
      try {
        const userStr = localStorage.getItem('currentUser');
        if (userStr) {
          setUser(JSON.parse(userStr));
        }
      } catch (e) {
        console.error('Failed to parse user from localStorage:', e);
      }
    }
  }, []);

  // 切换到普通模式（主应用的 generate 页面）
  const handleSwitchToNormalMode = () => {
    // 在 qiankun 环境中，使用 window.location 跳转到主应用
    // 如果在 iframe 中，尝试使用 parent，否则使用当前窗口
    try {
      if (window.parent !== window) {
        // 在 iframe 中
        window.parent.location.href = '/generate';
      } else {
        // 直接跳转
        window.location.href = '/generate';
      }
    } catch (e) {
      // 跨域限制，使用当前窗口
      window.location.href = '/generate';
    }
  };

  // 处理登录
  const handleLogin = () => {
    // 跳转到主应用的登录页面
    try {
      if (window.parent !== window) {
        window.parent.location.href = '/generate';
      } else {
        window.location.href = '/generate';
      }
    } catch (e) {
      window.location.href = '/generate';
    }
  };

  // 处理登出
  const handleLogout = () => {
    const sharedStore = (window as any).__SHARED_STORE__;
    if (sharedStore) {
      sharedStore.clear();
    } else {
      localStorage.removeItem('accessToken');
      localStorage.removeItem('currentUser');
    }
    setUser(null);
    // 刷新页面
    window.location.reload();
  };

  return (
    <header className="h-16 border-b border-slate-800/60 flex items-center justify-between px-6 bg-slate-900/40 backdrop-blur-2xl z-40">
      <div className="flex items-center gap-5">
        <button
          onClick={onBack}
          className="p-2 text-slate-500 hover:text-white hover:bg-slate-800 rounded-xl transition-all"
          title={lang === 'zh' ? '返回工作流列表' : 'Back to Workflows'}
        >
          <ChevronLeft size={20} />
        </button>
        <div className="flex flex-col">
          <input
            value={workflow.name}
            onChange={(e) => {
              onClearSnapshot();
              onWorkflowNameChange(e.target.value);
            }}
            className="bg-transparent border-none text-base font-bold focus:ring-0 p-0 hover:bg-slate-800/20 rounded px-1 transition-colors w-64"
          />
          <span className="text-[9px] text-slate-500 font-bold uppercase tracking-widest">
            {t('editing_logic')}
          </span>
        </div>
      </div>
      <div className="flex items-center gap-4">
        {/* Undo/Redo Controls */}
        <div className="flex items-center gap-1 bg-slate-800/50 border border-slate-800 rounded-xl p-1">
          <button
            onClick={onUndo}
            disabled={!canUndo}
            className="p-1.5 text-slate-400 hover:text-white hover:bg-slate-700 rounded-lg transition-all disabled:opacity-30 disabled:cursor-not-allowed"
            title={lang === 'zh' ? '撤销 (Ctrl+Z)' : 'Undo (Ctrl+Z)'}
          >
            <Undo size={14} />
          </button>
          <button
            onClick={onRedo}
            disabled={!canRedo}
            className="p-1.5 text-slate-400 hover:text-white hover:bg-slate-700 rounded-lg transition-all disabled:opacity-30 disabled:cursor-not-allowed"
            title={lang === 'zh' ? '重做 (Ctrl+Y)' : 'Redo (Ctrl+Y)'}
          >
            <Redo size={14} />
          </button>
        </div>
        
        {/* Zoom Controls */}
        <div className="flex items-center gap-1 bg-slate-800/50 border border-slate-800 rounded-xl p-1">
          <button
            onClick={onZoomIn}
            className="p-1.5 text-slate-400 hover:text-white hover:bg-slate-700 rounded-lg transition-all"
            title={lang === 'zh' ? '放大' : 'Zoom In'}
          >
            <ZoomIn size={14} />
          </button>
          <div className="px-2 py-1 text-[10px] font-bold text-slate-500 min-w-[3rem] text-center">
            {Math.round(view.zoom * 100)}%
          </div>
          <button
            onClick={onZoomOut}
            className="p-1.5 text-slate-400 hover:text-white hover:bg-slate-700 rounded-lg transition-all"
            title={lang === 'zh' ? '缩小' : 'Zoom Out'}
          >
            <ZoomOut size={14} />
          </button>
          <button
            onClick={onResetView}
            className="p-1.5 text-slate-400 hover:text-white hover:bg-slate-700 rounded-lg transition-all"
            title={lang === 'zh' ? '重置视图' : 'Reset View'}
          >
            <Maximize size={14} />
          </button>
        </div>

        <button
          onClick={onToggleLang}
          className="flex items-center gap-2 px-3 py-1.5 bg-slate-800/50 hover:bg-slate-800 text-slate-400 rounded-xl text-[10px] font-bold transition-all border border-slate-800"
        >
          <Languages size={12} /> {t('lang_name')}
        </button>
        {selectedRunId && (
          <>
            <div className="flex items-center gap-2 px-4 py-2 bg-[#90dce1]/20 rounded-xl border border-[#90dce1]/30 animate-pulse">
              <BookOpen size={14} className="text-[#90dce1]" />
              <span className="text-[10px] font-black uppercase text-[#90dce1]">
                {t('snapshot_view')}
              </span>
              <button onClick={onClearSnapshot} className="ml-2 hover:text-white">
                <X size={12} />
              </button>
            </div>
            <div className="flex items-center gap-2 px-4 py-2 bg-slate-800/50 rounded-xl border border-slate-800">
              <Timer size={14} className="text-[#90dce1]" />
              <span className="text-[10px] font-black uppercase text-slate-300">
                {t('run_time')}:{' '}
                {formatTime(
                  workflow.history.find((r) => r.id === selectedRunId)?.totalTime
                )}
              </span>
            </div>
          </>
        )}
        <button
          onClick={onSave}
          className={`flex items-center gap-2 px-6 py-2 rounded-xl text-sm font-bold border transition-all ${
            workflow.isDirty
              ? 'bg-[#90dce1]/10 border-[#90dce1] text-[#90dce1] hover:bg-[#90dce1]/20'
              : 'bg-slate-800 border-slate-700 text-slate-500'
          }`}
        >
          <Save size={16} /> {t('save_flow')}
        </button>
        <div className="w-px h-6 bg-slate-800"></div>
        {isRunning && (
          <button
            onClick={onPause}
            className={`flex items-center gap-2 px-6 py-2.5 rounded-2xl text-sm font-bold shadow-xl transition-all ${
              isPaused
                ? 'bg-yellow-600 hover:bg-yellow-500 text-white shadow-yellow-500/20'
                : 'bg-orange-600 hover:bg-orange-500 text-white shadow-orange-500/20'
            } active:scale-95`}
          >
            {isPaused ? <Play size={16} /> : <Pause size={16} />}
            {isPaused ? (lang === 'zh' ? '继续' : 'Resume') : (lang === 'zh' ? '暂停' : 'Pause')}
          </button>
        )}
        <button
          onClick={onRun}
          disabled={isRunning && !isPaused}
          className={`flex items-center gap-2 px-6 py-2.5 rounded-2xl text-sm font-bold shadow-xl transition-all ${
            isRunning && !isPaused
              ? 'bg-slate-800 text-slate-500'
              : 'bg-[#90dce1] hover:bg-[#7dd3da] text-white shadow-[#90dce1]/20 active:scale-95'
          }`}
        >
          {isRunning && !isPaused ? (
            <RefreshCw className="animate-spin" size={16} />
          ) : (
            <Zap size={16} />
          )}
          {isRunning && !isPaused ? t('executing') : t('run_fabric')}
        </button>

        {/* 用户信息卡片 */}
        <div className="flex items-center gap-2 px-3 py-1.5 bg-slate-800/50 hover:bg-slate-800 border border-slate-700 rounded-[20px] transition-all duration-200">
          {/* 用户头像 */}
          <div className="flex items-center justify-center w-7 h-7 flex-shrink-0 rounded-full overflow-hidden bg-slate-700 border border-slate-600">
            {user?.avatar_url ? (
              <img
                src={user.avatar_url}
                alt={user.username || 'User'}
                className="w-full h-full object-cover"
              />
            ) : (
              <div className="w-full h-full flex items-center justify-center bg-gradient-to-br from-slate-600/50 to-slate-700/50">
                <span className="text-[10px] text-slate-300 font-medium">
                  {user?.username?.[0]?.toUpperCase() || user?.email?.[0]?.toUpperCase() || '?'}
                </span>
              </div>
            )}
          </div>

          {/* 用户名 */}
          <div className="text-xs font-medium text-slate-300 tracking-[-0.01em] whitespace-nowrap overflow-hidden text-ellipsis max-w-[120px]">
            {user?.username || user?.email || (lang === 'zh' ? '未登录' : 'Not logged in')}
          </div>

          {/* 登录/登出按钮 */}
          {user?.username || user?.email ? (
            <button
              onClick={handleLogout}
              className="flex items-center justify-center w-6 h-6 p-0 bg-transparent border-0 rounded-full cursor-pointer transition-all duration-200 hover:bg-red-500/10 hover:scale-110 active:scale-100 flex-shrink-0 group"
              title={lang === 'zh' ? '登出' : 'Logout'}
            >
              <LogOut size={12} className="text-slate-400 group-hover:text-red-400 transition-colors" />
            </button>
          ) : (
            <button
              onClick={handleLogin}
              className="flex items-center justify-center w-6 h-6 p-0 bg-transparent border-0 rounded-full cursor-pointer transition-all duration-200 hover:bg-[#90dce1]/10 hover:scale-110 active:scale-100 flex-shrink-0 group"
              title={lang === 'zh' ? '登录' : 'Login'}
            >
              <LogIn size={12} className="text-slate-400 group-hover:text-[#90dce1] transition-colors" />
            </button>
          )}
        </div>
      </div>
    </header>
  );
};

