import React, { useState, useEffect } from 'react';
import { Wand2, Languages, Sparkle, Plus, LayoutGrid, LogOut, LogIn } from 'lucide-react';
import { useTranslation, Language } from '../../i18n/useTranslation';

interface HeaderProps {
  lang: Language;
  onToggleLang: () => void;
  onCreateWorkflow: () => void;
  onAIGenerate: () => void;
}

export const Header: React.FC<HeaderProps> = ({
  lang,
  onToggleLang,
  onCreateWorkflow,
  onAIGenerate
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
    // 跳转到主应用的登录页面，然后返回当前页面
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
    // 刷新页面或跳转到登录页
    window.location.reload();
  };

  return (
    <header className="h-20 border-b border-slate-800/60 flex items-center justify-between px-10 bg-slate-900/40 backdrop-blur-3xl z-40">
      <div className="flex items-center gap-4">
        <div className="w-12 h-12 bg-[#90dce1] rounded-2xl flex items-center justify-center shadow-xl shadow-[#90dce1]/20">
          <Wand2 className="text-white" size={24} />
        </div>
        <div className="flex flex-col">
          <h1 className="text-xl font-black uppercase tracking-widest text-white">
            {t('app_name')}
          </h1>
          <span className="text-[10px] text-slate-500 font-bold uppercase tracking-widest">
            {t('app_subtitle')}
          </span>
        </div>
        <button
          onClick={handleSwitchToNormalMode}
          className="flex items-center gap-3 ml-4 cursor-pointer transition-opacity duration-200 hover:opacity-80"
          title={lang === 'zh' ? '切换到普通模式' : 'Switch to Normal Mode'}
        >
          {/* 切换开关 */}
          <div className="relative w-12 h-6 bg-slate-600 dark:bg-slate-700 rounded-full transition-colors duration-200">
            <div className="absolute top-1 left-1 w-4 h-4 bg-white rounded-full shadow-md transition-transform duration-200 transform translate-x-0"></div>
            {/* 开关内的图标 */}
            <div className="absolute top-1 left-1 w-4 h-4 flex items-center justify-center pointer-events-none">
              <svg className="w-2.5 h-2.5 text-slate-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M8 7h12m0 0l-4-4m4 4l-4 4m0 6H4m0 0l4 4m-4-4l4-4" />
              </svg>
            </div>
          </div>
          {/* 文字标签 */}
          <span className="text-sm font-medium text-white tracking-tight whitespace-nowrap">
            {t('normal_mode')}
          </span>
        </button>
      </div>
      <div className="flex items-center gap-6">
        <button
          onClick={onToggleLang}
          className="flex items-center gap-2 px-4 py-2 bg-slate-800/50 hover:bg-slate-800 text-slate-300 rounded-xl text-xs font-bold transition-all border border-slate-800"
        >
          <Languages size={14} /> {t('lang_name')}
        </button>
        <button
          onClick={onAIGenerate}
          className="flex items-center gap-2 px-8 py-3 bg-[#90dce1] hover:bg-[#7dd3da] text-white rounded-2xl font-black text-sm transition-all active:scale-95"
        >
          <Sparkle size={18} /> {t('ai_generate_workflow')}
        </button>
        <button
          onClick={onCreateWorkflow}
          className="flex items-center gap-2 px-8 py-3 bg-[#90dce1] hover:bg-[#7dd3da] text-white rounded-2xl font-black text-sm transition-all active:scale-95"
        >
          <Plus size={18} /> {t('create_workflow')}
        </button>

        {/* 用户信息卡片 */}
        <div className="flex items-center gap-2.5 px-3 py-1.5 bg-slate-800/50 hover:bg-slate-800 border border-slate-700 rounded-[20px] transition-all duration-200">
          {/* 用户头像 */}
          <div className="flex items-center justify-center w-8 h-8 flex-shrink-0 rounded-full overflow-hidden bg-slate-700 border border-slate-600">
            {user?.avatar_url ? (
              <img
                src={user.avatar_url}
                alt={user.username || 'User'}
                className="w-full h-full object-cover"
              />
            ) : (
              <div className="w-full h-full flex items-center justify-center bg-gradient-to-br from-slate-600/50 to-slate-700/50">
                <span className="text-xs text-slate-300 font-medium">
                  {user?.username?.[0]?.toUpperCase() || user?.email?.[0]?.toUpperCase() || '?'}
                </span>
              </div>
            )}
          </div>

          {/* 用户名 */}
          <div className="text-sm font-medium text-slate-300 tracking-[-0.01em] whitespace-nowrap overflow-hidden text-ellipsis max-w-[150px]">
            {user?.username || user?.email || (lang === 'zh' ? '未登录' : 'Not logged in')}
          </div>

          {/* 登录/登出按钮 */}
          {user?.username || user?.email ? (
            <button
              onClick={handleLogout}
              className="flex items-center justify-center w-7 h-7 p-0 bg-transparent border-0 rounded-full cursor-pointer transition-all duration-200 hover:bg-red-500/10 hover:scale-110 active:scale-100 flex-shrink-0 group"
              title={lang === 'zh' ? '登出' : 'Logout'}
            >
              <LogOut size={14} className="text-slate-400 group-hover:text-red-400 transition-colors" />
            </button>
          ) : (
            <button
              onClick={handleLogin}
              className="flex items-center justify-center w-7 h-7 p-0 bg-transparent border-0 rounded-full cursor-pointer transition-all duration-200 hover:bg-[#90dce1]/10 hover:scale-110 active:scale-100 flex-shrink-0 group"
              title={lang === 'zh' ? '登录' : 'Login'}
            >
              <LogIn size={14} className="text-slate-400 group-hover:text-[#90dce1] transition-colors" />
            </button>
          )}
        </div>
      </div>
    </header>
  );
};
