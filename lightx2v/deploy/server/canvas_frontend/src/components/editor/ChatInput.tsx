import React, { useState, KeyboardEvent } from 'react';
import { Send } from 'lucide-react';
import { Language } from '../../i18n/useTranslation';

interface ChatInputProps {
  onSend: (message: string) => void;
  isProcessing: boolean;
  lang: Language;
}

export const ChatInput: React.FC<ChatInputProps> = ({
  onSend,
  isProcessing,
  lang
}) => {
  const [input, setInput] = useState('');

  const handleSend = () => {
    if (input.trim() && !isProcessing) {
      onSend(input.trim());
      setInput('');
    }
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="border-t border-slate-800 p-4 bg-slate-900/40">
      <div className="flex gap-2">
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={lang === 'zh' ? '描述你想要对工作流做的修改...' : 'Describe the changes you want to make to the workflow...'}
          className="flex-1 bg-slate-800/50 border border-slate-700 rounded-xl px-4 py-3 text-sm text-slate-200 placeholder-slate-500 resize-none focus:outline-none focus:ring-2 focus:ring-#90dce1 focus:border-transparent"
          rows={2}
          disabled={isProcessing}
        />
        <button
          onClick={handleSend}
          disabled={!input.trim() || isProcessing}
          className={`px-4 py-3 rounded-xl transition-all ${
            !input.trim() || isProcessing
              ? 'bg-slate-700 text-slate-500 cursor-not-allowed'
              : 'bg-[#90dce1] hover:bg-[#90dce1] text-white hover:scale-105'
          }`}
        >
          <Send size={18} />
        </button>
      </div>
      {isProcessing && (
        <div className="mt-2 text-xs text-slate-500 flex items-center gap-2">
          <div className="flex gap-1">
            <div className="w-1 h-1 bg-[#90dce1] rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
            <div className="w-1 h-1 bg-[#90dce1] rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
            <div className="w-1 h-1 bg-[#90dce1] rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
          </div>
          <span>{lang === 'zh' ? '正在处理...' : 'Processing...'}</span>
        </div>
      )}
    </div>
  );
};

