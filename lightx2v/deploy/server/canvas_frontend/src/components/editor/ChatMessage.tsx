import React, { useMemo, useState, useEffect } from 'react';
import { Bot, User, CheckCircle2, XCircle, ChevronDown, ChevronUp } from 'lucide-react';
import { Language } from '../../i18n/useTranslation';

interface ChatMessageProps {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: number;
  operations?: any[];
  operationResults?: any[];
  error?: string;
  lang: Language;
  onUndo?: () => void;
  onRetry?: () => void;
  thinking?: string;
  isStreaming?: boolean;
}

export const ChatMessage: React.FC<ChatMessageProps> = ({
  role,
  content,
  timestamp,
  operations,
  operationResults,
  error,
  lang,
  onUndo,
  onRetry,
  thinking,
  isStreaming
}) => {
  const isUser = role === 'user';
  const hasOperations = operations && operations.length > 0;
  const hasError = !!error;
  const allSuccess = operationResults?.every(r => r.success) ?? false;
  const hasThinking = !!thinking && thinking.trim().length > 0;
  
  // 当有最终答案时，自动折叠思考过程
  const [isThinkingExpanded, setIsThinkingExpanded] = useState(false);
  
  useEffect(() => {
    // 如果有思考过程且有最终答案（不是流式输出中），默认折叠
    if (hasThinking && !isStreaming && content.trim().length > 0) {
      setIsThinkingExpanded(false);
    } else if (hasThinking && isStreaming) {
      // 流式输出中，展开思考过程
      setIsThinkingExpanded(true);
    }
  }, [hasThinking, isStreaming, content]);

  // 统计操作类型和数量
  const operationSummary = useMemo(() => {
    if (!operations || operations.length === 0) return null;

    const summary: Record<string, number> = {};
    operations.forEach(op => {
      const type = op.type;
      if (type === 'add_node') {
        summary['add_node'] = (summary['add_node'] || 0) + 1;
      } else if (type === 'delete_node') {
        summary['delete_node'] = (summary['delete_node'] || 0) + 1;
      } else if (type === 'add_connection') {
        summary['add_connection'] = (summary['add_connection'] || 0) + 1;
      } else if (type === 'delete_connection') {
        summary['delete_connection'] = (summary['delete_connection'] || 0) + 1;
      } else if (type === 'update_node' || type === 'update_node_data') {
        summary['update_node'] = (summary['update_node'] || 0) + 1;
      } else if (type === 'replace_node') {
        summary['replace_node'] = (summary['replace_node'] || 0) + 1;
      } else if (type === 'move_node') {
        summary['move_node'] = (summary['move_node'] || 0) + 1;
      }
    });

    const summaryTexts: string[] = [];
    if (summary['add_node']) {
      summaryTexts.push(lang === 'zh' ? `添加${summary['add_node']}个节点` : `Add ${summary['add_node']} node${summary['add_node'] > 1 ? 's' : ''}`);
    }
    if (summary['delete_node']) {
      summaryTexts.push(lang === 'zh' ? `删除${summary['delete_node']}个节点` : `Delete ${summary['delete_node']} node${summary['delete_node'] > 1 ? 's' : ''}`);
    }
    if (summary['add_connection']) {
      summaryTexts.push(lang === 'zh' ? `添加${summary['add_connection']}个连接` : `Add ${summary['add_connection']} connection${summary['add_connection'] > 1 ? 's' : ''}`);
    }
    if (summary['delete_connection']) {
      summaryTexts.push(lang === 'zh' ? `删除${summary['delete_connection']}个连接` : `Delete ${summary['delete_connection']} connection${summary['delete_connection'] > 1 ? 's' : ''}`);
    }
    if (summary['update_node']) {
      summaryTexts.push(lang === 'zh' ? `修改${summary['update_node']}个节点` : `Modify ${summary['update_node']} node${summary['update_node'] > 1 ? 's' : ''}`);
    }
    if (summary['replace_node']) {
      summaryTexts.push(lang === 'zh' ? `替换${summary['replace_node']}个节点` : `Replace ${summary['replace_node']} node${summary['replace_node'] > 1 ? 's' : ''}`);
    }
    if (summary['move_node']) {
      summaryTexts.push(lang === 'zh' ? `移动${summary['move_node']}个节点` : `Move ${summary['move_node']} node${summary['move_node'] > 1 ? 's' : ''}`);
    }

    return summaryTexts.length > 0 ? summaryTexts : null;
  }, [operations, lang]);

  return (
    <div className={`flex gap-3 mb-4 ${isUser ? 'flex-row-reverse' : ''}`}>
      {/* Avatar */}
      <div
        className={`w-8 h-8 rounded-full flex items-center justify-center shrink-0 ${
          isUser ? 'bg-[#90dce1]' : 'bg-[#90dce1]'
        }`}
      >
        {isUser ? (
          <User size={16} className="text-white" />
        ) : (
          <Bot size={16} className="text-white" />
        )}
      </div>

      {/* Message Content */}
      <div className={`flex-1 ${isUser ? 'items-end' : 'items-start'} flex flex-col`}>
        <div
          className={`rounded-2xl px-4 py-3 max-w-[80%] ${
            isUser
              ? 'bg-[#90dce1]/20 border border-[#90dce1]/30 text-slate-100'
              : 'bg-slate-800/50 border border-slate-700 text-slate-200'
          }`}
        >
          {/* 思考过程（可折叠） */}
          {hasThinking && !isUser && (
            <div className="mb-3">
              <button
                onClick={() => setIsThinkingExpanded(!isThinkingExpanded)}
                className="flex items-center gap-2 w-full text-left text-xs text-slate-400 hover:text-slate-300 transition-colors"
              >
                {isThinkingExpanded ? (
                  <ChevronUp size={14} />
                ) : (
                  <ChevronDown size={14} />
                )}
                <span>{lang === 'zh' ? '思考过程' : 'Thinking Process'}</span>
                {isStreaming && (
                  <span className="ml-2 text-[#90dce1] animate-pulse">
                    {lang === 'zh' ? '思考中...' : 'Thinking...'}
                  </span>
                )}
              </button>
              {isThinkingExpanded && (
                <div className="mt-2 p-3 bg-slate-900/50 rounded-lg border border-slate-700/50">
                  <p className="text-xs text-slate-400 whitespace-pre-wrap break-words font-mono">
                    {thinking}
                    {isStreaming && (
                      <span className="inline-block w-2 h-4 bg-[#90dce1] ml-1 animate-pulse" />
                    )}
                  </p>
                </div>
              )}
            </div>
          )}
          
          {/* 最终答案 */}
          <div>
            {content.trim().length > 0 ? (
              <p className="text-sm whitespace-pre-wrap break-words">
                {content}
                {isStreaming && (
                  <span className="inline-block w-2 h-4 bg-[#90dce1] ml-1 animate-pulse" />
                )}
              </p>
            ) : isStreaming ? (
              <p className="text-sm text-slate-400">
                {lang === 'zh' ? '正在生成...' : 'Generating...'}
                <span className="inline-block w-2 h-4 bg-[#90dce1] ml-1 animate-pulse" />
              </p>
            ) : null}
          </div>

          {/* Operation Status */}
          {hasOperations && (
            <div className="mt-2 pt-2 border-t border-slate-700/50 space-y-1">
              <div className="flex items-center gap-2 text-xs">
                {hasError ? (
                  <>
                    <XCircle size={12} className="text-red-400" />
                    <span className="text-red-400">
                      {lang === 'zh' ? '操作失败' : 'Operation failed'}
                    </span>
                  </>
                ) : allSuccess ? (
                  <>
                    <CheckCircle2 size={12} className="text-emerald-400" />
                    <span className="text-emerald-400">
                      {lang === 'zh' ? '操作成功' : 'Operation successful'}
                    </span>
                  </>
                ) : (
                  <span className="text-yellow-400">
                    {lang === 'zh' ? '部分操作成功' : 'Partially successful'}
                  </span>
                )}
              </div>
              {/* Operation Summary */}
              {operationSummary && operationSummary.length > 0 && (
                <div className="flex flex-col gap-0.5 text-[10px] text-slate-400">
                  {operationSummary.map((text, index) => (
                    <span key={index}>{text}</span>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>

        {/* Action Buttons (for assistant messages) */}
        {!isUser && hasOperations && (
          <div className="flex gap-2 mt-2">
            {!hasError && onUndo && (
              <button
                onClick={onUndo}
                className="px-2 py-1 text-xs bg-slate-700 hover:bg-slate-600 rounded-lg transition-colors"
              >
                {lang === 'zh' ? '撤销' : 'Undo'}
              </button>
            )}
            {hasError && onRetry && (
              <button
                onClick={onRetry}
                className="px-2 py-1 text-xs bg-slate-700 hover:bg-slate-600 rounded-lg transition-colors"
              >
                {lang === 'zh' ? '重试' : 'Retry'}
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  );
};
