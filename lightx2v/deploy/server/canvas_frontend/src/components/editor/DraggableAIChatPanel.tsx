import React, { useRef, useEffect, useCallback, useState } from 'react';
import { X, Minimize2, Bot, GripVertical } from 'lucide-react';
import { ChatMessage } from './ChatMessage';
import { ChatInput } from './ChatInput';
import { Language } from '../../i18n/useTranslation';
import { ChatMessage as ChatMessageType } from '../../hooks/useAIChatWorkflow';
import { TOOLS } from '../../../constants';

interface DraggableAIChatPanelProps {
  isOpen: boolean;
  onClose: () => void;
  chatHistory: ChatMessageType[];
  isProcessing: boolean;
  onSendMessage: (message: string) => void;
  onUndo?: (messageId: string) => void;
  onRetry?: (messageId: string) => void;
  lang: Language;
  aiModel?: string;
  onModelChange?: (model: string) => void;
  position?: { x: number; y: number };
  size?: { width: number; height: number };
  onPositionChange?: (position: { x: number; y: number }) => void;
  onSizeChange?: (size: { width: number; height: number }) => void;
}

const DEFAULT_WIDTH = 400;
const DEFAULT_HEIGHT = 500;
const MIN_WIDTH = 300;
const MIN_HEIGHT = 300;

export const DraggableAIChatPanel: React.FC<DraggableAIChatPanelProps> = ({
  isOpen,
  onClose,
  chatHistory,
  isProcessing,
  onSendMessage,
  onUndo,
  onRetry,
  lang,
  aiModel = 'deepseek-v3-2-251201',
  onModelChange,
  position: initialPosition,
  size: initialSize,
  onPositionChange,
  onSizeChange
}) => {
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const panelRef = useRef<HTMLDivElement>(null);
  const headerRef = useRef<HTMLDivElement>(null);

  // 位置和大小状态
  const [position, setPosition] = useState<{ x: number; y: number }>(() => {
    if (initialPosition) return initialPosition;
    // 初始位置：右下角按钮的左边（按钮在 right-6 bottom-6，即距离右边和底部24px，按钮宽度56px）
    // 对话框应该在按钮左边，距离右边 24 + 56 + 24 = 104px
    // 使用视口单位（vw 和 vh）
    if (typeof window !== 'undefined') {
      const viewportWidth = window.innerWidth * 1.25; // 100vw 的像素值
      const viewportHeight = window.innerHeight * 1.25; // 100vh 的像素值
      return {
        x: viewportWidth - DEFAULT_WIDTH - 104,
        y: viewportHeight - DEFAULT_HEIGHT - 24
      };
    }
    return { x: 0, y: 0 };
  });

  const [size, setSize] = useState<{ width: number; height: number }>(() => {
    if (initialSize) return initialSize;
    return { width: DEFAULT_WIDTH, height: DEFAULT_HEIGHT };
  });

  // 拖拽状态
  const isDraggingRef = useRef(false);
  const isResizingRef = useRef(false);
  const dragStartRef = useRef({ x: 0, y: 0 });
  const resizeStartRef = useRef({ x: 0, y: 0, width: 0, height: 0 });
  const resizeDirectionRef = useRef<'n' | 's' | 'e' | 'w' | 'ne' | 'nw' | 'se' | 'sw' | null>(null);

  // 自动滚动到底部（只滚动容器内部，不影响页面）
  useEffect(() => {
    if (messagesEndRef.current && scrollContainerRef.current) {
      // 使用 requestAnimationFrame 确保 DOM 更新完成后再滚动
      requestAnimationFrame(() => {
        if (scrollContainerRef.current && messagesEndRef.current) {
          // 直接设置 scrollTop，避免 scrollIntoView 导致页面整体滚动
          scrollContainerRef.current.scrollTop = scrollContainerRef.current.scrollHeight;
        }
      });
    }
  }, [chatHistory, isProcessing]);

  // 更新位置和大小到父组件
  useEffect(() => {
    if (onPositionChange) {
      onPositionChange(position);
    }
  }, [position, onPositionChange]);

  useEffect(() => {
    if (onSizeChange) {
      onSizeChange(size);
    }
  }, [size, onSizeChange]);

  // 处理窗口拖拽
  const handleHeaderMouseDown = useCallback((e: React.MouseEvent) => {
    if ((e.target as HTMLElement).closest('button, select')) return;
    e.preventDefault();
    isDraggingRef.current = true;
    dragStartRef.current = {
      x: e.clientX - position.x,
      y: e.clientY - position.y
    };
    document.body.style.cursor = 'move';
    document.body.style.userSelect = 'none';
  }, [position]);

  const handleMouseMove = useCallback((e: MouseEvent) => {
    if (isDraggingRef.current) {
      const newX = e.clientX - dragStartRef.current.x;
      const newY = e.clientY - dragStartRef.current.y;

      // 限制在视口内（使用视口单位 vw 和 vh）
      const viewportWidth = window.innerWidth * 1.25; // 100vw 的像素值
      const viewportHeight = window.innerHeight * 1.25; // 100vh 的像素值
      const maxX = viewportWidth - size.width;
      const maxY = viewportHeight - size.height;

      setPosition({
        x: Math.max(0, Math.min(newX, maxX)),
        y: Math.max(0, Math.min(newY, maxY))
      });
    } else if (isResizingRef.current && resizeDirectionRef.current) {
      const direction = resizeDirectionRef.current;
      const deltaX = e.clientX - resizeStartRef.current.x;
      const deltaY = e.clientY - resizeStartRef.current.y;

      let newWidth = resizeStartRef.current.width;
      let newHeight = resizeStartRef.current.height;
      let newX = position.x;
      let newY = position.y;

      if (direction.includes('e')) {
        newWidth = Math.max(MIN_WIDTH, resizeStartRef.current.width + deltaX);
      }
      if (direction.includes('w')) {
        newWidth = Math.max(MIN_WIDTH, resizeStartRef.current.width - deltaX);
        newX = position.x + (resizeStartRef.current.width - newWidth);
      }
      if (direction.includes('s')) {
        newHeight = Math.max(MIN_HEIGHT, resizeStartRef.current.height + deltaY);
      }
      if (direction.includes('n')) {
        newHeight = Math.max(MIN_HEIGHT, resizeStartRef.current.height - deltaY);
        newY = position.y + (resizeStartRef.current.height - newHeight);
      }

      // 限制在视口内（使用视口单位 vw 和 vh）
      const viewportWidth = window.innerWidth * 1.25; // 100vw 的像素值
      const viewportHeight = window.innerHeight * 1.25; // 100vh 的像素值
      const maxX = viewportWidth - newWidth;
      const maxY = viewportHeight - newHeight;

      const clampedX = Math.max(0, Math.min(newX, maxX));
      const clampedY = Math.max(0, Math.min(newY, maxY));

      setSize({ width: newWidth, height: newHeight });
      setPosition({ x: clampedX, y: clampedY });
    }
  }, [position, size]);

  const handleMouseUp = useCallback(() => {
    isDraggingRef.current = false;
    isResizingRef.current = false;
    resizeDirectionRef.current = null;
    document.body.style.cursor = '';
    document.body.style.userSelect = '';
  }, []);

  useEffect(() => {
    // 始终添加事件监听器，在handleMouseMove中检查isDraggingRef和isResizingRef
    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };
  }, [handleMouseMove, handleMouseUp]);

  // 处理调整大小
  const handleResizeMouseDown = useCallback((direction: 'n' | 's' | 'e' | 'w' | 'ne' | 'nw' | 'se' | 'sw') => (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    isResizingRef.current = true;
    resizeDirectionRef.current = direction;
    resizeStartRef.current = {
      x: e.clientX,
      y: e.clientY,
      width: size.width,
      height: size.height
    };
    document.body.style.userSelect = 'none';
  }, [size]);

  if (!isOpen) return null;

  return (
    <div
      ref={panelRef}
      className="fixed z-50 flex flex-col bg-slate-900/95 backdrop-blur-xl border border-slate-700/60 rounded-xl shadow-2xl overflow-hidden"
      style={{
        left: `${position.x}px`,
        top: `${position.y}px`,
        width: `${size.width}px`,
        height: `${size.height}px`,
        minWidth: `${MIN_WIDTH}px`,
        minHeight: `${MIN_HEIGHT}px`
      }}
    >
      {/* Header - 可拖拽 */}
      <div
        ref={headerRef}
        onMouseDown={handleHeaderMouseDown}
        className="flex items-center justify-between px-4 py-3 border-b border-slate-800/60 bg-slate-900/60 cursor-move select-none"
      >
        <div className="flex items-center gap-2 flex-1 min-w-0">
          <GripVertical size={14} className="text-slate-400 shrink-0" />
          <div className="w-6 h-6 rounded-full bg-[#90dce1] flex items-center justify-center shrink-0">
            <Bot size={14} className="text-white" />
          </div>
          <span className="text-sm font-bold text-slate-200 shrink-0">
            {lang === 'zh' ? 'AI助手' : 'AI Assistant'}
          </span>
          {/* 模型选择器 */}
          {onModelChange && (() => {
            const geminiTextTool = TOOLS.find(t => t.id === 'text-generation');
            const availableModels = geminiTextTool?.models || [];

            return (
              <select
                value={aiModel}
                onChange={(e) => onModelChange(e.target.value)}
                className="ml-2 px-3 py-1.5 text-xs bg-slate-800/80 hover:bg-slate-800 border border-slate-700/60 hover:border-slate-600 rounded-lg text-slate-200 focus:outline-none focus:ring-2 focus:ring-#90dce1/50 focus:border-[#90dce1] transition-all cursor-pointer appearance-none bg-[url('data:image/svg+xml;charset=utf-8,%3Csvg xmlns=%27http://www.w3.org/2000/svg%27 fill=%27none%27 viewBox=%270 0 20 20%27%3E%3Cpath stroke=%27%23cbd5e1%27 stroke-linecap=%27round%27 stroke-linejoin=%27round%27 stroke-width=%271.5%27 d=%27M6 8l4 4 4-4%27/%3E%3C/svg%3E')] bg-[length:16px_16px] bg-[right_8px_center] bg-no-repeat pr-8 shadow-sm"
                disabled={isProcessing}
                title={lang === 'zh' ? '选择AI模型' : 'Select AI Model'}
                onClick={(e) => e.stopPropagation()}
                onMouseDown={(e) => e.stopPropagation()}
              >
                {availableModels.length > 0 ? (
                  availableModels.map(model => (
                    <option key={model.id} value={model.id}>
                      {model.name}
                    </option>
                  ))
                ) : (
                  <option value="deepseek-v3-2-251201">DeepSeek V3.2</option>
                )}
              </select>
            );
          })()}
        </div>
        <div className="flex items-center gap-1 shrink-0">
          <button
            onClick={onClose}
            className="p-1.5 text-slate-400 hover:text-white hover:bg-slate-700 rounded-lg transition-all"
            title={lang === 'zh' ? '关闭' : 'Close'}
            onMouseDown={(e) => e.stopPropagation()}
          >
            <X size={14} />
          </button>
        </div>
      </div>

      {/* Messages */}
      <div
        ref={scrollContainerRef}
        className="flex-1 overflow-y-auto p-4 custom-scrollbar"
        style={{ minHeight: 0 }}
      >
        {chatHistory.length === 0 ? (
          <div className="flex flex-col items-center justify-center min-h-full text-center px-4">
            <Bot size={32} className="text-[#90dce1] mb-3" />
            <p className="text-sm text-slate-400 mb-1">
              {lang === 'zh' ? '你好！我可以帮你修改工作流' : 'Hello! I can help you modify the workflow'}
            </p>
            <p className="text-xs text-slate-500">
              {lang === 'zh' ? '试试说："做一个文字生成数字人视频的工作流"' : 'Try saying: "Create a workflow that generates digital human video from text"'}
            </p>
          </div>
        ) : (
          <>
            {chatHistory.map((message) => (
              <ChatMessage
                key={message.id}
                {...message}
                lang={lang}
                onUndo={onUndo ? () => onUndo(message.id) : undefined}
                onRetry={onRetry ? () => onRetry(message.id) : undefined}
                thinking={message.thinking}
                isStreaming={message.isStreaming}
              />
            ))}
            {isProcessing && (
              <div className="flex gap-3 mb-4">
                <div className="w-8 h-8 rounded-full bg-[#90dce1] flex items-center justify-center shrink-0">
                  <Bot size={16} className="text-white" />
                </div>
                <div className="flex-1 bg-slate-800/50 border border-slate-700 rounded-2xl px-4 py-3">
                  <div className="flex items-center gap-2">
                    <div className="flex gap-1">
                      <div className="w-1.5 h-1.5 bg-[#90dce1] rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
                      <div className="w-1.5 h-1.5 bg-[#90dce1] rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
                      <div className="w-1.5 h-1.5 bg-[#90dce1] rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
                    </div>
                    <span className="text-sm text-slate-400">
                      {lang === 'zh' ? '正在思考...' : 'Thinking...'}
                    </span>
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </>
        )}
      </div>

      {/* Input */}
      <ChatInput
        onSend={onSendMessage}
        isProcessing={isProcessing}
        lang={lang}
      />

      {/* 调整大小的手柄 */}
      {/* 边缘 */}
      <div
        className="absolute top-0 left-0 right-0 h-1 cursor-ns-resize z-10"
        onMouseDown={handleResizeMouseDown('n')}
      />
      <div
        className="absolute bottom-0 left-0 right-0 h-1 cursor-ns-resize z-10"
        onMouseDown={handleResizeMouseDown('s')}
      />
      <div
        className="absolute top-0 bottom-0 left-0 w-1 cursor-ew-resize z-10"
        onMouseDown={handleResizeMouseDown('w')}
      />
      <div
        className="absolute top-0 bottom-0 right-0 w-1 cursor-ew-resize z-10"
        onMouseDown={handleResizeMouseDown('e')}
      />
      {/* 角落 */}
      <div
        className="absolute top-0 left-0 w-3 h-3 cursor-nwse-resize z-10"
        onMouseDown={handleResizeMouseDown('nw')}
      />
      <div
        className="absolute top-0 right-0 w-3 h-3 cursor-nesw-resize z-10"
        onMouseDown={handleResizeMouseDown('ne')}
      />
      <div
        className="absolute bottom-0 left-0 w-3 h-3 cursor-nesw-resize z-10"
        onMouseDown={handleResizeMouseDown('sw')}
      />
      <div
        className="absolute bottom-0 right-0 w-3 h-3 cursor-nwse-resize z-10"
        onMouseDown={handleResizeMouseDown('se')}
      />
    </div>
  );
};
