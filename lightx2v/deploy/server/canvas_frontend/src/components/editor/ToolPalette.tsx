import React from 'react';
import { ChevronLeft, Plus } from 'lucide-react';
import { ToolDefinition } from '../../../types';
import { TOOLS } from '../../../constants';
import { useTranslation, Language } from '../../i18n/useTranslation';
import { getIcon } from '../../utils/icons';

interface ToolPaletteProps {
  lang: Language;
  collapsed: boolean;
  onToggleCollapse: () => void;
  onAddNode: (tool: ToolDefinition) => void;
}

export const ToolPalette: React.FC<ToolPaletteProps> = ({
  lang,
  collapsed,
  onToggleCollapse,
  onAddNode
}) => {
  const { t } = useTranslation(lang);

  const categories = ['Input', 'AI Model', 'Image Processing'];
  const categoryMap: Record<string, string> = {
    Input: lang === 'zh' ? '输入' : 'Input',
    'AI Model': lang === 'zh' ? 'AI 模型' : 'AI Model',
    'Image Processing': lang === 'zh' ? '图像处理' : 'Image Processing'
  };

  return (
    <>
      {collapsed && (
        <button
          onClick={onToggleCollapse}
          className="absolute left-4 top-4 z-40 flex items-center gap-2 px-3 py-2.5 bg-slate-800/90 hover:bg-[#90dce1] border border-slate-700 hover:border-[#90dce1] rounded-xl transition-all shadow-lg hover:shadow-[#90dce1]/20"
          title={lang === 'zh' ? '展开工具面板' : 'Expand Tool Palette'}
        >
          <Plus size={18} className="text-slate-300 hover:text-white" />
          <span className="text-sm font-medium text-slate-300 hover:text-white">
            {lang === 'zh' ? '添加节点' : 'Add Node'}
          </span>
        </button>
      )}
      <aside
        className={`${
          collapsed ? 'w-0' : 'w-72'
        } border-r border-slate-800/60 bg-slate-900/40 backdrop-blur-xl flex flex-col z-30 overflow-hidden transition-all duration-300 relative`}
      >
        <div className={`flex-1 overflow-y-auto p-4 space-y-8 ${collapsed ? 'hidden' : ''}`}>
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-[10px] font-black text-slate-500 uppercase tracking-[0.2em]">
              {t('tool_palette')}
            </h3>
            <button
              onClick={onToggleCollapse}
              className="p-1.5 text-slate-400 hover:text-white hover:bg-slate-700 rounded-lg transition-all"
              title={lang === 'zh' ? '收起' : 'Collapse'}
            >
              <ChevronLeft size={14} />
            </button>
          </div>
          {categories.map((cat) => (
            <div key={cat} className="space-y-2.5">
              <span className="text-[9px] text-slate-600 font-black uppercase">
                {categoryMap[cat]}
              </span>
              {TOOLS.filter((t) => t.category === cat).map((tool) => (
                <div
                  key={tool.id}
                  onClick={() => onAddNode(tool)}
                  className="flex items-center gap-3 p-3 rounded-2xl bg-slate-800/20 border border-slate-800/60 hover:border-[#90dce1]/40 hover:bg-slate-800/40 cursor-pointer transition-all active:scale-95 group"
                >
                  <div className="p-2.5 rounded-xl bg-slate-800 group-hover:bg-[#90dce1] group-hover:text-white transition-colors">
                    {React.createElement(getIcon(tool.icon), { size: 16 })}
                  </div>
                  <div className="flex flex-col">
                    <span className="text-xs font-bold text-slate-300">
                      {lang === 'zh' ? tool.name_zh : tool.name}
                    </span>
                    <span className="text-[9px] text-slate-500 line-clamp-1">
                      {lang === 'zh' ? tool.description_zh : tool.description}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          ))}
        </div>
      </aside>
    </>
  );
};
