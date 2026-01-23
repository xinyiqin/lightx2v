import React from 'react';
import { Layers, Trash2, Calendar, Sparkle } from 'lucide-react';
import { WorkflowState } from '../../../types';
import { useTranslation, Language } from '../../i18n/useTranslation';

interface WorkflowCardProps {
  workflow: WorkflowState;
  lang: Language;
  onOpen: (workflow: WorkflowState) => void;
  onDelete: (id: string, e: React.MouseEvent) => void;
  isPreset?: boolean;
}

export const WorkflowCard: React.FC<WorkflowCardProps> = ({
  workflow,
  lang,
  onOpen,
  onDelete,
  isPreset = false
}) => {
  const { t } = useTranslation(lang);

  // Extract preview content from workflow for preset cards
  const textInputNode = workflow.nodes.find(n => n.toolId === 'text-prompt' && n.data?.value);
  const imageInputNode = workflow.nodes.find(n => n.toolId === 'image-input' && n.data?.value && Array.isArray(n.data.value) && n.data.value.length > 0);
  
  const previewText = textInputNode?.data?.value || null;
  const previewImage = imageInputNode?.data?.value?.[0] || null;

  // Preset workflow card style (original style)
  if (isPreset) {
    return (
      <div
        onClick={() => onOpen(workflow)}
        className="group bg-slate-900/50 border border-slate-800 hover:border-emerald-500/50 rounded-[32px] p-6 flex flex-col transition-all hover:shadow-2xl hover:shadow-emerald-500/10 cursor-pointer relative active:scale-95"
      >
        <div className="absolute inset-0 bg-gradient-to-br from-emerald-500/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity rounded-[32px]"></div>
        <div className="flex justify-between items-start relative z-10 mb-3 flex-shrink-0">
          <div className="p-3 bg-slate-800 group-hover:bg-emerald-600 rounded-2xl text-slate-500 group-hover:text-white transition-all shadow-inner">
            <Sparkle size={20} />
          </div>
          <span className="text-[8px] font-black uppercase tracking-widest bg-emerald-500/10 text-emerald-400 px-2 py-1 rounded-full border border-emerald-500/20">
            {t('system_preset')}
          </span>
        </div>
        <div className="space-y-3 relative z-10 flex flex-col min-h-0">
          {/* Preview content */}
          {previewImage ? (
            <div className="w-full aspect-[3/4] rounded-xl overflow-hidden bg-slate-800/50 flex items-center justify-center flex-shrink-0">
              <img
                src={
                  previewImage.startsWith('/') 
                    ? (previewImage.startsWith('/assets/') && !previewImage.startsWith('/canvas/')
                        ? `${(window as any).__ASSET_BASE_PATH__ || '/canvas'}${previewImage}`
                        : previewImage)
                    : (previewImage.startsWith('data:') 
                        ? previewImage 
                        : `data:image/png;base64,${previewImage}`)
                }
                alt="Preview"
                className="w-full h-full object-cover"
                onError={(e) => { (e.target as HTMLImageElement).style.display = 'none'; }}
              />
            </div>
          ) : previewText ? (
            <div className="w-full aspect-[3/4] rounded-xl overflow-hidden bg-slate-800/50 p-4 flex items-center justify-center flex-shrink-0">
              <p className="text-xs text-slate-300 line-clamp-6 leading-relaxed text-center">{previewText}</p>
            </div>
          ) : (
            <div className="w-full aspect-[3/4] rounded-xl bg-slate-800/50 flex-shrink-0"></div>
          )}
          {/* Title below preview */}
          <div className="flex flex-col space-y-1 flex-shrink-0">
            <h3 className="text-lg font-black text-slate-200 group-hover:text-white transition-colors truncate">
              {workflow.name}
            </h3>
            <p className="text-[10px] font-bold text-slate-500 uppercase tracking-widest line-clamp-1">
              {lang === 'zh' ? '多模态自动化创作流水线' : 'Automated multi-modal pipeline'}
            </p>
          </div>
        </div>
      </div>
    );
  }

  // My workflow card style
  return (
    <div
      onClick={() => onOpen(workflow)}
      className="group bg-slate-900/50 border border-slate-800 hover:border-[#90dce1]/50 rounded-[32px] p-6 flex flex-col justify-between h-56 transition-all hover:shadow-2xl hover:shadow-[#90dce1]/10 cursor-pointer relative overflow-hidden active:scale-95"
    >
      <div className="absolute inset-0 bg-gradient-to-br from-[#90dce1]/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity"></div>
      <div className="flex justify-between items-start relative z-10">
        <div className="p-3 bg-slate-800 group-hover:bg-[#90dce1] rounded-2xl text-slate-500 group-hover:text-white transition-all shadow-inner">
          <Layers size={20} />
        </div>
        <button
          onClick={(e) => onDelete(workflow.id, e)}
          className="p-2 text-slate-700 hover:text-red-400 transition-colors"
        >
          <Trash2 size={16} />
        </button>
      </div>
      <div className="space-y-2 relative z-10">
        <h3 className="text-lg font-black text-slate-200 group-hover:text-white transition-colors truncate">
          {workflow.name}
        </h3>
        <div className="flex items-center gap-2 text-[10px] font-bold text-slate-500 uppercase tracking-widest">
          <Calendar size={10} />
          {new Date(workflow.updatedAt).toLocaleDateString()}
        </div>
      </div>
    </div>
  );
};

