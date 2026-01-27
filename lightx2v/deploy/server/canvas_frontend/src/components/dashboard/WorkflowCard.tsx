import React from 'react';
import { Layers, Trash2, Calendar, Sparkle, Lock, Globe, Heart, Users, Boxes } from 'lucide-react';
import { WorkflowState } from '../../../types';
import { useTranslation, Language } from '../../i18n/useTranslation';

interface WorkflowCardProps {
  workflow: WorkflowState;
  lang: Language;
  onOpen: (workflow: WorkflowState) => void;
  onDelete: (id: string, e: React.MouseEvent) => void;
  isPreset?: boolean;
  onToggleVisibility?: (workflowId: string, visibility: 'private' | 'public') => void;
  onToggleThumbsup?: (workflowId: string) => void;
  mode?: 'MY' | 'COMMUNITY' | 'PRESET';
  accentColor?: string;
}

export const WorkflowCard: React.FC<WorkflowCardProps> = ({
  workflow,
  lang,
  onOpen,
  onDelete,
  isPreset = false,
  onToggleVisibility,
  onToggleThumbsup,
  mode = 'MY',
  accentColor = '#90dce1'
}) => {
  const { t } = useTranslation(lang);

  // Extract preview content from workflow (image first, then text)
  const textInputNode = workflow.nodes.find(n => n.toolId === 'text-input' && n.data?.value);
  const imageInputNode = workflow.nodes.find(n => n.toolId === 'image-input' && n.data?.value);
  const imageValue = imageInputNode?.data?.value;

  const previewText = typeof textInputNode?.data?.value === 'string' ? textInputNode?.data?.value : null;
  const previewImage = Array.isArray(imageValue)
    ? imageValue[0]
    : (typeof imageValue === 'string' ? imageValue : null);

  const cardMode = isPreset ? 'PRESET' : mode;
  const visibility = workflow.visibility || (cardMode !== 'MY' ? 'public' : 'private');
  const isPublic = visibility === 'public';
  const nextVisibility = isPublic ? 'private' : 'public';
  const thumsupCount = workflow.thumsupCount ?? 0;
  const thumsupLiked = workflow.thumsupLiked ?? false;
  const authorName = workflow.authorName || workflow.authorId || 'Anonymous';
  const showThumbsup = visibility === 'public';
  const categoryColor = cardMode === 'PRESET'
    ? '#a78bfa'
    : cardMode === 'COMMUNITY'
    ? '#fbbf24'
    : accentColor;

  return (
    <div
      onClick={(e) => {
        const target = e.target as HTMLElement | null;
        if (target?.closest?.('[data-card-action="true"]')) return;
        onOpen(workflow);
      }}
      className="group flex flex-col bg-slate-900/50 border rounded-[32px] overflow-hidden cursor-pointer transition-all hover:shadow-[0_30px_60px_rgba(0,0,0,0.6)] active:scale-[0.98] aspect-[9/16] relative w-full"
      style={{ borderColor: `${categoryColor}22` }}
    >
      {cardMode === 'COMMUNITY' && (
        <div className="absolute inset-0 bg-gradient-to-br from-violet-600/5 to-transparent pointer-events-none" />
      )}

      <div className="relative flex-1 bg-slate-950/50 flex items-center justify-center overflow-hidden">
        {previewImage ? (
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
            className="w-full h-full object-cover transition-transform duration-700 group-hover:scale-110"
            onError={(e) => { (e.target as HTMLImageElement).style.display = 'none'; }}
          />
        ) : previewText ? (
          <div className="px-10 py-8 text-[13px] text-slate-900 italic line-clamp-[10] leading-relaxed bg-white w-full h-full border-b border-slate-800/50 flex items-center justify-center text-center">
            {previewText}
          </div>
        ) : (
          <div className="p-8 text-[10px] text-slate-700 font-bold uppercase tracking-widest text-center opacity-30 flex flex-col items-center">
            <Boxes size={48} className="mb-4" /> No Preview Data
          </div>
        )}
        <div
          className="absolute top-5 left-5 p-2.5 backdrop-blur-2xl rounded-2xl shadow-2xl flex items-center justify-center transition-all group-hover:scale-110 z-10"
          style={{ backgroundColor: `${categoryColor}cc`, color: '#000' }}
        >
          {cardMode === 'PRESET'
            ? <Sparkle size={20} />
            : cardMode === 'COMMUNITY'
            ? <Users size={20} />
            : (isPublic ? <Globe size={20} /> : <Lock size={20} />)}
        </div>
      </div>

      <div className="p-7 space-y-4 shrink-0 bg-slate-900/80 backdrop-blur-md relative border-t border-white/5">
        <div>
          <div className="flex items-center justify-between mb-1 gap-3">
            <h3 className="text-lg font-black text-slate-100 group-hover:text-white transition-colors truncate flex-1 tracking-tight">
              {workflow.name}
            </h3>
            {cardMode === 'MY' && (
              <div className={`px-2.5 py-1 rounded-full text-[9px] font-black uppercase tracking-tighter ${
                isPublic ? 'bg-green-500/20 text-green-400' : 'bg-slate-800 text-slate-500'
              }`}>
                {isPublic ? t('visibility_public') : t('visibility_private')}
              </div>
            )}
          </div>

          {cardMode === 'COMMUNITY' ? (
            <div className="flex items-center justify-between mt-4 pt-4 border-t border-white/5">
              <div className="flex items-center gap-2.5">
                <div className="w-7 h-7 rounded-full bg-slate-800 border border-white/10 flex items-center justify-center overflow-hidden shadow-lg">
                  <img
                    src={`https://api.dicebear.com/7.x/avataaars/svg?seed=${authorName}&backgroundColor=1e293b`}
                    alt="Author"
                    className="w-full h-full"
                  />
                </div>
                <span className="text-[11px] font-black text-slate-400 uppercase tracking-tight">
                  {authorName}
                </span>
              </div>
              <div className="flex items-center gap-3">
                {showThumbsup && (
                  <button
                    data-card-action="true"
                    onClick={(e) => {
                      e.stopPropagation();
                      onToggleThumbsup?.(workflow.id);
                    }}
                    className={`flex items-center gap-1.5 px-2 py-1.5 rounded-xl transition-all duration-300 hover:bg-rose-500/10 active:scale-90 ${
                      thumsupLiked ? 'text-rose-500' : 'text-slate-500 hover:text-rose-400'
                    }`}
                    title="Thumsup"
                  >
                    <Heart size={16} className={thumsupLiked ? 'fill-rose-500' : ''} />
                    <span className="text-[11px] font-black">{thumsupCount}</span>
                  </button>
                )}
              </div>
            </div>
          ) : (
            <div className="flex items-center justify-between mt-2">
              <div className="flex items-center gap-2 text-[11px] font-bold text-slate-500 uppercase tracking-widest">
                <Calendar size={12} /> {new Date(workflow.updatedAt).toLocaleDateString()}
              </div>
              <div className="flex items-center gap-2">
                {showThumbsup && (
                  <button
                    data-card-action="true"
                    onClick={(e) => {
                      e.stopPropagation();
                      onToggleThumbsup?.(workflow.id);
                    }}
                    className={`flex items-center gap-1.5 px-2 py-1.5 rounded-xl transition-all duration-300 hover:bg-rose-500/10 active:scale-90 ${
                      thumsupLiked ? 'text-rose-500' : 'text-slate-500 hover:text-rose-400'
                    }`}
                    title="Thumsup"
                  >
                    <Heart size={16} className={thumsupLiked ? 'fill-rose-500' : ''} />
                    <span className="text-[11px] font-black">{thumsupCount}</span>
                  </button>
                )}
                {cardMode === 'MY' && (
                  <button
                    data-card-action="true"
                    onClick={(e) => {
                      e.stopPropagation();
                      onDelete(workflow.id, e);
                    }}
                    className="p-2.5 text-slate-700 hover:text-red-400 transition-all hover:bg-red-400/5 rounded-xl active:scale-90"
                  >
                    <Trash2 size={18} />
                  </button>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
