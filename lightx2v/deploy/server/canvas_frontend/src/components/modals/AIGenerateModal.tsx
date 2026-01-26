import React from 'react';
import { X, Sparkle, RefreshCw } from 'lucide-react';
import { useTranslation, Language } from '../../i18n/useTranslation';

interface AIGenerateModalProps {
  lang: Language;
  isOpen: boolean;
  description: string;
  isGenerating: boolean;
  onClose: () => void;
  onDescriptionChange: (description: string) => void;
  onGenerate: () => void;
}

export const AIGenerateModal: React.FC<AIGenerateModalProps> = ({
  lang,
  isOpen,
  description,
  isGenerating,
  onClose,
  onDescriptionChange,
  onGenerate
}) => {
  const { t } = useTranslation(lang);

  if (!isOpen) return null;

  return (
    <div
      className="fixed inset-0 bg-black/50 dark:bg-black/60 backdrop-blur-sm z-[150] flex items-center justify-center p-4"
      onClick={onClose}
    >
      <div
        className="relative w-full max-w-3xl max-h-[90vh] bg-slate-900/95 backdrop-blur-[40px] border border-slate-800/60 rounded-3xl shadow-2xl overflow-hidden flex flex-col"
        onClick={e => e.stopPropagation()}
      >
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-800/60">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-gradient-to-br from-#90dce1 to-#90dce1 rounded-xl flex items-center justify-center">
              <Sparkle size={20} className="text-white" />
            </div>
            <h3 className="text-xl font-semibold text-slate-200">{t('ai_generate_workflow')}</h3>
          </div>
          <button
            onClick={onClose}
            className="w-9 h-9 flex items-center justify-center bg-slate-800/80 border border-slate-700 text-slate-400 hover:text-slate-200 rounded-full transition-all"
          >
            <X size={16} />
          </button>
        </div>
        <div className="flex-1 overflow-y-auto p-6">
          <div className="space-y-6">
            <div>
              <label className="block text-sm font-medium text-slate-300 mb-3">{t('describe_workflow')}</label>
              <textarea
                value={description}
                onChange={e => onDescriptionChange(e.target.value)}
                placeholder={t('workflow_example_placeholder')}
                className="w-full h-48 bg-slate-800 border border-slate-700 rounded-xl p-4 text-sm text-slate-200 placeholder-slate-500 focus:border-[#90dce1] focus:outline-none resize-none leading-relaxed"
              />
              <p className="mt-2 text-xs text-slate-500">
                {t('workflow_description_hint')}
              </p>
            </div>
          </div>
        </div>
        <div className="px-6 py-4 border-t border-slate-800/60 flex gap-3">
          <button
            onClick={onClose}
            className="flex-1 px-4 py-3 bg-slate-800 border border-slate-700 text-slate-300 rounded-xl hover:bg-slate-700 transition-all"
            disabled={isGenerating}
          >
            Cancel
          </button>
          <button
            onClick={onGenerate}
            disabled={!description.trim() || isGenerating}
            className="flex-1 px-4 py-3 bg-gradient-to-r from-#90dce1 to-#90dce1 text-white rounded-xl hover:from-#90dce1 hover:to-#90dce1 transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
          >
            {isGenerating ? (
              <>
                <RefreshCw size={16} className="animate-spin" />
                {t('generating_workflow')}
              </>
            ) : (
              <>
                <Sparkle size={16} />
                {t('generate_workflow')}
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
};
