import React, { useState } from 'react';
import { X, Download, SaveAll } from 'lucide-react';
import { DataType } from '../../../types';
import { downloadFile } from '../../utils/download';
import { useTranslation, Language } from '../../i18n/useTranslation';
import { pcmToWavUrl } from '../../utils/audio';
import { getAssetPath } from '../../utils/assetPath';

interface ExpandedOutputModalProps {
  lang: Language;
  expandedOutput: { nodeId: string; fieldId?: string } | null;
  expandedResultData: {
    content: any;
    label: string;
    type: DataType;
    nodeId: string;
    originalOutput: any;
  } | null;
  isEditingResult: boolean;
  tempEditValue: string;
  onClose: () => void;
  onEditToggle: () => void;
  onSaveEdit: () => void;
  onTempEditValueChange: (value: string) => void;
}

export const ExpandedOutputModal: React.FC<ExpandedOutputModalProps> = ({
  lang,
  expandedOutput,
  expandedResultData,
  isEditingResult,
  tempEditValue,
  onClose,
  onEditToggle,
  onSaveEdit,
  onTempEditValueChange
}) => {
  const { t } = useTranslation(lang);

  if (!expandedOutput || !expandedResultData) return null;

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center p-8 bg-slate-950/90 backdrop-blur-2xl animate-in fade-in duration-300">
      <div className="bg-slate-900 border border-slate-800 rounded-[40px] shadow-2xl w-full max-w-5xl h-full flex flex-col relative overflow-hidden">
        <div className="p-6 border-b border-slate-800 flex items-center justify-between">
          <div className="flex flex-col gap-1">
            <div className="flex items-center gap-4">
              <h2 className="text-xl font-black uppercase tracking-widest">{expandedResultData.label}</h2>
              {expandedResultData.type === DataType.TEXT && (
                <button
                  onClick={onEditToggle}
                  className={`px-3 py-1.5 rounded-lg text-[10px] font-black uppercase transition-all ${
                    isEditingResult ? 'bg-#90dce1 text-white shadow-lg' : 'bg-slate-800 text-slate-400 hover:text-white'
                  }`}
                >
                  {isEditingResult ? t('save_changes') : t('edit_mode')}
                </button>
              )}
            </div>
            {isEditingResult && (
              <p className="text-[10px] text-#90dce1 font-bold uppercase animate-pulse">{t('manual_edit_hint')}</p>
            )}
          </div>
          <div className="flex items-center gap-3">
            {isEditingResult ? (
              <button
                onClick={onSaveEdit}
                className="p-3 bg-emerald-600 hover:bg-emerald-500 text-white rounded-2xl transition-all active:scale-90 shadow-lg shadow-emerald-500/20 flex items-center gap-2 px-6"
              >
                <SaveAll size={20} />
                <span className="text-sm font-black uppercase">{t('save_changes')}</span>
              </button>
            ) : (
              <button
                onClick={() => downloadFile(expandedResultData.content, expandedResultData.label, expandedResultData.type)}
                className="p-3 bg-slate-800 hover:bg-slate-700 rounded-2xl transition-all active:scale-90"
              >
                <Download size={20} />
              </button>
            )}
            <button onClick={onClose} className="p-3 text-slate-400 hover:text-white transition-all">
              <X size={24} />
            </button>
          </div>
        </div>
        <div className="flex-1 p-12 overflow-y-auto flex items-center justify-center custom-scrollbar">
          {expandedResultData.type === DataType.TEXT ? (
            isEditingResult ? (
              <textarea
                value={tempEditValue}
                onChange={e => onTempEditValueChange(e.target.value)}
                className="w-full h-full bg-slate-950 border-2 border-#90dce1/50 rounded-3xl p-8 text-base text-indigo-100 resize-none focus:ring-0 focus:border-#90dce1 font-mono transition-all custom-scrollbar selection:bg-#90dce1/30"
                placeholder="Manually edit the AI output..."
                autoFocus
              />
            ) : typeof expandedResultData.content === 'object' ? (
              <pre className="text-xs bg-slate-950/50 p-8 rounded-3xl border border-slate-800/50 text-indigo-300 max-w-3xl w-full overflow-auto selection:bg-#90dce1/20">
                {JSON.stringify(expandedResultData.content, null, 2)}
              </pre>
            ) : (
              <p className="text-lg leading-relaxed max-w-3xl whitespace-pre-wrap selection:bg-#90dce1/20">
                {expandedResultData.content}
              </p>
            )
          ) : expandedResultData.type === DataType.IMAGE ? (
            <img src={getAssetPath(expandedResultData.content)} className="max-h-full rounded-2xl shadow-2xl border border-slate-800" />
          ) : expandedResultData.type === DataType.AUDIO ? (
            (() => {
              const getMediaValue = (value: any) => {
                if (!value) return '';
                if (typeof value === 'string') return value;
                if (Array.isArray(value)) {
                  return value.find(item => typeof item === 'string') || '';
                }
                if (typeof value === 'object') {
                  return (
                    (typeof value.url === 'string' && value.url) ||
                    (typeof value.file_url === 'string' && value.file_url) ||
                    (typeof value.path === 'string' && value.path) ||
                    (typeof value.src === 'string' && value.src) ||
                    (typeof value.data === 'string' && value.data) ||
                    ''
                  );
                }
                return '';
              };

              const audioValue = getMediaValue(expandedResultData.content);
              if (!audioValue) {
                return <div className="text-sm text-slate-500">No audio data</div>;
              }

              const isUrlLike = audioValue.startsWith('http://') ||
                audioValue.startsWith('https://') ||
                audioValue.startsWith('/') ||
                audioValue.startsWith('./assets') ||
                audioValue.startsWith('blob:') ||
                audioValue.startsWith('data:');

              const audioSrc = audioValue.startsWith('data:')
                ? audioValue
                : (isUrlLike ? getAssetPath(audioValue) : pcmToWavUrl(audioValue));

              return (
                <audio
                  controls
                  autoPlay
                  src={audioSrc}
                />
              );
            })()
          ) : (
            <video controls autoPlay src={getAssetPath(expandedResultData.content)} className="max-h-full rounded-2xl shadow-2xl" />
          )}
        </div>
      </div>
    </div>
  );
};
