import React, { useState, useEffect } from 'react';
import { X, Download, SaveAll } from 'lucide-react';
import { DataType } from '../../../types';
import { downloadFile } from '../../utils/download';
import { useTranslation, Language } from '../../i18n/useTranslation';
import { pcmToWavUrl } from '../../utils/audio';
import { getAssetPath } from '../../utils/assetPath';
import { isLightX2VResultRef, type LightX2VResultRef } from '../../hooks/useWorkflowExecution';

interface ExpandedOutputModalProps {
  lang: Language;
  expandedOutput: { nodeId: string; fieldId?: string } | null;
  resolveLightX2VResultRef?: (ref: LightX2VResultRef) => Promise<string>;
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
  resolveLightX2VResultRef,
  isEditingResult,
  tempEditValue,
  onClose,
  onEditToggle,
  onSaveEdit,
  onTempEditValueChange
}) => {
  const { t } = useTranslation(lang);
  const [resolvedMediaUrl, setResolvedMediaUrl] = useState<string | null>(null);

  useEffect(() => {
    if (!expandedResultData?.content || !resolveLightX2VResultRef || !isLightX2VResultRef(expandedResultData.content)) {
      setResolvedMediaUrl(null);
      return;
    }
    let cancelled = false;
    resolveLightX2VResultRef(expandedResultData.content).then(url => {
      if (!cancelled) setResolvedMediaUrl(url);
    }).catch(() => { if (!cancelled) setResolvedMediaUrl(null); });
    return () => { cancelled = true; };
  }, [expandedResultData?.content, resolveLightX2VResultRef]);

  if (!expandedOutput || !expandedResultData) return null;

  const getDownloadUrl = (): string | null => {
    const c = expandedResultData.content;
    if (expandedResultData.type !== DataType.TEXT && isLightX2VResultRef(c) && resolvedMediaUrl) return resolvedMediaUrl;
    if (typeof c === 'string') return c || null;
    if (Array.isArray(c)) {
      const first = c.find((item: any) => typeof item === 'string');
      return first ?? null;
    }
    if (c && typeof c === 'object') {
      return (
        (typeof (c as any).url === 'string' && (c as any).url) ||
        (typeof (c as any).file_url === 'string' && (c as any).file_url) ||
        (typeof (c as any).path === 'string' && (c as any).path) ||
        (typeof (c as any).src === 'string' && (c as any).src) ||
        (typeof (c as any).data === 'string' && (c as any).data) ||
        (typeof (c as any)._full_data === 'string' && (c as any)._full_data) ||
        null
      );
    }
    return null;
  };

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
                onClick={async () => {
                  const url = getDownloadUrl();
                  if (url == null) return;
                  try {
                    await downloadFile(url, expandedResultData.label, expandedResultData.type);
                  } catch (e) {
                    console.error('[ExpandedOutputModal] Download failed:', e);
                  }
                }}
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
            (() => {
              const raw = expandedResultData.content;
              const isRef = isLightX2VResultRef(raw);
              const imgSrc = isRef ? (resolvedMediaUrl ?? '') : (raw != null && raw !== '' ? (typeof raw === 'string' && (raw.startsWith('http') || raw.startsWith('data:')) ? raw : getAssetPath(raw)) : '');
              if (!imgSrc || imgSrc === '') {
                return <div className="text-sm text-slate-500">{isRef && !resolvedMediaUrl ? 'Loading...' : 'No image data'}</div>;
              }
              return <img src={imgSrc} className="max-h-full rounded-2xl shadow-2xl border border-slate-800" alt="" />;
            })()
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
            (() => {
              const raw = expandedResultData.content;
              const isRef = isLightX2VResultRef(raw);
              const getMediaValue = (val: any): string => {
                if (val == null) return '';
                if (typeof val === 'string') return val;
                if (Array.isArray(val)) {
                  const first = val.find((item: any) => typeof item === 'string');
                  return first != null ? first : '';
                }
                if (typeof val === 'object') {
                  return (
                    (typeof val.data === 'string' && val.data) ||
                    (typeof val.url === 'string' && val.url) ||
                    (typeof val.file_url === 'string' && val.file_url) ||
                    (typeof val._full_data === 'string' && val._full_data) ||
                    ''
                  );
                }
                return '';
              };
              const v = getMediaValue(raw);
              const videoSrc = isRef ? (resolvedMediaUrl ?? '') : (v !== '' ? (v.startsWith('http') || v.startsWith('data:') ? v : getAssetPath(v)) : '');
              if (!videoSrc || videoSrc === '') {
                return <div className="text-sm text-slate-500">{isRef && !resolvedMediaUrl ? 'Loading...' : 'No video data'}</div>;
              }
              return (
                <video
                  controls
                  autoPlay
                  src={videoSrc}
                  className="max-h-full rounded-2xl shadow-2xl"
                />
              );
            })()
          )}
        </div>
      </div>
    </div>
  );
};
