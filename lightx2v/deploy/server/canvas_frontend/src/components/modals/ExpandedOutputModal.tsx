import React, { useState, useEffect } from 'react';
import { X, Download, SaveAll } from 'lucide-react';
import { DataType } from '../../../types';
import { downloadFile } from '../../utils/download';
import { useTranslation, Language } from '../../i18n/useTranslation';
import { pcmToWavUrl } from '../../utils/audio';
import { getAssetPath } from '../../utils/assetPath';
import { isLightX2VResultRef, type LightX2VResultRef } from '../../hooks/useWorkflowExecution';
import { collectLightX2VResultRefs } from '../../utils/resultRef';
import { AudioNodePreview } from '../previews/AudioNodePreview';

interface ExpandedOutputModalProps {
  lang: Language;
  expandedOutput: { nodeId: string; fieldId?: string } | null;
  resolveLightX2VResultRef?: (ref: LightX2VResultRef) => Promise<string>;
  /** 仅用于本地 file 类输出（kind: 'file', file_id）；x2v 任务结果用 resolveLightX2VResultRef，不请求本地 node output url */
  getNodeOutputUrl?: (nodeId: string, portId: string, fileId?: string, runId?: string) => Promise<string | null>;
  workflowId?: string;
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

function isFileRef(c: any): boolean {
  return c != null && typeof c === 'object' && (c.kind === 'file' || c._type === 'file') && !!c.file_id;
}

/** content 为 ref 或 port-keyed 包裹的 ref 时返回该 ref，否则返回 null */
function getLightX2VRefFromContent(content: any): LightX2VResultRef | null {
  if (!content || typeof content !== 'object') return null;
  if (isLightX2VResultRef(content)) return content;
  const refs = collectLightX2VResultRefs(content);
  return refs.length > 0 ? refs[0] : null;
}

export const ExpandedOutputModal: React.FC<ExpandedOutputModalProps> = ({
  lang,
  expandedOutput,
  expandedResultData,
  resolveLightX2VResultRef,
  getNodeOutputUrl,
  workflowId,
  isEditingResult,
  tempEditValue,
  onClose,
  onEditToggle,
  onSaveEdit,
  onTempEditValueChange
}) => {
  const { t } = useTranslation(lang);
  const [resolvedMediaUrl, setResolvedMediaUrl] = useState<string | null>(null);
  const [resolvedFileUrl, setResolvedFileUrl] = useState<string | null>(null);

  // x2v 任务结果（含云端）：只走 resolveLightX2VResultRef，不请求本地 node output url
  const lightX2VRef = expandedResultData?.content ? getLightX2VRefFromContent(expandedResultData.content) : null;
  useEffect(() => {
    if (!lightX2VRef || !resolveLightX2VResultRef) {
      setResolvedMediaUrl(null);
      return;
    }
    let cancelled = false;
    resolveLightX2VResultRef(lightX2VRef).then(url => {
      if (!cancelled) setResolvedMediaUrl(url);
    }).catch(() => { if (!cancelled) setResolvedMediaUrl(null); });
    return () => { cancelled = true; };
  }, [lightX2VRef?.task_id, lightX2VRef?.output_name, lightX2VRef?.is_cloud, resolveLightX2VResultRef]);

  // 仅对本地 file 类输出（kind: 'file', file_id）请求本地后端；x2v ref 已在上方处理，此处不再请求
  useEffect(() => {
    if (!expandedResultData || !expandedOutput || !workflowId) {
      setResolvedFileUrl(null);
      return;
    }
    const content = expandedResultData.content;
    if (getLightX2VRefFromContent(content) != null) {
      setResolvedFileUrl(null);
      return;
    }
    if (!isFileRef(content)) {
      setResolvedFileUrl(null);
      return;
    }
    const type = expandedResultData.type;
    if (type !== DataType.AUDIO && type !== DataType.IMAGE && type !== DataType.VIDEO) {
      setResolvedFileUrl(null);
      return;
    }
    const fileId = (content as { file_id?: string }).file_id;
    const runId = (content as { run_id?: string }).run_id;
    const portId = expandedOutput.fieldId
      || (content && typeof content === 'object' && !Array.isArray(content) && Object.keys(content).length === 1 ? Object.keys(content)[0] : null)
      || 'output';
    if (!workflowId || !fileId || !getNodeOutputUrl) {
      setResolvedFileUrl(null);
      return;
    }
    let cancelled = false;
    getNodeOutputUrl(expandedResultData.nodeId, portId, fileId, runId).then(url => {
      if (!cancelled && url) setResolvedFileUrl(url);
    }).catch(() => {});
    return () => { cancelled = true; };
  }, [expandedResultData?.content, expandedResultData?.nodeId, expandedResultData?.type, expandedOutput?.fieldId, workflowId, getNodeOutputUrl]);

  if (!expandedOutput || !expandedResultData) return null;

  const isTextType = expandedResultData.type === DataType.TEXT;
  const originalText = isTextType
    ? (typeof expandedResultData.content === 'object'
        ? JSON.stringify(expandedResultData.content, null, 2)
        : (expandedResultData.content ?? ''))
    : '';
  const hasTextChanges = isTextType && tempEditValue !== originalText;

  const getDownloadUrl = (): string | null => {
    const c = expandedResultData.content;
    if (resolvedFileUrl) return getAssetPath(resolvedFileUrl) || resolvedFileUrl;
    if (expandedResultData.type !== DataType.TEXT && getLightX2VRefFromContent(c) && resolvedMediaUrl) return resolvedMediaUrl;
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
            {isEditingResult && expandedResultData.type === DataType.TEXT && (
              <p className="text-[10px] text-#90dce1 font-bold uppercase animate-pulse">{t('manual_edit_hint')}</p>
            )}
          </div>
          <div className="flex items-center gap-3">
            {isTextType && isEditingResult ? (
              <button
                onClick={onSaveEdit}
                disabled={!hasTextChanges}
                className={`p-3 rounded-2xl transition-all flex items-center gap-2 px-6 ${
                  hasTextChanges
                    ? 'bg-emerald-600 hover:bg-emerald-500 text-white active:scale-90 shadow-lg shadow-emerald-500/20 cursor-pointer'
                    : 'bg-slate-700 text-slate-500 cursor-not-allowed'
                }`}
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
              const isRef = getLightX2VRefFromContent(raw) != null;
              const isUrlRef = raw && typeof raw === 'object' && (raw as any).kind === 'url' && typeof (raw as any).url === 'string';
              const fromApi = resolvedFileUrl ? (getAssetPath(resolvedFileUrl) || resolvedFileUrl) : '';
              const imgSrc = fromApi
                || (isRef ? (resolvedMediaUrl ?? '') : (isUrlRef ? (raw as any).url : (raw != null && raw !== '' ? (typeof raw === 'string' && (raw.startsWith('http') || raw.startsWith('data:')) ? raw : getAssetPath(raw)) : '')));
              if (!imgSrc || imgSrc === '') {
                return <div className="text-sm text-slate-500">{(isRef && !resolvedMediaUrl) || (isFileRef(raw) && !resolvedFileUrl) ? 'Loading...' : 'No image data'}</div>;
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

              const fromApi = resolvedFileUrl ? (getAssetPath(resolvedFileUrl) || resolvedFileUrl) : '';
              const audioValue = getMediaValue(expandedResultData.content);
              const fallback = !audioValue ? '' : (audioValue.startsWith('data:') ? audioValue : (audioValue.startsWith('http') || audioValue.startsWith('/') || audioValue.startsWith('./assets') || audioValue.startsWith('blob:') ? getAssetPath(audioValue) : pcmToWavUrl(audioValue)));
              const audioSrc = (resolvedMediaUrl || fromApi || fallback) || '';
              if (!audioSrc) {
                return <div className="text-sm text-slate-500">{isFileRef(expandedResultData.content) && !resolvedFileUrl ? 'Loading...' : 'No audio data'}</div>;
              }
              return (
                <div className="w-full max-w-2xl">
                  <AudioNodePreview
                    audioData={{
                      original: audioSrc,
                      trimmed: audioSrc,
                      range: { start: 0, end: 100 }
                    }}
                    readOnly
                  />
                </div>
              );
            })()
          ) : (
            (() => {
              const raw = expandedResultData.content;
              const isRef = getLightX2VRefFromContent(raw) != null;
              const fromApi = resolvedFileUrl ? (getAssetPath(resolvedFileUrl) || resolvedFileUrl) : '';
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
              const fallback = isRef ? (resolvedMediaUrl ?? '') : (v !== '' ? (v.startsWith('http') || v.startsWith('data:') ? v : getAssetPath(v)) : '');
              const videoSrc = fromApi || fallback;
              if (!videoSrc || videoSrc === '') {
                return <div className="text-sm text-slate-500">{(isRef && !resolvedMediaUrl) || (isFileRef(raw) && !resolvedFileUrl) ? 'Loading...' : 'No video data'}</div>;
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
