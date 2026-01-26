import React from 'react';
import {
  History,
  ChevronUp,
  ChevronDown,
  ToggleLeft,
  ToggleRight,
  RefreshCw,
  Maximize2,
  ArrowUpRight,
  AlertCircle,
  Volume2,
  Video as VideoIcon
} from 'lucide-react';
import { WorkflowState, WorkflowNode, NodeStatus, DataType, GenerationRun } from '../../../types';
import { TOOLS } from '../../../constants';
import { useTranslation, Language } from '../../i18n/useTranslation';
import { getIcon } from '../../utils/icons';
import { formatTime } from '../../utils/format';
import { getAssetPath } from '../../utils/assetPath';

interface ResultsPanelProps {
  lang: Language;
  workflow: WorkflowState;
  resultsCollapsed: boolean;
  onToggleCollapsed: () => void;
  activeResultsList: WorkflowNode[];
  sourceOutputs: Record<string, any>;
  selectedRunId: string | null;
  onSelectRun: (runId: string | null) => void;
  onToggleShowIntermediate: () => void;
  onExpandOutput: (nodeId: string, fieldId?: string) => void;
  onPinOutputToCanvas?: (content: any, type: DataType) => void;
}

export const ResultsPanel: React.FC<ResultsPanelProps> = ({
  lang,
  workflow,
  resultsCollapsed,
  onToggleCollapsed,
  activeResultsList,
  sourceOutputs,
  selectedRunId,
  onSelectRun,
  onToggleShowIntermediate,
  onExpandOutput,
  onPinOutputToCanvas
}) => {
  const { t } = useTranslation(lang);

  return (
    <footer
      className={`${
        resultsCollapsed ? 'h-12' : 'h-80'
      } border-t border-slate-800/60 bg-slate-900/60 backdrop-blur-3xl z-40 flex flex-col overflow-hidden transition-all duration-300`}
    >
      <div className="px-8 py-4 border-b border-slate-800/60 flex items-center justify-between">
        <div className="flex items-center gap-6">
          <button
            onClick={onToggleCollapsed}
            className="flex items-center gap-2 text-[10px] font-black uppercase tracking-[0.3em] text-#90dce1 hover:text-indigo-300 transition-all"
          >
            <History size={16} />
            {t('execution_results')}
            {resultsCollapsed ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
          </button>
          {!resultsCollapsed && (
            <>
              <div className="h-4 w-px bg-slate-800"></div>
              <button
                onClick={onToggleShowIntermediate}
                className="flex items-center gap-2 text-[9px] font-black uppercase text-slate-400 hover:text-indigo-300 transition-all"
              >
                {workflow.showIntermediateResults ? (
                  <ToggleRight size={16} className="text-#90dce1" />
                ) : (
                  <ToggleLeft size={16} />
                )}
                {t('show_intermediate')}
              </button>
            </>
          )}
        </div>
        {!resultsCollapsed && (
          <div className="flex gap-4">
            {workflow.history.map(r => (
              <button
                key={r.id}
                onClick={() => onSelectRun(r.id)}
                className={`group relative px-4 py-1.5 rounded-full text-[9px] font-bold border transition-all ${
                  selectedRunId === r.id
                    ? 'bg-#90dce1 border-#90dce1 text-white'
                    : 'bg-slate-800 border-slate-700 text-slate-400'
                }`}
              >
                {new Date(r.timestamp).toLocaleTimeString()}
                {r.totalTime !== undefined && (
                  <span className="absolute -top-8 left-1/2 -translate-x-1/2 bg-slate-800 border border-slate-700 text-[8px] px-2 py-1 rounded opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap shadow-xl">
                    {t('run_time')}: {formatTime(r.totalTime)}
                  </span>
                )}
              </button>
            ))}
            {selectedRunId && (
              <button
                onClick={() => onSelectRun(null)}
                className="p-1.5 text-slate-500 hover:text-white transition-all active:scale-90"
              >
                <RefreshCw size={14} />
              </button>
            )}
          </div>
        )}
      </div>
      {!resultsCollapsed && (
        <div className="flex-1 overflow-x-auto p-8 flex gap-8 items-start custom-scrollbar">
          {activeResultsList.length === 0 && !workflow.isRunning ? (
            <div className="flex-1 flex flex-col items-center justify-center opacity-10 animate-pulse">
              <RefreshCw size={48} className="mb-4 animate-spin-slow" />
              <span className="text-[10px] font-black uppercase tracking-widest">{t('awaiting_execution')}</span>
            </div>
          ) : (
            activeResultsList.map(node => {
              const tool = TOOLS.find(t => t.id === node.toolId);
              if (!tool) return null;
              const res = sourceOutputs[node.id] || (tool.category === 'Input' ? node.data.value : null);
              const type = tool?.outputs[0]?.type || DataType.TEXT;
              const isTerminal = !workflow.connections.some(c => c.sourceNodeId === node.id);

              const elapsed =
                node.status === NodeStatus.RUNNING
                  ? ((performance.now() - (node.startTime || performance.now())) / 1000).toFixed(1) + 's'
                  : formatTime(node.executionTime);

              return (
                <div
                  key={node.id}
                  className={`min-w-[320px] max-w-[420px] bg-slate-900/50 rounded-[32px] border p-6 flex flex-col shadow-2xl relative overflow-hidden group transition-all h-[190px] ${
                    node.status === NodeStatus.ERROR
                      ? 'border-red-500/30 bg-red-500/5'
                      : isTerminal
                      ? 'border-emerald-500/30'
                      : 'border-slate-800/60'
                  }`}
                >
                  <div className="flex items-center justify-between mb-4">
                    <div className="flex flex-col gap-0.5">
                      <span className="text-[9px] font-black text-slate-500 uppercase flex items-center gap-2">
                        {React.createElement(getIcon(tool.icon), { size: 12 })}{' '}
                        {lang === 'zh' ? tool.name_zh : tool.name}
                      </span>
                      {(node.status === NodeStatus.RUNNING || node.executionTime !== undefined) && (
                        <span
                          className={`text-[8px] font-bold ${
                            node.status === NodeStatus.RUNNING ? 'text-#90dce1' : 'text-slate-600'
                          }`}
                        >
                          {t('run_time')}: {elapsed}
                        </span>
                      )}
                    </div>
                    <div className="flex gap-2">
                      {node.status !== NodeStatus.ERROR && (
                        <>
                          <button
                            onClick={() => onExpandOutput(node.id)}
                            className="p-1.5 text-slate-500 hover:text-white transition-all"
                          >
                            <Maximize2 size={14} />
                          </button>
                          {onPinOutputToCanvas && (
                            <button
                              onClick={() => onPinOutputToCanvas(res, type)}
                              className="p-1.5 text-slate-500 hover:text-#90dce1 transition-all"
                            >
                              <ArrowUpRight size={14} />
                            </button>
                          )}
                        </>
                      )}
                    </div>
                  </div>
                  <div
                    className={`flex-1 overflow-y-auto rounded-xl p-3 custom-scrollbar ${
                      node.status === NodeStatus.ERROR
                        ? 'bg-red-500/10 border border-red-500/20'
                        : 'bg-slate-950/40'
                    }`}
                  >
                    {node.status === NodeStatus.ERROR ? (
                      <div className="h-full flex flex-col items-center justify-center text-center p-2">
                        <AlertCircle size={24} className="text-red-500 mb-2" />
                        <p className="text-[10px] text-red-400 font-bold uppercase mb-1">{t('execution_failed')}</p>
                        <p className="text-[9px] text-slate-400 line-clamp-3 leading-relaxed">{node.error}</p>
                      </div>
                    ) : type === DataType.TEXT ? (
                      typeof res === 'object' ? (
                        <div className="space-y-2">
                          {Object.entries(res || {}).map(([k, v]) => (
                            <div
                              key={k}
                              onClick={() => onExpandOutput(node.id, k)}
                              className="group/field p-2 bg-slate-900/60 rounded-lg border border-slate-800 hover:border-#90dce1/50 cursor-pointer transition-all"
                            >
                              <div className="flex items-center justify-between mb-1">
                                <span className="text-[8px] font-black text-#90dce1 uppercase tracking-tighter">
                                  {k}
                                </span>
                                <Maximize2 size={8} className="text-slate-600 group-hover/field:text-#90dce1" />
                              </div>
                              <p className="text-[9px] text-slate-400 line-clamp-1">{v as string}</p>
                            </div>
                          ))}
                        </div>
                      ) : (
                        <p className="text-[10px] text-slate-400 line-clamp-4 leading-relaxed">{res}</p>
                      )
                    ) : type === DataType.IMAGE ? (
                      <div className="flex gap-2 overflow-x-auto h-full pb-1 custom-scrollbar">
                        {(Array.isArray(res) ? res : [res]).map((img, i) => (
                          <img
                            key={i}
                            src={getAssetPath(img)}
                            className="h-full w-auto object-cover rounded-lg border border-slate-800"
                          />
                        ))}
                      </div>
                    ) : (
                      <div className="flex items-center justify-center h-full text-#90dce1">
                        {type === DataType.AUDIO ? <Volume2 size={24} /> : <VideoIcon size={24} />}
                      </div>
                    )}
                  </div>
                </div>
              );
            })
          )}
        </div>
      )}
    </footer>
  );
};
