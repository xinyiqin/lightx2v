import React, { useState } from 'react';
import {
  Settings,
  Trash2,
  Boxes,
  MessageSquare,
  AlertCircle,
  RefreshCw,
  Target,
  Upload,
  Search,
  ListPlus,
  CheckCircle2,
  X,
  Globe
} from 'lucide-react';
import { WorkflowState, WorkflowNode, Port, DataType } from '../../../types';
import { TOOLS } from '../../../constants';
import { useTranslation, Language } from '../../i18n/useTranslation';
import { uploadNodeInputFile } from '../../utils/workflowFileManager';

interface NodeConfigPanelProps {
  lang: Language;
  workflow: WorkflowState;
  selectedNodeId: string | null;
  activeOutputs: Record<string, any>;
  sourceOutputs: Record<string, any>;
  disconnectedInputs: Array<{
    nodeId: string;
    port: Port;
    toolName: string;
    isSourceNode?: boolean;
    dataType: DataType;
  }>;
  lightX2VVoiceList?: { voices?: any[]; emotions?: string[]; languages?: any[] } | null;
  loadingVoiceList: boolean;
  voiceSearchQuery: string;
  setVoiceSearchQuery: (query: string) => void;
  showVoiceFilter: boolean;
  setShowVoiceFilter: (show: boolean) => void;
  voiceFilterGender: string;
  setVoiceFilterGender: (gender: string) => void;
  filteredVoices: any[];
  isFemaleVoice: (voiceType: string) => boolean;
  cloneVoiceList: any[];
  loadingCloneVoiceList: boolean;
  onUpdateNodeData: (nodeId: string, key: string, value: any) => void;
  onDeleteNode: (nodeId: string) => void;
  onGlobalInputChange: (nodeId: string, portId: string, value: any) => void;
  onShowCloneVoiceModal: () => void;
  collapsed?: boolean;
  style?: React.CSSProperties;
}

export const NodeConfigPanel: React.FC<NodeConfigPanelProps> = ({
  lang,
  workflow,
  selectedNodeId,
  activeOutputs,
  sourceOutputs,
  disconnectedInputs,
  lightX2VVoiceList,
  loadingVoiceList,
  voiceSearchQuery,
  setVoiceSearchQuery,
  showVoiceFilter,
  setShowVoiceFilter,
  voiceFilterGender,
  setVoiceFilterGender,
  filteredVoices,
  isFemaleVoice,
  cloneVoiceList,
  loadingCloneVoiceList,
  onUpdateNodeData,
  onDeleteNode,
  onGlobalInputChange,
  onShowCloneVoiceModal,
  collapsed = false,
  style
}) => {
  const { t } = useTranslation(lang);
  const [uploadingNodes, setUploadingNodes] = useState<Set<string>>(new Set());
  const selectedNode = selectedNodeId ? workflow.nodes.find(n => n.id === selectedNodeId) : null;

  if (selectedNodeId && selectedNode) {
      return (
        <aside
          className={`flex flex-col z-30 transition-all ${collapsed ? 'h-0 overflow-hidden' : 'p-6 overflow-y-auto'}`}
          style={style}
      >
        <div className="space-y-8 animate-in slide-in-from-right-4 duration-300">
          <div className="flex items-center justify-between">
            <h2 className="text-xs font-black uppercase tracking-widest text-slate-500 flex items-center gap-2">
              <Settings size={14} /> {t('settings')}
            </h2>
            <button
              onClick={() => onDeleteNode(selectedNode.id)}
              className="p-2 text-slate-500 hover:text-red-400 transition-colors"
            >
              <Trash2 size={16} />
            </button>
          </div>
          <div className="space-y-6">
            {/* Model Selection */}
            {TOOLS.find(t => t.id === selectedNode.toolId)?.models && (
              <div className="space-y-2">
                <span className="text-[10px] text-slate-500 font-black uppercase flex items-center gap-2">
                  <Boxes size={12} /> {t('select_model')}
                </span>
                <select
                  value={selectedNode.data.model}
                  onChange={e => onUpdateNodeData(selectedNode.id, 'model', e.target.value)}
                  className="w-full bg-slate-800/80 hover:bg-slate-800 rounded-xl p-3 text-xs border border-slate-700/60 hover:border-slate-600 text-slate-200 focus:outline-none focus:ring-2 focus:ring-#90dce1/50 focus:border-[#90dce1] transition-all cursor-pointer appearance-none bg-[url('data:image/svg+xml;charset=utf-8,%3Csvg xmlns=%27http://www.w3.org/2000/svg%27 fill=%27none%27 viewBox=%270 0 20 20%27%3E%3Cpath stroke=%27%23cbd5e1%27 stroke-linecap=%27round%27 stroke-linejoin=%27round%27 stroke-width=%271.5%27 d=%27M6 8l4 4 4-4%27/%3E%3C/svg%3E')] bg-[length:20px_20px] bg-[right_12px_center] bg-no-repeat pr-10 shadow-sm"
                >
                  {TOOLS.find(t => t.id === selectedNode.toolId)?.models?.map(m => (
                    <option key={m.id} value={m.id}>{m.name}</option>
                  ))}
                </select>
              </div>
            )}

            {/* Web Search Toggle for DeepSeek and Doubao */}
            {selectedNode.toolId === 'text-generation' &&
              (selectedNode.data.model?.startsWith('deepseek-') || selectedNode.data.model?.startsWith('doubao-')) && (
              <div className="space-y-2">
                <span className="text-[10px] text-slate-500 font-black uppercase flex items-center gap-2">
                  <Globe size={12} /> {lang === 'zh' ? '联网搜索' : 'Web Search'}
                </span>
                <label className="flex items-center gap-3 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={selectedNode.data.useSearch || false}
                    onChange={e => onUpdateNodeData(selectedNode.id, 'useSearch', e.target.checked)}
                    className="w-5 h-5 rounded border-slate-600 bg-slate-800 text-[#90dce1] focus:ring-#90dce1 focus:ring-offset-slate-900 cursor-pointer"
                  />
                  <span className="text-xs text-slate-300">
                    {lang === 'zh' ? '启用联网搜索功能' : 'Enable web search'}
                  </span>
                </label>
              </div>
            )}

            {/* Gemini Text Mode */}
            {selectedNode.toolId === 'text-generation' && (
              <div className="space-y-6">
                <div className="space-y-2">
                  <span className="text-[10px] text-slate-500 font-black uppercase">{t('mode')}</span>
                  <select
                    value={selectedNode.data.mode}
                    onChange={e => onUpdateNodeData(selectedNode.id, 'mode', e.target.value)}
                    className="w-full bg-slate-800/80 hover:bg-slate-800 rounded-xl p-3 text-xs border border-slate-700/60 hover:border-slate-600 text-slate-200 focus:outline-none focus:ring-2 focus:ring-#90dce1/50 focus:border-[#90dce1] transition-all cursor-pointer appearance-none bg-[url('data:image/svg+xml;charset=utf-8,%3Csvg xmlns=%27http://www.w3.org/2000/svg%27 fill=%27none%27 viewBox=%270 0 20 20%27%3E%3Cpath stroke=%27%23cbd5e1%27 stroke-linecap=%27round%27 stroke-linejoin=%27round%27 stroke-width=%271.5%27 d=%27M6 8l4 4 4-4%27/%3E%3C/svg%3E')] bg-[length:20px_20px] bg-[right_12px_center] bg-no-repeat pr-10 shadow-sm"
                  >
                    {['basic', 'enhance', 'summarize', 'polish', 'custom'].map(m => (
                      <option key={m} value={m}>{m.toUpperCase()}</option>
                    ))}
                  </select>
                </div>

                {selectedNode.data.mode === 'custom' && (
                  <div className="space-y-2">
                    <span className="text-[10px] text-slate-500 font-black uppercase">System Instruction</span>
                    <textarea
                      value={selectedNode.data.customInstruction || ''}
                      onChange={e => onUpdateNodeData(selectedNode.id, 'customInstruction', e.target.value)}
                      className="w-full h-32 bg-slate-800/40 border border-slate-700/50 rounded-2xl p-4 text-xs resize-none focus:border-[#90dce1] transition-all"
                      placeholder="Set global AI behavior..."
                    />
                  </div>
                )}

                {/* Structured Outputs */}
                <div className="space-y-4">
                  <span className="text-[10px] text-slate-500 font-black uppercase">{t('structured_outputs')}</span>
                  {selectedNode.data.customOutputs?.map((o: any, i: number) => (
                    <div key={i} className="p-4 bg-slate-950/40 border border-slate-800 rounded-[24px] space-y-3">
                      <div className="flex items-center gap-2">
                        <input
                          value={o.id}
                          placeholder="Field ID (e.g. prompt)"
                          onChange={e => {
                            const n = [...selectedNode.data.customOutputs];
                            n[i].id = e.target.value;
                            onUpdateNodeData(selectedNode.id, 'customOutputs', n);
                          }}
                          className="bg-transparent border-none text-[10px] font-black text-[#90dce1] flex-1 p-0 focus:ring-0"
                        />
                        <button
                          onClick={() => {
                            const n = selectedNode.data.customOutputs.filter((_: any, idx: number) => idx !== i);
                            onUpdateNodeData(selectedNode.id, 'customOutputs', n);
                          }}
                          className="text-slate-600 hover:text-red-400 transition-colors"
                        >
                          <X size={14} />
                        </button>
                      </div>
                      <textarea
                        value={o.description || ''}
                        placeholder="Instructions for AI (intent, constraints)..."
                        onChange={e => {
                          const n = [...selectedNode.data.customOutputs];
                          n[i].description = e.target.value;
                          onUpdateNodeData(selectedNode.id, 'customOutputs', n);
                        }}
                        className="w-full h-16 bg-slate-900/50 border border-slate-800 rounded-xl p-2 text-[10px] text-slate-400 resize-none focus:border-[#90dce1] focus:ring-0 transition-all"
                      />
                    </div>
                  ))}
                  <button
                    onClick={() => {
                      const n = [...(selectedNode.data.customOutputs || []), { id: `out_${Date.now().toString().slice(-4)}`, label: 'Output', description: '' }];
                      onUpdateNodeData(selectedNode.id, 'customOutputs', n);
                    }}
                    className="w-full py-2 border border-dashed border-slate-700 rounded-xl text-[10px] text-slate-500 uppercase hover:text-[#90dce1] transition-all"
                  >
                    + {t('add_output')}
                  </button>
                </div>
              </div>
            )}

            {/* Aspect Ratio */}
            {(selectedNode.toolId === 'text-to-image' || selectedNode.toolId === 'image-to-image' || selectedNode.toolId.includes('video-gen')) && (
              <div className="space-y-2">
                <span className="text-[10px] text-slate-500 font-black uppercase">{t('aspect_ratio')}</span>
                <select
                  value={selectedNode.data.aspectRatio}
                  onChange={e => onUpdateNodeData(selectedNode.id, 'aspectRatio', e.target.value)}
                  className="w-full bg-slate-800/80 hover:bg-slate-800 rounded-xl p-3 text-xs border border-slate-700/60 hover:border-slate-600 text-slate-200 focus:outline-none focus:ring-2 focus:ring-#90dce1/50 focus:border-[#90dce1] transition-all cursor-pointer appearance-none bg-[url('data:image/svg+xml;charset=utf-8,%3Csvg xmlns=%27http://www.w3.org/2000/svg%27 fill=%27none%27 viewBox=%270 0 20 20%27%3E%3Cpath stroke=%27%23cbd5e1%27 stroke-linecap=%27round%27 stroke-linejoin=%27round%27 stroke-width=%271.5%27 d=%27M6 8l4 4 4-4%27/%3E%3C/svg%3E')] bg-[length:20px_20px] bg-[right_12px_center] bg-no-repeat pr-10 shadow-sm"
                >
                  {selectedNode.toolId.includes('video-gen')
                    ? ['16:9', '9:16'].map(r => <option key={r} value={r}>{r}</option>)
                    : ['1:1', '4:3', '3:4', '16:9', '9:16'].map(r => <option key={r} value={r}>{r}</option>)
                  }
                </select>
              </div>
            )}

            {/* TTS Settings */}
            {selectedNode.toolId === 'tts' && (() => {
              const isLightX2V = selectedNode.data.model === 'lightx2v' || selectedNode.data.model?.startsWith('lightx2v');

              if (!isLightX2V) {
                return (
                  <div className="space-y-2">
                    <span className="text-[10px] text-slate-500 font-black uppercase">{t('voice')}</span>
                    <select
                      value={selectedNode.data.voice || 'Kore'}
                      onChange={e => onUpdateNodeData(selectedNode.id, 'voice', e.target.value)}
                      className="w-full bg-slate-800/80 hover:bg-slate-800 rounded-xl p-3 text-xs border border-slate-700/60 hover:border-slate-600 text-slate-200 focus:outline-none focus:ring-2 focus:ring-#90dce1/50 focus:border-[#90dce1] transition-all cursor-pointer appearance-none bg-[url('data:image/svg+xml;charset=utf-8,%3Csvg xmlns=%27http://www.w3.org/2000/svg%27 fill=%27none%27 viewBox=%270 0 20 20%27%3E%3Cpath stroke=%27%23cbd5e1%27 stroke-linecap=%27round%27 stroke-linejoin=%27round%27 stroke-width=%271.5%27 d=%27M6 8l4 4 4-4%27/%3E%3C/svg%3E')] bg-[length:20px_20px] bg-[right_12px_center] bg-no-repeat pr-10 shadow-sm"
                    >
                      {['Kore', 'Puck', 'Fenrir', 'Charon', 'Zephyr'].map(v => (
                        <option key={v} value={v}>{v}</option>
                      ))}
                    </select>
                  </div>
                );
              }

              return (
                <div className="space-y-4">
                  <div className="space-y-2">
                    <span className="text-[10px] text-slate-500 font-black uppercase">Voice Type</span>
                    {loadingVoiceList ? (
                      <div className="w-full bg-slate-800 rounded-xl p-3 text-xs border border-slate-700 text-slate-400 text-center">
                        Loading voices...
                      </div>
                    ) : lightX2VVoiceList?.voices && lightX2VVoiceList.voices.length > 0 ? (
                      <div className="space-y-3">
                        {/* Search and Filter */}
                        <div className="flex items-center gap-2">
                          <div className="relative flex-1">
                            <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500 w-4 h-4 pointer-events-none" />
                            <input
                              type="text"
                              value={voiceSearchQuery}
                              onChange={e => setVoiceSearchQuery(e.target.value)}
                              placeholder="Search voices..."
                              className="w-full bg-slate-800 rounded-xl pl-10 pr-3 py-2 text-xs border border-slate-700 text-slate-300 placeholder-slate-500 focus:outline-none focus:border-[#90dce1]"
                            />
                          </div>
                          <button
                            onClick={() => setShowVoiceFilter(!showVoiceFilter)}
                            className={`px-3 py-2 bg-slate-800 rounded-xl text-xs border border-slate-700 text-slate-300 hover:bg-slate-700 transition-colors flex items-center gap-1 ${showVoiceFilter ? 'border-[#90dce1] bg-slate-700' : ''}`}
                          >
                            <ListPlus className="w-3 h-3" />
                            Filter
                          </button>
                        </div>

                        {/* Filter Panel */}
                        {showVoiceFilter && (
                          <div className="bg-slate-800 rounded-xl p-3 border border-slate-700">
                            <div className="space-y-1">
                              <span className="text-[10px] text-slate-500 font-black uppercase">Gender</span>
                              <select
                                value={voiceFilterGender}
                                onChange={e => setVoiceFilterGender(e.target.value)}
                                className="w-full bg-slate-900/80 hover:bg-slate-900 rounded-lg px-3 py-1.5 text-xs border border-slate-700/60 hover:border-slate-600 text-slate-300 focus:outline-none focus:ring-2 focus:ring-#90dce1/50 focus:border-[#90dce1] transition-all cursor-pointer appearance-none bg-[url('data:image/svg+xml;charset=utf-8,%3Csvg xmlns=%27http://www.w3.org/2000/svg%27 fill=%27none%27 viewBox=%270 0 20 20%27%3E%3Cpath stroke=%27%23cbd5e1%27 stroke-linecap=%27round%27 stroke-linejoin=%27round%27 stroke-width=%271.5%27 d=%27M6 8l4 4 4-4%27/%3E%3C/svg%3E')] bg-[length:16px_16px] bg-[right_8px_center] bg-no-repeat pr-8 shadow-sm"
                              >
                                <option value="all">All</option>
                                <option value="female">Female</option>
                                <option value="male">Male</option>
                              </select>
                            </div>
                          </div>
                        )}

                        {/* Voice Cards Grid */}
                        <div className="bg-slate-800 rounded-xl border border-slate-700 p-3 max-h-[400px] overflow-y-auto">
                          {filteredVoices.length === 0 ? (
                            <div className="text-center py-8 text-slate-500 text-xs">
                              No voices found matching your criteria
                            </div>
                          ) : (
                            <div className="grid grid-cols-2 gap-2">
                              {filteredVoices.map((voice: any, index: number) => {
                                const uniqueKey = voice.voice_type
                                  ? `${voice.voice_type}_${voice.resource_id || ''}_${index}`
                                  : `voice_${index}`;
                                const isSelected = selectedNode.data.voiceType === voice.voice_type;

                                return (
                                  <label
                                    key={uniqueKey}
                                    className={`relative cursor-pointer m-0 p-3 rounded-lg border-2 transition-all ${
                                      isSelected
                                        ? 'border-[#90dce1] bg-[#90dce1]/10 shadow-lg shadow-[#90dce1]/20'
                                        : 'border-slate-700 bg-slate-900/50 hover:border-slate-600 hover:bg-slate-800'
                                    }`}
                                  >
                                    <input
                                      type="radio"
                                      name="voice-selection"
                                      value={voice.voice_type}
                                      checked={isSelected}
                                      onChange={() => {
                                        onUpdateNodeData(selectedNode.id, 'voiceType', voice.voice_type);
                                        if (voice.resource_id) {
                                          onUpdateNodeData(selectedNode.id, 'resourceId', voice.resource_id);
                                        }
                                      }}
                                      className="sr-only"
                                    />
                                    {voice.version === '2.0' && (
                                      <div className="absolute top-1 right-1 px-1 py-0.5 bg-[#90dce1]/90 text-white text-[7px] font-semibold rounded z-10">
                                        v2.0
                                      </div>
                                    )}
                                    {isSelected && (
                                      <div className="absolute bottom-2 right-2 w-4 h-4 bg-[#90dce1] rounded-full flex items-center justify-center z-20">
                                        <CheckCircle2 className="w-2.5 h-2.5 text-white" />
                                      </div>
                                    )}
                                    <div className="text-xs font-medium text-slate-200 text-center truncate w-full pt-1">
                                      {voice.name || voice.voice_name || voice.voice_type}
                                    </div>
                                  </label>
                                );
                              })}
                            </div>
                          )}
                        </div>
                      </div>
                    ) : (
                      <input
                        type="text"
                        value={selectedNode.data.voiceType || 'zh_female_vv_uranus_bigtts'}
                        onChange={e => onUpdateNodeData(selectedNode.id, 'voiceType', e.target.value)}
                        className="w-full bg-slate-800 rounded-xl p-3 text-xs border border-slate-700"
                        placeholder="zh_female_vv_uranus_bigtts"
                      />
                    )}
                  </div>
                  <div className="space-y-2">
                    <span className="text-[10px] text-slate-500 font-black uppercase">Emotion</span>
                    {lightX2VVoiceList?.emotions && lightX2VVoiceList.emotions.length > 0 ? (
                      <select
                        value={selectedNode.data.emotion || ''}
                        onChange={e => onUpdateNodeData(selectedNode.id, 'emotion', e.target.value)}
                        className="w-full bg-slate-800/80 hover:bg-slate-800 rounded-xl p-3 text-xs border border-slate-700/60 hover:border-slate-600 text-slate-200 focus:outline-none focus:ring-2 focus:ring-#90dce1/50 focus:border-[#90dce1] transition-all cursor-pointer appearance-none bg-[url('data:image/svg+xml;charset=utf-8,%3Csvg xmlns=%27http://www.w3.org/2000/svg%27 fill=%27none%27 viewBox=%270 0 20 20%27%3E%3Cpath stroke=%27%23cbd5e1%27 stroke-linecap=%27round%27 stroke-linejoin=%27round%27 stroke-width=%271.5%27 d=%27M6 8l4 4 4-4%27/%3E%3C/svg%3E')] bg-[length:20px_20px] bg-[right_12px_center] bg-no-repeat pr-10 shadow-sm"
                      >
                        <option value="">None</option>
                        {lightX2VVoiceList.emotions.map((emotion: string, index: number) => (
                          <option key={`emotion_${emotion}_${index}`} value={emotion}>{emotion}</option>
                        ))}
                      </select>
                    ) : (
                      <input
                        type="text"
                        value={selectedNode.data.emotion || ''}
                        onChange={e => onUpdateNodeData(selectedNode.id, 'emotion', e.target.value)}
                        className="w-full bg-slate-800 rounded-xl p-3 text-xs border border-slate-700"
                        placeholder="Optional emotion"
                      />
                    )}
                  </div>
                  <div className="grid grid-cols-2 gap-2">
                    <div className="space-y-2">
                      <span className="text-[10px] text-slate-500 font-black uppercase">Emotion Scale (1-5)</span>
                      <input
                        type="number"
                        min="1"
                        max="5"
                        value={selectedNode.data.emotionScale || 3}
                        onChange={e => onUpdateNodeData(selectedNode.id, 'emotionScale', parseInt(e.target.value))}
                        className="w-full bg-slate-800 rounded-xl p-3 text-xs border border-slate-700"
                      />
                    </div>
                    <div className="space-y-2">
                      <span className="text-[10px] text-slate-500 font-black uppercase">Speech Rate (-50~100)</span>
                      <input
                        type="number"
                        min="-50"
                        max="100"
                        value={selectedNode.data.speechRate || 0}
                        onChange={e => onUpdateNodeData(selectedNode.id, 'speechRate', parseInt(e.target.value))}
                        className="w-full bg-slate-800 rounded-xl p-3 text-xs border border-slate-700"
                      />
                    </div>
                    <div className="space-y-2">
                      <span className="text-[10px] text-slate-500 font-black uppercase">Pitch (-12~12)</span>
                      <input
                        type="number"
                        min="-12"
                        max="12"
                        value={selectedNode.data.pitch || 0}
                        onChange={e => onUpdateNodeData(selectedNode.id, 'pitch', parseInt(e.target.value))}
                        className="w-full bg-slate-800 rounded-xl p-3 text-xs border border-slate-700"
                      />
                    </div>
                    <div className="space-y-2">
                      <span className="text-[10px] text-slate-500 font-black uppercase">Loudness (-50~100)</span>
                      <input
                        type="number"
                        min="-50"
                        max="100"
                        value={selectedNode.data.loudnessRate || 0}
                        onChange={e => onUpdateNodeData(selectedNode.id, 'loudnessRate', parseInt(e.target.value))}
                        className="w-full bg-slate-800 rounded-xl p-3 text-xs border border-slate-700"
                      />
                    </div>
                  </div>
                </div>
              );
            })()}

            {/* Voice Clone Settings */}
            {selectedNode.toolId === 'lightx2v-voice-clone' && (
              <div className="space-y-4">
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-[10px] text-slate-500 font-black uppercase">Cloned Voice</span>
                    <button
                      onClick={onShowCloneVoiceModal}
                      className="px-2 py-1 text-[9px] bg-[#90dce1]/20 hover:bg-[#90dce1]/30 text-[#90dce1] rounded border border-[#90dce1]/30 transition-all"
                    >
                      + New
                    </button>
                  </div>
                  {loadingCloneVoiceList ? (
                    <div className="w-full bg-slate-800 rounded-xl p-3 text-xs border border-slate-700 text-slate-400 text-center">
                      Loading voices...
                    </div>
                  ) : cloneVoiceList.length > 0 ? (
                    <select
                      value={selectedNode.data.speakerId || ''}
                      onChange={e => onUpdateNodeData(selectedNode.id, 'speakerId', e.target.value)}
                      className="w-full bg-slate-800/80 hover:bg-slate-800 rounded-xl p-3 text-xs border border-slate-700/60 hover:border-slate-600 text-slate-200 focus:outline-none focus:ring-2 focus:ring-#90dce1/50 focus:border-[#90dce1] transition-all cursor-pointer appearance-none bg-[url('data:image/svg+xml;charset=utf-8,%3Csvg xmlns=%27http://www.w3.org/2000/svg%27 fill=%27none%27 viewBox=%270 0 20 20%27%3E%3Cpath stroke=%27%23cbd5e1%27 stroke-linecap=%27round%27 stroke-linejoin=%27round%27 stroke-width=%271.5%27 d=%27M6 8l4 4 4-4%27/%3E%3C/svg%3E')] bg-[length:20px_20px] bg-[right_12px_center] bg-no-repeat pr-10 shadow-sm"
                    >
                      <option value="">Select a cloned voice...</option>
                      {cloneVoiceList.map((voice: any) => (
                        <option key={voice.speaker_id} value={voice.speaker_id}>
                          {voice.name || voice.speaker_id}
                        </option>
                      ))}
                    </select>
                  ) : (
                    <div className="p-3 bg-amber-500/10 border border-amber-500/30 rounded-xl text-xs text-amber-400">
                      No cloned voices found. Click "+ New" to create one.
                    </div>
                  )}
                  {selectedNode.data.speakerId && (
                    <div className="p-2 bg-emerald-500/10 border border-emerald-500/30 rounded-lg">
                      <span className="text-[9px] text-emerald-400 font-mono">ID: {selectedNode.data.speakerId}</span>
                    </div>
                  )}
                </div>
                <div className="space-y-2">
                  <span className="text-[10px] text-slate-500 font-black uppercase">Style</span>
                  <input
                    type="text"
                    value={selectedNode.data.style || '正常'}
                    onChange={e => onUpdateNodeData(selectedNode.id, 'style', e.target.value)}
                    className="w-full bg-slate-800 rounded-xl p-3 text-xs border border-slate-700"
                    placeholder="正常"
                  />
                </div>
                <div className="grid grid-cols-2 gap-2">
                  <div className="space-y-2">
                    <span className="text-[10px] text-slate-500 font-black uppercase">Speed (0.5~2.0)</span>
                    <input
                      type="number"
                      step="0.1"
                      min="0.5"
                      max="2.0"
                      value={selectedNode.data.speed || 1.0}
                      onChange={e => onUpdateNodeData(selectedNode.id, 'speed', parseFloat(e.target.value))}
                      className="w-full bg-slate-800 rounded-xl p-3 text-xs border border-slate-700"
                    />
                  </div>
                  <div className="space-y-2">
                    <span className="text-[10px] text-slate-500 font-black uppercase">Volume (-12~12)</span>
                    <input
                      type="number"
                      min="-12"
                      max="12"
                      value={selectedNode.data.volume || 0}
                      onChange={e => onUpdateNodeData(selectedNode.id, 'volume', parseFloat(e.target.value))}
                      className="w-full bg-slate-800 rounded-xl p-3 text-xs border border-slate-700"
                    />
                  </div>
                  <div className="space-y-2">
                    <span className="text-[10px] text-slate-500 font-black uppercase">Pitch (-24~24)</span>
                    <input
                      type="number"
                      min="-24"
                      max="24"
                      value={selectedNode.data.pitch || 0}
                      onChange={e => onUpdateNodeData(selectedNode.id, 'pitch', parseFloat(e.target.value))}
                      className="w-full bg-slate-800 rounded-xl p-3 text-xs border border-slate-700"
                    />
                  </div>
                  <div className="space-y-2">
                    <span className="text-[10px] text-slate-500 font-black uppercase">Language</span>
                    <select
                      value={selectedNode.data.language || 'ZH_CN'}
                      onChange={e => onUpdateNodeData(selectedNode.id, 'language', e.target.value)}
                      className="w-full bg-slate-800/80 hover:bg-slate-800 rounded-xl p-3 text-xs border border-slate-700/60 hover:border-slate-600 text-slate-200 focus:outline-none focus:ring-2 focus:ring-#90dce1/50 focus:border-[#90dce1] transition-all cursor-pointer appearance-none bg-[url('data:image/svg+xml;charset=utf-8,%3Csvg xmlns=%27http://www.w3.org/2000/svg%27 fill=%27none%27 viewBox=%270 0 20 20%27%3E%3Cpath stroke=%27%23cbd5e1%27 stroke-linecap=%27round%27 stroke-linejoin=%27round%27 stroke-width=%271.5%27 d=%27M6 8l4 4 4-4%27/%3E%3C/svg%3E')] bg-[length:20px_20px] bg-[right_12px_center] bg-no-repeat pr-10 shadow-sm"
                    >
                      {['ZH_CN', 'EN_US', 'ZH_CN_SICHUAN', 'ZH_CN_HK'].map(l => (
                        <option key={l} value={l}>{l}</option>
                      ))}
                    </select>
                  </div>
                </div>
              </div>
            )}

            {/* Connected AI Output Fields */}
            {selectedNode && (() => {
              const tool = TOOLS.find(t => t.id === selectedNode.toolId);
              if (!tool) return null;

              const connectedInputs = tool.inputs
                .map(port => {
                  const conn = workflow.connections.find(c => c.targetNodeId === selectedNode.id && c.targetPortId === port.id);
                  if (!conn) return null;

                  const sourceNode = workflow.nodes.find(n => n.id === conn.sourceNodeId);
                  if (!sourceNode || sourceNode.toolId !== 'text-generation' || !sourceNode.data.customOutputs) return null;

                  const isCustomOutput = sourceNode.data.customOutputs.some((o: any) => o.id === conn.sourcePortId);
                  if (!isCustomOutput) return null;

                  const sourceOutput = activeOutputs[conn.sourceNodeId] || sourceOutputs[conn.sourceNodeId];
                  let fieldValue = '';
                  if (sourceOutput && typeof sourceOutput === 'object' && conn.sourcePortId in sourceOutput) {
                    fieldValue = typeof sourceOutput[conn.sourcePortId] === 'string' ? sourceOutput[conn.sourcePortId] : JSON.stringify(sourceOutput[conn.sourcePortId], null, 2);
                  }

                  const overrideValue = selectedNode.data.inputOverrides?.[port.id];
                  const displayValue = overrideValue !== undefined ? (typeof overrideValue === 'string' ? overrideValue : JSON.stringify(overrideValue, null, 2)) : fieldValue;

                  const fieldLabel = sourceNode.data.customOutputs.find((o: any) => o.id === conn.sourcePortId)?.label || conn.sourcePortId;

                  return { port, conn, sourceNode, fieldLabel, displayValue, fieldValue, overrideValue };
                })
                .filter((item): item is NonNullable<typeof item> => item !== null);

              if (connectedInputs.length === 0) return null;

              return (
                <div className="space-y-4">
                  <span className="text-[10px] text-slate-500 font-black uppercase flex items-center gap-2">
                    <MessageSquare size={12} />
                    {lang === 'zh' ? '连接的AI输出字段' : 'Connected AI Output Fields'}
                  </span>
                  {connectedInputs.map(({ port, conn, sourceNode, fieldLabel, displayValue, fieldValue, overrideValue }) => (
                    <div key={port.id} className="p-4 bg-slate-950/40 border border-slate-800 rounded-[24px] space-y-3">
                      <div className="flex items-center justify-between">
                        <div className="flex flex-col gap-1">
                          <span className="text-[9px] font-black text-[#90dce1] uppercase">{fieldLabel}</span>
                          <span className="text-[8px] text-slate-500">
                            {lang === 'zh' ? '来自' : 'From'}: {lang === 'zh' ? TOOLS.find(t => t.id === sourceNode.toolId)?.name_zh : TOOLS.find(t => t.id === sourceNode.toolId)?.name} → {port.label}
                          </span>
                        </div>
                        {overrideValue !== undefined && (
                          <button
                            onClick={() => {
                              const newOverrides = { ...(selectedNode.data.inputOverrides || {}) };
                              delete newOverrides[port.id];
                              onUpdateNodeData(selectedNode.id, 'inputOverrides', Object.keys(newOverrides).length > 0 ? newOverrides : undefined);
                            }}
                            className="p-1.5 text-slate-500 hover:text-red-400 transition-colors"
                            title={lang === 'zh' ? '恢复原始值' : 'Restore original value'}
                          >
                            <RefreshCw size={12} />
                          </button>
                        )}
                      </div>
                      <textarea
                        value={displayValue}
                        onChange={e => {
                          const newOverrides = { ...(selectedNode.data.inputOverrides || {}) };
                          newOverrides[port.id] = e.target.value;
                          onUpdateNodeData(selectedNode.id, 'inputOverrides', newOverrides);
                        }}
                        className="w-full h-32 bg-slate-900/50 border border-slate-800 rounded-xl p-3 text-[10px] text-slate-300 resize-none focus:border-[#90dce1] focus:ring-0 transition-all font-mono"
                        placeholder={lang === 'zh' ? '编辑字段内容...' : 'Edit field content...'}
                      />
                      {overrideValue !== undefined && (
                        <div className="flex items-center gap-2 text-[8px] text-amber-400">
                          <AlertCircle size={10} />
                          <span>{lang === 'zh' ? '已修改，将使用此值覆盖连接的值' : 'Modified: This value will override the connected value'}</span>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              );
            })()}
          </div>
        </div>
      </aside>
    );
  }

  // Global Inputs Panel
  return (
    <aside className="w-80 border-l border-slate-800/60 bg-slate-900/40 backdrop-blur-xl flex flex-col z-30 p-6 overflow-y-auto">
      <div className="space-y-12">
        <div className="space-y-6">
          <h2 className="text-xs font-black uppercase tracking-widest text-slate-500 flex items-center gap-2">
            <Target size={14} /> {t('global_inputs')}
          </h2>
          {disconnectedInputs.length === 0 ? (
            <p className="text-[10px] text-slate-600 italic px-2">{t('all_inputs_automated')}</p>
          ) : (
            <div className="space-y-4">
              {disconnectedInputs.map(item => (
                <div key={`${item.nodeId}-${item.port.id}`} className="space-y-2 p-5 bg-slate-800/20 border border-slate-800 rounded-[32px]">
                  <span className="text-[9px] font-black text-slate-500 uppercase px-1">
                    {item.port.label} ({lang === 'zh' ? '针对' : 'for'} {item.toolName})
                  </span>
                  {item.dataType === DataType.TEXT ? (
                    <textarea
                      value={item.isSourceNode ? (workflow.nodes.find(n => n.id === item.nodeId)?.data.value || '') : (workflow.globalInputs[`${item.nodeId}-${item.port.id}`] || '')}
                      onChange={e => item.isSourceNode ? onUpdateNodeData(item.nodeId, 'value', e.target.value) : onGlobalInputChange(item.nodeId, item.port.id, e.target.value)}
                      className="w-full h-24 bg-slate-900 border border-slate-800 rounded-xl p-3 text-xs resize-none focus:border-[#90dce1] transition-all custom-scrollbar"
                    />
                  ) : (
                    <div className="space-y-3">
                      <label className="flex items-center justify-center w-full h-12 border border-dashed border-slate-700 rounded-xl cursor-pointer hover:border-[#90dce1] hover:bg-[#90dce1]/5 transition-all gap-2">
                        <Upload size={14} className="text-slate-500" />
                        <span className="text-[10px] text-slate-500 font-bold uppercase">{lang === 'zh' ? '上传文件' : 'Upload File'}</span>
                        <input
                          type="file"
                          accept={item.dataType === DataType.IMAGE ? "image/*" : item.dataType === DataType.AUDIO ? "audio/*" : "video/*"}
                          className="hidden"
                          disabled={uploadingNodes.has(item.nodeId)}
                          onChange={async (e) => {
                            const file = e.target.files?.[0];
                            if (!file) return;

                            // 需要 workflow.id 才能上传
                            if (!workflow.id || (!workflow.id.startsWith('workflow-') && !workflow.id.match(/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i))) {
                              console.error('[NodeConfigPanel] Cannot upload file: workflow ID is not available');
                              return;
                            }

                            setUploadingNodes(prev => new Set(prev).add(item.nodeId));

                            try {
                              if (item.isSourceNode) {
                                const node = workflow.nodes.find(n => n.id === item.nodeId);
                                if (!node) return;

                                const tool = TOOLS.find(t => t.id === node.toolId);
                                if (!tool || tool.category !== 'Input') {
                                  console.error('[NodeConfigPanel] Cannot upload file: node is not an input node');
                                  return;
                                }

                                const outputPort = tool.outputs[0];
                                if (!outputPort) {
                                  console.error('[NodeConfigPanel] Cannot upload file: output port not found');
                                  return;
                                }

                                const result = await uploadNodeInputFile(workflow.id!, item.nodeId, outputPort.id, file);
                                if (result) {
                                  // 更新 node.data.value，始终使用数组格式
                                  const currentValue = node.data.value || [];
                                  const existingUrls = Array.isArray(currentValue) ? currentValue : [currentValue].filter(Boolean);
                                  const newValue = item.dataType === DataType.IMAGE
                                    ? [...existingUrls, result.file_url]
                                    : [result.file_url];
                                  onUpdateNodeData(item.nodeId, 'value', newValue);
                                }
                              } else {
                                // 对于非源节点的全局输入，暂时保持原逻辑（base64）
                                const base64 = await new Promise<string>((resolve) => {
                                  const reader = new FileReader();
                                  reader.onloadend = () => resolve(reader.result as string);
                                  reader.readAsDataURL(file);
                                });
                                onGlobalInputChange(item.nodeId, item.port.id, base64);
                              }
                            } catch (err) {
                              console.error('[NodeConfigPanel] Error uploading file:', err);
                            } finally {
                              setUploadingNodes(prev => {
                                const next = new Set(prev);
                                next.delete(item.nodeId);
                                return next;
                              });
                            }
                          }}
                        />
                      </label>
                      {uploadingNodes.has(item.nodeId) ? (
                        <div className="flex items-center gap-1 text-[9px] text-[#90dce1]">
                          <RefreshCw size={8} className="animate-spin" />
                          <span>{lang === 'zh' ? '上传中...' : 'Uploading...'}</span>
                        </div>
                      ) : (
                        <>
                          {item.isSourceNode ? (
                            workflow.nodes.find(n => n.id === item.nodeId)?.data.value && (
                              <div className="text-[9px] text-slate-400">
                                {lang === 'zh' ? '已上传文件' : 'File uploaded'}
                              </div>
                            )
                          ) : (
                            workflow.globalInputs[`${item.nodeId}-${item.port.id}`] && (
                              <div className="text-[9px] text-slate-400">
                                {lang === 'zh' ? '已上传文件' : 'File uploaded'}
                              </div>
                            )
                          )}
                        </>
                      )}
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </aside>
  );
};
