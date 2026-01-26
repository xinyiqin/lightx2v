import React, { useRef } from 'react';
import {
  RefreshCw,
  Trash2,
  CheckCircle2,
  AlertCircle,
  PlayCircle as PlayIcon,
  FastForward,
  X,
  Plus,
  Upload,
  Volume2,
  Video as VideoIcon,
  Edit3,
  Maximize2,
  Play,
  ChevronDown,
  CheckCircle2 as CheckCircle,
  Globe
} from 'lucide-react';
import { WorkflowNode, WorkflowState, NodeStatus, DataType, Port, ToolDefinition } from '../../../types';
import { TOOLS } from '../../../constants';
import { useTranslation, Language } from '../../i18n/useTranslation';
import { getIcon } from '../../utils/icons';
import { formatTime } from '../../utils/format';
import { screenToWorld, ViewState } from '../../utils/canvas';
import { getAssetPath } from '../../utils/assetPath';
import { uploadNodeInputFile } from '../../utils/workflowFileManager';

interface NodeProps {
  node: WorkflowNode;
  workflow: WorkflowState;
  isSelected: boolean;
  activeOutputs: Record<string, any>;
  sourceOutputs: Record<string, any>;
  nodeHeight: number;
  onSelect: (nodeId: string) => void;
  onDragStart: (nodeId: string, offsetX: number, offsetY: number) => void;
  onDrag: (nodeId: string, x: number, y: number) => void;
  onDragEnd: () => void;
  getNodeOutputs: (node: WorkflowNode) => Port[];
  canvasRef: React.RefObject<HTMLDivElement>;
  view: ViewState;
  lang: Language;
  // Additional props for node interactions
  showReplaceMenu?: string | null;
  showOutputQuickAdd?: { nodeId: string; portId: string } | null;
  showModelSelect?: string | null;
  showVoiceSelect?: string | null;
  lightX2VVoiceList?: { voices?: any[]; emotions?: string[]; languages?: any[] } | null;
  cloneVoiceList?: any[];
  onUpdateNodeData: (nodeId: string, key: string, value: any) => void;
  onDeleteNode: (nodeId: string) => void;
  onReplaceNode: (nodeId: string, newToolId: string) => void;
  onRunWorkflow: (nodeId?: string, runThisOnly?: boolean) => void;
  onSetReplaceMenu: (nodeId: string | null) => void;
  onSetOutputQuickAdd: (value: { nodeId: string; portId: string } | null) => void;
  onSetModelSelect: (nodeId: string | null) => void;
  onSetVoiceSelect: (nodeId: string | null) => void;
  onSetExpandedOutput: (value: { nodeId: string; fieldId?: string } | null) => void;
  onSetShowAudioEditor: (nodeId: string | null) => void;
  onSetConnecting: (value: {
    nodeId: string;
    portId: string;
    type: DataType;
    direction: 'in' | 'out';
    startX: number;
    startY: number;
  } | null) => void;
  onAddConnection: (connection: {
    id: string;
    sourceNodeId: string;
    sourcePortId: string;
    targetNodeId: string;
    targetPortId: string;
  }) => void;
  onClearSelectedRunId: () => void;
  getReplaceableTools: (nodeId: string) => ToolDefinition[];
  getCompatibleToolsForOutput: (outputType: DataType) => ToolDefinition[];
  quickAddInput: (node: WorkflowNode, port: Port) => void;
  quickAddOutput: (node: WorkflowNode, port: Port, toolId: string) => void;
  connecting: {
    nodeId: string;
    portId: string;
    type: DataType;
    direction: 'in' | 'out';
    startX: number;
    startY: number;
  } | null;
  onNodeHeightChange?: (nodeId: string, height: number) => void;
}

export const Node: React.FC<NodeProps> = ({
  node,
  workflow,
  isSelected,
  activeOutputs,
  sourceOutputs,
  nodeHeight,
  onSelect,
  onDragStart,
  onDrag,
  onDragEnd,
  getNodeOutputs,
  canvasRef,
  view,
  lang,
  showReplaceMenu,
  showOutputQuickAdd,
  showModelSelect,
  showVoiceSelect,
  lightX2VVoiceList,
  cloneVoiceList = [],
  onUpdateNodeData,
  onDeleteNode,
  onReplaceNode,
  onRunWorkflow,
  onSetReplaceMenu,
  onSetOutputQuickAdd,
  onSetModelSelect,
  onSetVoiceSelect,
  onSetExpandedOutput,
  onSetShowAudioEditor,
  onSetConnecting,
  onAddConnection,
  onClearSelectedRunId,
  getReplaceableTools,
  getCompatibleToolsForOutput,
  quickAddInput,
  quickAddOutput,
  connecting,
  onNodeHeightChange
}) => {
  const { t } = useTranslation(lang);
  const nodeRef = useRef<HTMLDivElement>(null);
  const lastHeightRef = useRef<number>(0);

  const tool = TOOLS.find((t) => t.id === node.toolId);
  if (!tool) return null;

  const outputs = getNodeOutputs(node);

  // Measure and report node height when it changes
  React.useEffect(() => {
    if (nodeRef.current && onNodeHeightChange) {
      const currentHeight = nodeRef.current.offsetHeight;
      if (currentHeight !== lastHeightRef.current) {
        lastHeightRef.current = currentHeight;
        onNodeHeightChange(node.id, currentHeight);
      }
    }
  }, [node.id, onNodeHeightChange, node.data, outputs.length, activeOutputs[node.id], node.status]);
  const nodeResultRaw = sourceOutputs[node.id] || (tool.category === 'Input' ? node.data.value : null);
  // Extract actual value from reference objects (for history outputs or saved outputs)
  const nodeResult = nodeResultRaw && typeof nodeResultRaw === 'object' && !Array.isArray(nodeResultRaw) && nodeResultRaw.type === 'url'
    ? nodeResultRaw.data  // Extract URL from { type: 'url', data: '...' }
    : nodeResultRaw && typeof nodeResultRaw === 'object' && !Array.isArray(nodeResultRaw) && nodeResultRaw.type === 'text'
    ? nodeResultRaw.data  // Extract text from { type: 'text', data: '...' }
    : nodeResultRaw;  // Use as-is for strings, arrays, or other types
  const firstOutputType = outputs[0]?.type || DataType.TEXT;

  const durationText =
    node.status === NodeStatus.RUNNING
      ? ((performance.now() - (node.startTime || performance.now())) / 1000).toFixed(1) + 's'
      : formatTime(node.executionTime);

  const isInputNode = tool.category === 'Input';
  const hasData =
    (isInputNode && node.data.value && (Array.isArray(node.data.value) ? node.data.value.length > 0 : true)) ||
    (!isInputNode && sourceOutputs[node.id]);
  const shouldShowPreview = hasData && node.toolId !== 'text-input';

  const handleNodeClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    onClearSelectedRunId();
    onSelect(node.id);
  };

  const handleNodeMouseDown = (e: React.MouseEvent) => {
    if ((e.target as HTMLElement).closest('input, textarea, button, label')) return;
    const rect = canvasRef.current?.getBoundingClientRect();
    if (!rect) return;
    const world = screenToWorld(
      e.clientX - rect.left,
      e.clientY - rect.top,
      view,
      rect
    );
    onDragStart(node.id, world.x - node.x, world.y - node.y);
  };

  // Note: onDrag is handled by parent component (Canvas) via mouse move events
  // The parent will call onDrag when draggingNode is set

  const handlePortMouseDown = (port: Port, direction: 'in' | 'out') => (e: React.MouseEvent) => {
    e.stopPropagation();
    
    if (direction === 'out') {
      // For output ports, calculate position from node bottom (same as Connection component)
      const outputPortIndex = outputs.findIndex((p) => p.id === port.id);
      const nodeBottomY = node.y + nodeHeight;
      const startX = node.x + 224; // Output port center X
      const startY = nodeBottomY - ((outputs.length - 1 - outputPortIndex) * 30) - 24; // Output port center Y
      onSetConnecting({
        nodeId: node.id,
        portId: port.id,
        type: port.type,
        direction,
        startX,
        startY
      });
    } else {
      // For input ports, calculate position from node top (same as Connection component)
      const tool = TOOLS.find((t) => t.id === node.toolId);
      const inputPortIndex = tool?.inputs.findIndex((p) => p.id === port.id) ?? -1;
      const startX = node.x; // Input port center X
      const startY = node.y + 71 + (inputPortIndex * 30); // Input port center Y
      onSetConnecting({
        nodeId: node.id,
        portId: port.id,
        type: port.type,
        direction,
        startX,
        startY
      });
    }
  };

  const handlePortMouseUp = (port: Port, direction: 'in' | 'out') => () => {
    if (connecting && connecting.direction !== direction && connecting.type === port.type) {
      onClearSelectedRunId();
      if (direction === 'in') {
        onAddConnection({
          id: `conn-${Date.now()}`,
          sourceNodeId: connecting.nodeId,
          sourcePortId: connecting.portId,
          targetNodeId: node.id,
          targetPortId: port.id
        });
      } else {
        onAddConnection({
          id: `conn-${Date.now()}`,
          sourceNodeId: node.id,
          sourcePortId: port.id,
          targetNodeId: connecting.nodeId,
          targetPortId: connecting.portId
        });
      }
      onSetConnecting(null);
    }
  };

  const [isUploading, setIsUploading] = React.useState(false);
  
  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>, isMultiple: boolean = false) => {
    const files = Array.from(e.target.files || []);
    if (files.length === 0) return;
    
    // 需要 workflow.id 才能上传
    if (!workflow.id || (!workflow.id.startsWith('workflow-') && !workflow.id.match(/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i))) {
      console.error('[Node] Cannot upload file: workflow ID is not available');
      return;
    }
    
    setIsUploading(true);
    
    try {
      const tool = TOOLS.find(t => t.id === node.toolId);
      if (!tool || tool.category !== 'Input') {
        console.error('[Node] Cannot upload file: node is not an input node');
        return;
      }
      
      const outputPort = tool.outputs[0];
      if (!outputPort) {
        console.error('[Node] Cannot upload file: output port not found');
        return;
      }
      
      // 上传所有文件
      const uploadPromises = files.map((file: File) =>
        uploadNodeInputFile(workflow.id!, node.id, outputPort.id, file)
          .then(result => {
            if (result) {
              return result.file_url;
            }
            return null;
          })
          .catch(err => {
            console.error(`[Node] Error uploading file:`, err);
            return null;
          })
      );
      
      const fileUrls = await Promise.all(uploadPromises);
      const validUrls = fileUrls.filter((url: string | null) => url !== null);
      
      if (validUrls.length > 0) {
        // 更新 node.data.value，始终使用数组格式
        const currentValue = node.data.value || [];
        const existingUrls = Array.isArray(currentValue) ? currentValue : [currentValue].filter(Boolean);
        const newValue = isMultiple ? [...existingUrls, ...validUrls] : validUrls;
        onUpdateNodeData(node.id, 'value', newValue);
      }
    } catch (err) {
      console.error('[Node] Error uploading files:', err);
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div
      ref={nodeRef}
      className={`node-element absolute w-56 bg-slate-900 border rounded-3xl shadow-2xl transition-all z-10 group ${
        isSelected ? 'border-[#90dce1] ring-8 ring-#90dce1/10' : 'border-slate-800'
      }`}
      style={{ left: node.x, top: node.y }}
      onClick={handleNodeClick}
      onMouseDown={handleNodeMouseDown}
    >
      {/* Replace and Delete Menu */}
      {isSelected && (
        <div className="absolute -top-14 left-1/2 -translate-x-1/2 flex items-center gap-2 z-20 replace-menu-container">
          <div className="relative">
            <button
              onMouseDown={(e) => e.stopPropagation()}
              onClick={(e) => {
                e.stopPropagation();
                onSetReplaceMenu(showReplaceMenu === node.id ? null : node.id);
              }}
              className="p-2 bg-[#90dce1] text-white rounded-full shadow-lg hover:bg-[#90dce1] transition-all active:scale-90"
              title={lang === 'zh' ? '替换节点' : 'Replace Node'}
            >
              <RefreshCw size={16} />
            </button>
            {showReplaceMenu === node.id && (
              <div className="absolute top-full left-1/2 -translate-x-1/2 mt-2 w-48 bg-slate-800 border border-slate-700 rounded-xl shadow-2xl z-30 max-h-64 overflow-y-auto custom-scrollbar">
                {getReplaceableTools(node.id).length > 0 ? (
                  getReplaceableTools(node.id).map((replaceTool) => (
                    <button
                      key={replaceTool.id}
                      onClick={(e) => {
                        e.stopPropagation();
                        onReplaceNode(node.id, replaceTool.id);
                      }}
                      className="w-full px-4 py-2 text-left text-xs text-slate-300 hover:bg-[#90dce1]/20 hover:text-white transition-colors flex items-center gap-2"
                    >
                      <div className="p-1 rounded bg-slate-700">
                        {React.createElement(getIcon(replaceTool.icon), { size: 12 })}
                      </div>
                      <span>{lang === 'zh' ? replaceTool.name_zh : replaceTool.name}</span>
                    </button>
                  ))
                ) : (
                  <div className="px-4 py-3 text-xs text-slate-500 text-center">
                    {lang === 'zh' ? '没有可替换的工具' : 'No replaceable tools'}
                  </div>
                )}
              </div>
            )}
          </div>
          <button
            onMouseDown={(e) => e.stopPropagation()}
            onClick={(e) => {
              e.stopPropagation();
              onDeleteNode(node.id);
              onSetReplaceMenu(null);
            }}
            className="p-2 bg-red-500 text-white rounded-full shadow-lg hover:bg-red-600 transition-all active:scale-90"
          >
            <Trash2 size={16} />
          </button>
        </div>
      )}

      {/* Node Header */}
      <div
        className={`px-4 py-3 border-b flex items-center justify-between bg-slate-800/40 rounded-t-3xl ${
          node.status === NodeStatus.RUNNING ? 'animate-pulse bg-[#90dce1]/10 border-[#90dce1]/20' : ''
        }`}
      >
        <div className="flex items-center gap-2 truncate flex-1 min-w-0">
          <div
            className={`p-1.5 rounded-lg shrink-0 ${
              node.status === NodeStatus.RUNNING ? 'bg-[#90dce1] text-white' : 'bg-slate-800 text-slate-400'
            }`}
          >
            {React.createElement(getIcon(tool.icon), { size: 10 })}
          </div>
          <span className="text-[10px] font-black uppercase truncate tracking-widest">
            {lang === 'zh' ? tool.name_zh : tool.name}
          </span>
        </div>
        <div className="flex items-center gap-1.5 shrink-0">
          {(node.status === NodeStatus.RUNNING || node.executionTime !== undefined) && (
            <span
              className={`text-[8px] font-bold ${
                node.status === NodeStatus.RUNNING ? 'text-[#90dce1]' : 'text-slate-500'
              }`}
            >
              {durationText}
            </span>
          )}

          {!isInputNode && (
            <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
              <button
                title={t('run_this_only')}
                onClick={(e) => {
                  e.stopPropagation();
                  onRunWorkflow(node.id, true);
                }}
                className="p-1 text-slate-400 hover:text-[#90dce1] transition-colors"
              >
                <PlayIcon size={12} />
              </button>
              <button
                title={t('run_from_here')}
                onClick={(e) => {
                  e.stopPropagation();
                  onRunWorkflow(node.id, false);
                }}
                className="p-1 text-slate-400 hover:text-emerald-400 transition-colors"
              >
                <FastForward size={12} />
              </button>
            </div>
          )}

          {node.status === NodeStatus.SUCCESS && <CheckCircle2 size={12} className="text-emerald-500" />}
          {node.status === NodeStatus.ERROR && <AlertCircle size={12} className="text-red-500" />}
        </div>
      </div>

      {/* Node Content */}
      <div className="p-4 space-y-4">
        {/* Input Node Content */}
        {isInputNode && (
          <div onMouseDown={(e) => e.stopPropagation()} className="space-y-3">
            {node.toolId === 'text-input' && (
              <textarea
                value={node.data.value || ''}
                onChange={(e) => onUpdateNodeData(node.id, 'value', e.target.value)}
                className="w-full h-24 bg-slate-950/50 border border-slate-800 rounded-xl p-2 text-[10px] resize-none focus:ring-1 focus:ring-#90dce1 transition-all text-slate-300 custom-scrollbar"
                placeholder="Type here..."
              />
            )}
            {node.toolId === 'image-input' && (
              <div className="space-y-2">
                <div className="flex flex-wrap gap-1.5">
                  {/* Use activeOutputs if available (loaded from URL as data URL), otherwise use node.data.value */}
                  {((activeOutputs[node.id] || node.data.value) || []).map((img: string, i: number) => {
                    // activeOutputs should contain data URLs (loaded via getWorkflowFileByFileId)
                    // node.data.value may contain file paths (like /api/v1/workflow/...)
                    // If it's a file path, we should have already loaded it to activeOutputs
                    // But if not, we need to handle it
                    const imgSrc = img.startsWith('/api/v1/workflow/') 
                      ? img  // This shouldn't happen if loading worked correctly, but handle it
                      : getAssetPath(img);
                    return (
                      <div key={i} className="relative w-8 h-8 group/img">
                        <img 
                          src={imgSrc} 
                          className="w-full h-full object-cover rounded border border-slate-700" 
                          alt={`Image ${i + 1}`}
                          onError={(e) => {
                            // If image fails to load and it's a workflow file path, try loading it
                            const target = e.target as HTMLImageElement;
                            if (img.startsWith('/api/v1/workflow/')) {
                              // Extract file_id and load via API
                              const match = img.match(/\/api\/v1\/workflow\/([^/]+)\/file\/(.+)$/);
                              if (match) {
                                const [, workflowId, fileId] = match;
                                // Import getWorkflowFileByFileId dynamically
                                import('../../utils/workflowFileManager').then(({ getWorkflowFileByFileId }) => {
                                  getWorkflowFileByFileId(workflowId, fileId).then(dataUrl => {
                                    if (dataUrl) {
                                      target.src = dataUrl;
                                    }
                                  });
                                });
                              }
                            }
                          }}
                        />
                        <button
                          onClick={() => {
                            const currentValue = activeOutputs[node.id] || node.data.value || [];
                            const next = Array.isArray(currentValue) 
                              ? currentValue.filter((_: any, idx: number) => idx !== i)
                              : [];
                            onUpdateNodeData(node.id, 'value', next);
                          }}
                          className="absolute -top-1 -right-1 p-0.5 bg-red-500 rounded-full opacity-0 group-hover/img:opacity-100 transition-opacity"
                        >
                          <X size={6} />
                        </button>
                      </div>
                    );
                  })}
                  <label className="w-8 h-8 flex items-center justify-center border border-dashed border-slate-700 rounded cursor-pointer hover:border-[#90dce1] transition-colors">
                    <Plus size={10} className="text-slate-500" />
                    <input
                      type="file"
                      multiple
                      accept="image/*"
                      className="hidden"
                      onChange={(e) => handleFileUpload(e, true)}
                    />
                  </label>
                </div>
              </div>
            )}
            {(node.toolId === 'audio-input' || node.toolId === 'video-input') && (
              <div className="space-y-2">
                {node.data.value ? (
                  <div className="flex items-center justify-between p-2 bg-slate-950/50 rounded-xl border border-slate-800">
                    <div className="flex items-center gap-2 overflow-hidden">
                      {node.toolId === 'audio-input' ? (
                        <Volume2 size={12} className="text-[#90dce1] shrink-0" />
                      ) : (
                        <VideoIcon size={12} className="text-[#90dce1] shrink-0" />
                      )}
                      <span className="text-[8px] text-slate-400 truncate">Media File</span>
                    </div>
                    <div className="flex items-center gap-1">
                      {node.toolId === 'audio-input' && (
                        <button
                          onClick={() => onSetShowAudioEditor(node.id)}
                          className="p-1 text-slate-600 hover:text-[#90dce1] transition-colors"
                          title={lang === 'zh' ? '编辑音频' : 'Edit Audio'}
                        >
                          <Edit3 size={10} />
                        </button>
                      )}
                      <button
                        onClick={() => onUpdateNodeData(node.id, 'value', null)}
                        className="p-1 text-slate-600 hover:text-red-400"
                      >
                        <X size={10} />
                      </button>
                    </div>
                  </div>
                ) : (
                  <label className="flex items-center justify-center gap-2 w-full py-3 border border-dashed border-slate-700 rounded-xl cursor-pointer hover:border-[#90dce1] hover:bg-[#90dce1]/5 transition-all disabled:opacity-50 disabled:cursor-not-allowed">
                    {isUploading ? (
                      <>
                        <RefreshCw size={12} className="text-[#90dce1] animate-spin" />
                        <span className="text-[9px] font-black text-[#90dce1] uppercase">
                          {lang === 'zh' ? '上传中...' : 'Uploading...'}
                        </span>
                      </>
                    ) : (
                      <>
                        <Upload size={12} className="text-slate-500" />
                        <span className="text-[9px] font-black text-slate-500 uppercase">
                          {lang === 'zh' ? '上传' : 'Upload'}
                        </span>
                      </>
                    )}
                    <input
                      type="file"
                      accept={node.toolId === 'audio-input' ? 'audio/*' : 'video/*'}
                      className="hidden"
                      disabled={isUploading}
                      onChange={(e) => handleFileUpload(e, false)}
                    />
                  </label>
                )}
              </div>
            )}
          </div>
        )}

        {/* Error Display */}
        {node.status === NodeStatus.ERROR && (
          <div className="bg-red-500/10 border border-red-500/20 p-2 rounded-xl text-[8px] text-red-400 leading-tight">
            <span className="font-bold uppercase mb-1 block">{t('execution_error')}</span>
            {node.error}
          </div>
        )}

        {/* Input Ports */}
        {tool.inputs.map((p) => {
          const isConnected = workflow.connections.some(
            (c) => c.targetNodeId === node.id && c.targetPortId === p.id
          );
          return (
            <div key={p.id} className="flex items-center gap-2 text-[9px] font-bold text-slate-500 relative group/port">
              {!isConnected && (
                <button
                  onMouseDown={(e) => e.stopPropagation()}
                  onClick={(e) => {
                    e.stopPropagation();
                    quickAddInput(node, p);
                  }}
                  className="opacity-0 group-hover/port:opacity-100 transition-opacity p-1 bg-[#90dce1] text-white rounded-lg absolute -left-12 z-20 shadow-xl hover:bg-[#90dce1] active:scale-90 flex items-center justify-center"
                  title={t('quick_add_source')}
                >
                  <Plus size={14} />
                </button>
              )}
              <div
                className="port w-3 h-3 rounded-full bg-slate-800 border-2 border-slate-950 absolute -left-[24px] cursor-crosshair hover:bg-[#90dce1] transition-colors"
                onMouseDown={handlePortMouseDown(p, 'in')}
                onMouseUp={handlePortMouseUp(p, 'in')}
              />
              <span className="truncate">{p.label}</span>
            </div>
          );
        })}

        {/* Output Ports */}
        {outputs.map((p) => {
          const isConnected = workflow.connections.some(
            (c) => c.sourceNodeId === node.id && c.sourcePortId === p.id
          );
          const compatibleTools = getCompatibleToolsForOutput(p.type);
          const showMenu = showOutputQuickAdd?.nodeId === node.id && showOutputQuickAdd?.portId === p.id;
          return (
            <div key={p.id} className="flex items-center justify-end gap-2 text-[9px] font-bold text-slate-500 relative group/port">
              {!isConnected && compatibleTools.length > 0 && (
                <button
                  onMouseDown={(e) => e.stopPropagation()}
                  onClick={(e) => {
                    e.stopPropagation();
                    onSetOutputQuickAdd(showMenu ? null : { nodeId: node.id, portId: p.id });
                  }}
                  className="opacity-0 group-hover/port:opacity-100 transition-opacity p-1 bg-[#90dce1] text-white rounded-lg absolute -right-12 z-20 shadow-xl hover:bg-[#90dce1] active:scale-90 flex items-center justify-center"
                  title={lang === 'zh' ? '快速添加节点' : 'Quick Add Node'}
                >
                  <Plus size={14} />
                </button>
              )}
              {showMenu && (
                <div
                  className="absolute top-0 left-full ml-2 w-48 bg-slate-800 border border-slate-700 rounded-xl shadow-2xl z-30 max-h-64 overflow-y-auto custom-scrollbar output-quick-add-menu"
                  onClick={(e) => e.stopPropagation()}
                >
                  {compatibleTools.map((tool) => (
                    <button
                      key={tool.id}
                      onClick={(e) => {
                        e.stopPropagation();
                        quickAddOutput(node, p, tool.id);
                      }}
                      className="w-full px-4 py-2 text-left text-xs text-slate-300 hover:bg-[#90dce1]/20 hover:text-white transition-colors flex items-center gap-2"
                    >
                      <div className="p-1 rounded bg-slate-700">
                        {React.createElement(getIcon(tool.icon), { size: 12 })}
                      </div>
                      <span>{lang === 'zh' ? tool.name_zh : tool.name}</span>
                    </button>
                  ))}
                </div>
              )}
              <span className="truncate">{p.label}</span>
              <div
                className="port w-3 h-3 rounded-full bg-slate-800 border-2 border-slate-950 absolute -right-[24px] cursor-crosshair hover:bg-[#90dce1] transition-colors"
                onMouseDown={handlePortMouseDown(p, 'out')}
                onMouseUp={handlePortMouseUp(p, 'out')}
              />
            </div>
          );
        })}
      </div>

      {/* Preview */}
      {shouldShowPreview && (
        <div
          onClick={(e) => {
            e.stopPropagation();
            onSetExpandedOutput({ nodeId: node.id });
          }}
          onMouseDown={(e) => e.stopPropagation()}
          className="absolute -right-36 top-0 max-w-32 max-h-32 bg-slate-800/95 rounded-2xl border border-slate-700 shadow-2xl overflow-hidden cursor-pointer hover:scale-110 hover:border-[#90dce1] transition-all z-30 group/thumb flex items-center justify-center"
        >
          {firstOutputType === DataType.IMAGE ? (
            <img
              src={getAssetPath(
                Array.isArray(nodeResult) 
                  ? (nodeResult.length > 0 ? nodeResult[0] : '') 
                  : (nodeResult || '')
              )}
              className="max-w-full max-h-full w-auto h-auto object-contain"
              alt="Preview"
              onError={(e) => {
                // 如果图片加载失败，尝试使用原始值（可能是 base64）
                const target = e.currentTarget;
                const originalSrc = Array.isArray(nodeResult) 
                  ? (nodeResult.length > 0 ? nodeResult[0] : '') 
                  : (nodeResult || '');
                if (target.src !== originalSrc) {
                  target.src = originalSrc;
                }
              }}
            />
          ) : firstOutputType === DataType.TEXT ? (
            <div className="p-3 text-[8px] text-slate-300 overflow-hidden leading-snug font-medium selection:bg-transparent w-full h-full">
              {typeof nodeResult === 'object'
                ? JSON.stringify(nodeResult).slice(0, 100)
                : nodeResult.toString().slice(0, 100)}
              ...
            </div>
          ) : firstOutputType === DataType.AUDIO ? (
            <div className="w-full h-full flex flex-col items-center justify-center text-[#90dce1] bg-[#90dce1]/5 min-w-32 min-h-24">
              <Volume2 size={32} className="mb-1" />
              <div className="w-20 h-1.5 bg-[#90dce1]/20 rounded-full overflow-hidden">
                <div className="w-1/2 h-full bg-[#90dce1] animate-pulse"></div>
              </div>
            </div>
          ) : (
            <div className="w-full h-full relative bg-black group/video min-w-32 min-h-24 flex items-center justify-center">
              <video
                src={getAssetPath(
                  Array.isArray(nodeResult) 
                    ? (nodeResult.length > 0 ? nodeResult[0] : '') 
                    : (nodeResult || '')
                )}
                className="max-w-full max-h-full w-auto h-auto object-contain opacity-60 group-hover/thumb:opacity-100 transition-opacity"
                muted
                preload="none"
                loading="lazy"
                onMouseOver={(e) => {
                  const video = e.currentTarget;
                  if (video.readyState < 2) {
                    video.load();
                  }
                  video.play().catch(() => {});
                }}
                onMouseOut={(e) => {
                  e.currentTarget.pause();
                  e.currentTarget.currentTime = 0;
                }}
              />
              <div className="absolute inset-0 flex items-center justify-center text-white pointer-events-none group-hover/thumb:scale-125 transition-transform">
                <Play size={24} className="drop-shadow-lg" fill="currentColor" />
              </div>
            </div>
          )}
          <div className="absolute inset-x-0 bottom-0 h-6 bg-gradient-to-t from-slate-950/90 to-transparent flex items-center px-2.5">
            <span className="text-[7px] font-black uppercase text-white/80 tracking-widest flex items-center gap-1">
              <Maximize2 size={10} /> {t('inspect_result')}
            </span>
          </div>
        </div>
      )}

      {/* Model and Voice Selectors */}
      <div className="absolute top-full left-0 mt-2 flex items-center gap-2 z-20">
        {/* Model selector */}
        {tool.models && tool.models.length > 0 && (
          <div className="relative model-select-container">
            <button
              onMouseDown={(e) => e.stopPropagation()}
              onClick={(e) => {
                e.stopPropagation();
                onSetModelSelect(showModelSelect === node.id ? null : node.id);
                onSetVoiceSelect(null);
              }}
              className="px-2 py-1 text-[9px] font-bold bg-slate-700 hover:bg-slate-600 text-slate-300 rounded-md transition-colors flex items-center gap-1 shadow-lg"
              title={lang === 'zh' ? '选择模型' : 'Select Model'}
            >
              <span className="truncate max-w-[80px]">
                {tool.models.find((m) => m.id === node.data.model)?.name || tool.models[0]?.name || 'Model'}
              </span>
              <ChevronDown size={10} className={showModelSelect === node.id ? 'rotate-180' : ''} />
            </button>
            {showModelSelect === node.id && (
              <div
                className="absolute top-full left-0 mt-1 w-48 bg-slate-800 border border-slate-700 rounded-xl shadow-2xl z-30 max-h-64 overflow-y-auto custom-scrollbar"
                onClick={(e) => e.stopPropagation()}
              >
                {tool.models.map((model) => (
                  <button
                    key={model.id}
                    onClick={(e) => {
                      e.stopPropagation();
                      onUpdateNodeData(node.id, 'model', model.id);
                      onSetModelSelect(null);
                    }}
                    className={`w-full px-4 py-2 text-left text-xs transition-colors flex items-center gap-2 ${
                      node.data.model === model.id
                        ? 'bg-[#90dce1]/20 text-white'
                        : 'text-slate-300 hover:bg-[#90dce1]/20 hover:text-white'
                    }`}
                  >
                    <span>{model.name}</span>
                    {node.data.model === model.id && <CheckCircle size={12} className="ml-auto" />}
                  </button>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Web search toggle for DeepSeek and Doubao models */}
        {node.toolId === 'text-generation' && 
          (node.data.model?.startsWith('deepseek-') || node.data.model?.startsWith('doubao-')) && (
          <button
            onMouseDown={(e) => e.stopPropagation()}
            onClick={(e) => {
              e.stopPropagation();
              onUpdateNodeData(node.id, 'useSearch', !node.data.useSearch);
            }}
            className={`px-2 py-1 text-[9px] font-bold rounded-md transition-colors flex items-center gap-1 shadow-lg ${
              node.data.useSearch
                ? 'bg-[#90dce1]/80 hover:bg-[#90dce1] text-white'
                : 'bg-slate-700 hover:bg-slate-600 text-slate-300'
            }`}
            title={lang === 'zh' ? '联网搜索' : 'Web Search'}
          >
            <Globe size={10} />
          </button>
        )}

        {/* Voice selector for TTS nodes */}
        {node.toolId === 'tts' &&
          (node.data.model === 'lightx2v' || node.data.model?.startsWith('lightx2v')) &&
          lightX2VVoiceList?.voices && (
            <div className="relative voice-select-container">
              <button
                onMouseDown={(e) => e.stopPropagation()}
                onClick={(e) => {
                  e.stopPropagation();
                  onSetVoiceSelect(showVoiceSelect === node.id ? null : node.id);
                  onSetModelSelect(null);
                }}
                className="px-2 py-1 text-[9px] font-bold bg-slate-700 hover:bg-slate-600 text-slate-300 rounded-md transition-colors flex items-center gap-1 shadow-lg"
                title={lang === 'zh' ? '选择音色' : 'Select Voice'}
              >
                <span className="truncate max-w-[80px]">
                  {lightX2VVoiceList.voices.find((v: any) => v.voice_type === node.data.voiceType)?.name ||
                    node.data.voiceType ||
                    'Voice'}
                </span>
                <ChevronDown size={10} className={showVoiceSelect === node.id ? 'rotate-180' : ''} />
              </button>
              {showVoiceSelect === node.id && (
                <div
                  className="absolute top-full left-0 mt-1 w-56 bg-slate-800 border border-slate-700 rounded-xl shadow-2xl z-30 max-h-64 overflow-y-auto custom-scrollbar"
                  onClick={(e) => e.stopPropagation()}
                >
                  {lightX2VVoiceList.voices.map((voice: any) => (
                    <button
                      key={voice.voice_type}
                      onClick={(e) => {
                        e.stopPropagation();
                        onUpdateNodeData(node.id, 'voiceType', voice.voice_type);
                        if (voice.resource_id) {
                          onUpdateNodeData(node.id, 'resourceId', voice.resource_id);
                        }
                        onSetVoiceSelect(null);
                      }}
                      className={`w-full px-4 py-2 text-left text-xs transition-colors flex items-center gap-2 ${
                        node.data.voiceType === voice.voice_type
                          ? 'bg-[#90dce1]/20 text-white'
                          : 'text-slate-300 hover:bg-[#90dce1]/20 hover:text-white'
                      }`}
                    >
                      <span className="truncate">{voice.name || voice.voice_type}</span>
                      {node.data.voiceType === voice.voice_type && <CheckCircle size={12} className="ml-auto" />}
                    </button>
                  ))}
                </div>
              )}
            </div>
          )}

        {/* Voice selector for voice clone nodes */}
        {node.toolId === 'lightx2v-voice-clone' && cloneVoiceList.length > 0 && (
          <div className="relative voice-select-container">
            <button
              onMouseDown={(e) => e.stopPropagation()}
              onClick={(e) => {
                e.stopPropagation();
                onSetVoiceSelect(showVoiceSelect === node.id ? null : node.id);
                onSetModelSelect(null);
              }}
              className="px-2 py-1 text-[9px] font-bold bg-slate-700 hover:bg-slate-600 text-slate-300 rounded-md transition-colors flex items-center gap-1 shadow-lg"
              title={lang === 'zh' ? '选择克隆音色' : 'Select Clone Voice'}
            >
              <span className="truncate max-w-[80px]">
                {cloneVoiceList.find((v) => v.speaker_id === node.data.speakerId)?.name ||
                  node.data.speakerId ||
                  'Voice'}
              </span>
              <ChevronDown size={10} className={showVoiceSelect === node.id ? 'rotate-180' : ''} />
            </button>
            {showVoiceSelect === node.id && (
              <div
                className="absolute top-full left-0 mt-1 w-56 bg-slate-800 border border-slate-700 rounded-xl shadow-2xl z-30 max-h-64 overflow-y-auto custom-scrollbar"
                onClick={(e) => e.stopPropagation()}
              >
                {cloneVoiceList.map((voice: any) => (
                  <button
                    key={voice.speaker_id}
                    onClick={(e) => {
                      e.stopPropagation();
                      onUpdateNodeData(node.id, 'speakerId', voice.speaker_id);
                      onSetVoiceSelect(null);
                    }}
                    className={`w-full px-4 py-2 text-left text-xs transition-colors flex items-center gap-2 ${
                      node.data.speakerId === voice.speaker_id
                        ? 'bg-[#90dce1]/20 text-white'
                        : 'text-slate-300 hover:bg-[#90dce1]/20 hover:text-white'
                    }`}
                  >
                    <span className="truncate">{voice.name || voice.speaker_id}</span>
                    {node.data.speakerId === voice.speaker_id && <CheckCircle size={12} className="ml-auto" />}
                  </button>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

