
import React, { useState, useCallback, useRef, useEffect, useMemo } from 'react';
import {
  Plus, Play, Pause, Save, Trash2, Search, Settings,
  Layers, ChevronRight, AlertCircle, CheckCircle2,
  X, Type, Image as ImageIcon, Volume2, Video as VideoIcon,
  Cpu, Sparkles, AlignLeft, Download, RefreshCw,
  Terminal, MousePointer2, Wand2, Globe, Palette, Clapperboard, UserCircle, UserCog,
  Maximize, ZoomIn, ZoomOut, Zap, MessageSquare, PenTool, FileText, Star, Edit3, Boxes,
  Camera, Mic, Wand, ListPlus, Hash, Info, PlayCircle, FastForward, ArrowUpRight,
  Target, Activity, History, Clock, Maximize2, DownloadCloud, BookOpen, ChevronLeft, ChevronDown, ChevronUp,
  Calendar, LayoutGrid, Sparkle, ToggleLeft, ToggleRight, Timer, PlayCircle as PlayIcon,
  Key, Globe2, Upload, Languages, ShieldCheck, TriangleAlert, SaveAll, Eraser
} from 'lucide-react';
import { TOOLS, updateLightX2VModels } from './constants';
import {
  WorkflowNode, Connection, WorkflowState,
  NodeStatus, DataType, Port, ToolDefinition, GenerationRun
} from './types';
import { geminiText, geminiImage, geminiSpeech, geminiVideo, lightX2VTask, lightX2VTTS, lightX2VVoiceClone, lightX2VVoiceCloneTTS, lightX2VGetVoiceList, lightX2VGetCloneVoiceList, lightX2VGetModelList, deepseekText, doubaoText, ppchatGeminiText } from './services/geminiService';
import { removeGeminiWatermark } from './services/watermarkRemover';
import { PRESET_WORKFLOWS } from './preset_workflow';
import { Editor } from './src/components/editor/Editor';
import { Dashboard } from './src/components/dashboard/Dashboard';
import { ExpandedOutputModal } from './src/components/modals/ExpandedOutputModal';
import { AIGenerateModal } from './src/components/modals/AIGenerateModal';
import { CloneVoiceModal } from './src/components/modals/CloneVoiceModal';
import { AudioEditorModal } from './src/components/modals/AudioEditorModal';
import { useWorkflow } from './src/hooks/useWorkflow';
import { useCanvas } from './src/hooks/useCanvas';
import { useVoiceList } from './src/hooks/useVoiceList';
import { useTranslation } from './src/i18n/useTranslation';
import { useNodeManagement } from './src/hooks/useNodeManagement';
import { useConnectionManagement } from './src/hooks/useConnectionManagement';
import { useModalState } from './src/hooks/useModalState';
import { useResultManagement } from './src/hooks/useResultManagement';
import { useAIGenerateWorkflow } from './src/hooks/useAIGenerateWorkflow';
import { useWorkflowExecution } from './src/hooks/useWorkflowExecution';
import { useUndoRedo } from './src/hooks/useUndoRedo';
import { useAIChatWorkflow } from './src/hooks/useAIChatWorkflow';
import { useWorkflowAutoSave } from './src/hooks/useWorkflowAutoSave';
import { getAccessToken, initLightX2VToken, apiRequest } from './src/utils/apiClient';
import { checkWorkflowOwnership, getCurrentUserId } from './src/utils/workflowUtils';

// --- Main App ---

const App: React.FC = () => {
  const [lang, setLang] = useState<'en' | 'zh'>(() => {
    const saved = localStorage.getItem('omniflow_lang');
    return (saved as any) || 'zh';
  });
  const [currentView, setCurrentView] = useState<'DASHBOARD' | 'EDITOR'>('DASHBOARD');
  const [activeTab, setActiveTab] = useState<'MY' | 'PRESET'>('MY');

  // Use useWorkflow Hook
  const {
    myWorkflows,
    workflow,
    setWorkflow,
    setMyWorkflows,
    saveWorkflowToLocal,
    saveWorkflowToDatabase,
    loadWorkflow,
    loadWorkflows,
    deleteWorkflow: deleteWorkflowFromHook,
    isLoading: isLoadingWorkflows,
    isSaving: isSavingWorkflow
  } = useWorkflow();
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [selectedConnectionId, setSelectedConnectionId] = useState<string | null>(null);
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [activeOutputs, setActiveOutputs] = useState<Record<string, any>>({});

  // Canvas ref
  const canvasRef = useRef<HTMLDivElement>(null);
  const nodeHeightsRef = useRef<Map<string, number>>(new Map());

  // Use useCanvas Hook
  const {
    view,
    setView,
    isPanning,
    isOverNode,
    setIsOverNode,
    draggingNode,
    connecting,
    setConnecting,
    mousePos,
    zoomIn,
    zoomOut,
    resetView,
    handleMouseMove: canvasHandleMouseMove,
    handleMouseDown: canvasHandleMouseDown,
    handleMouseUp: canvasHandleMouseUp,
    handleMouseLeave: canvasHandleMouseLeave,
    handleWheel: canvasHandleWheel,
    handleNodeDragStart: canvasHandleNodeDragStart,
    handleNodeDrag: canvasHandleNodeDrag,
    handleNodeDragEnd: canvasHandleNodeDragEnd,
    screenToWorldCoords
  } = useCanvas(workflow, canvasRef);

  // Use useModalState Hook
  const modalState = useModalState();

  // Use useUndoRedo Hook
  const undoRedo = useUndoRedo({
    workflow,
    setWorkflow
  });

  const [ticker, setTicker] = useState(0);
  const [validationErrors, setValidationErrors] = useState<{ message: string; type: 'ENV' | 'INPUT' }[]>([]);
  const [globalError, setGlobalError] = useState<{ message: string; details?: string } | null>(null);
  const [aiWorkflowDescription, setAIWorkflowDescription] = useState('');
  const [isGeneratingWorkflow, setIsGeneratingWorkflow] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const isPausedRef = useRef(false);
  const runningTaskIdsRef = useRef<Map<string, string>>(new Map()); // Map<nodeId, taskId> for tracking LightX2V tasks
  const abortControllerRef = useRef<AbortController | null>(null); // AbortController for cancelling tasks

  // Use translation hook from components
  const { t } = useTranslation(lang);

  // 初始化 LIGHTX2V_TOKEN（确保在组件挂载时也初始化）
  useEffect(() => {
    initLightX2VToken();
  }, []);

  // 缓存配置，避免重复计算
  const configCacheRef = useRef<{ config: { url: string; token: string }; key: string } | null>(null);
  const lightX2VConfigRef = useRef<{ url: string; token: string } | null>(null);
  const configKeyRef = useRef<string>('');

  // Helper function to get LightX2V config from shared store and API client
  // 优先使用环境变量 LIGHTX2V_URL，如果没有则根据是否有 apiClient 决定
  const getLightX2VConfig = useCallback((workflow: WorkflowState | null) => {
    // 从主应用获取 sharedStore 和 apiClient
    const sharedStore = (window as any).__SHARED_STORE__;
    const apiClient = (window as any).__API_CLIENT__;

    // 优先检查环境变量 LIGHTX2V_URL
    const DEFAULT_LIGHTX2V_URL = 'https://x2v.light-ai.top';
    const envUrl = process.env.LIGHTX2V_URL;

    // 创建缓存 key（基于配置的关键因素）
    const cacheKey = `${envUrl || 'empty'}:${!!apiClient}`;

    // 如果配置没有变化，返回缓存的配置
    if (configCacheRef.current && configCacheRef.current.key === cacheKey) {
      return configCacheRef.current.config;
    }

    // 获取 token（只在配置变化时调用）
    const token = getAccessToken();

    let config: { url: string; token: string };

    // 如果环境变量明确设置了（不为空且不等于空字符串），优先使用环境变量
    // 即使存在 apiClient，也应该使用环境变量的 URL，因为用户明确指定了要直接访问该 URL
    if (envUrl && envUrl.trim()) {
      const url = envUrl.trim();
      config = {
        url: url,
        token: token
      };
    } else if (apiClient) {
      // 如果环境变量未设置，优先使用 apiClient（如果存在）
      // 这样可以避免跨域问题，因为请求会通过主应用的代理
      config = {
        url: '', // 空字符串表示使用相对路径和主应用的 apiClient
        token: token
      };
    } else {
      // 只有在独立运行模式且没有 apiClient 时，才使用默认值
      // 如果是在主应用中运行（qiankun），应该总是有 apiClient
      // 如果没有 apiClient，说明可能是独立运行，使用默认 URL
      const url = DEFAULT_LIGHTX2V_URL;
      (process.env as any).LIGHTX2V_URL = url;
      config = {
        url: url,
        token: token
      };
    }

    // 缓存配置
    configCacheRef.current = { config, key: cacheKey };

    return config;
  }, []);

  // 获取并更新 LightX2V 模型列表（只在配置真正变化时调用，而不是每次 workflow 变化时）
  // 同时加载本地和云端模型列表
  useEffect(() => {
    const loadLightX2VModels = async () => {
      try {
        const config = getLightX2VConfig(workflow);
        const configKey = `${config.url || 'empty'}:${config.token ? 'hasToken' : 'noToken'}`;

        // 检查云端配置
        const cloudUrl = (process.env.LIGHTX2V_CLOUD_URL || 'https://x2v.light-ai.top').trim();
        const cloudToken = (process.env.LIGHTX2V_CLOUD_TOKEN || '').trim();
        const cloudConfigKey = `${cloudUrl}:${cloudToken ? 'hasToken' : 'noToken'}`;
        const fullConfigKey = `${configKey}:${cloudConfigKey}`;

        // 如果配置没有变化，跳过（这是主要的优化点）
        if (configKeyRef.current === fullConfigKey && lightX2VConfigRef.current) {
          return;
        }

        // 更新缓存的配置
        lightX2VConfigRef.current = config;
        configKeyRef.current = fullConfigKey;

        // 并行加载本地和云端模型列表
        const loadPromises: Promise<Array<{ task: string; model_cls: string; stage: string }>>[] = [];

        // 加载本地模型列表
        if (config.url || config.token) {
          loadPromises.push(
            lightX2VGetModelList(config.url, config.token).catch(err => {
              console.warn('[LightX2V] 获取本地模型列表失败:', err);
              return [];
            })
          );
        } else {
          loadPromises.push(Promise.resolve([]));
        }

        // 加载云端模型列表
        if (cloudUrl && cloudToken) {
          loadPromises.push(
            lightX2VGetModelList(cloudUrl, cloudToken).catch(err => {
              console.warn('[LightX2V] 获取云端模型列表失败:', err);
              return [];
            })
          );
        } else {
          loadPromises.push(Promise.resolve([]));
        }

        const [localModels, cloudModels] = await Promise.all(loadPromises);

        // 更新模型列表（本地模型 + 云端模型带 -cloud 后缀）
        if (localModels.length > 0 || cloudModels.length > 0) {
          updateLightX2VModels(localModels, cloudModels);
          console.log('[LightX2V] 模型列表已更新:', {
            local: localModels.length,
            cloud: cloudModels.length
          });
        }
      } catch (error) {
        console.error('[LightX2V] 获取模型列表失败:', error);
        // 不抛出错误，允许应用继续运行
      }
    };

    loadLightX2VModels();
    // 注意：虽然依赖 workflow，但内部有配置检查，所以实际只在配置变化时才会调用 API
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [getLightX2VConfig]);

  // Use useVoiceList Hook (after getLightX2VConfig is defined)
  const voiceList = useVoiceList(workflow, selectedNodeId, getLightX2VConfig);

  // Use useWorkflowAutoSave Hook
  const autoSaveHook = useWorkflowAutoSave({
    workflow,
    onSave: async (w) => {
      // Auto-save callback (optional)
      console.log('[App] Auto-save completed for workflow:', w.id);
    }
  });

  // Helper function to get node outputs (needed by hooks)
  const getNodeOutputs = (node: WorkflowNode): Port[] => {
    const tool = TOOLS.find(t => t.id === node.toolId);
    if (node.toolId === 'text-generation' && node.data.customOutputs) return node.data.customOutputs.map((o: any) => ({ ...o, type: DataType.TEXT }));
    return tool?.outputs || [];
  };

  // Use useNodeManagement Hook
  const nodeManagement = useNodeManagement({
    workflow,
    setWorkflow,
    selectedNodeId,
    setSelectedNodeId,
    selectedRunId,
    setValidationErrors,
    setActiveOutputs,
    canvasRef,
    screenToWorldCoords,
    view,
    getNodeOutputs,
    lang
  });

  // Use useConnectionManagement Hook
  const connectionManagement = useConnectionManagement({
    workflow,
    setWorkflow,
    selectedConnectionId,
    setSelectedConnectionId,
    selectedRunId,
    setConnecting
  });

  // Use useResultManagement Hook
  const resultManagement = useResultManagement({
    workflow,
    selectedRunId,
    activeOutputs,
    expandedOutput: modalState.expandedOutput,
    tempEditValue: modalState.tempEditValue,
    setActiveOutputs,
    setWorkflow,
    setExpandedOutput: modalState.setExpandedOutput,
    setIsEditingResult: modalState.setIsEditingResult,
    lang
  });

  // Use useAIGenerateWorkflow Hook
  const aiGenerateWorkflow = useAIGenerateWorkflow({
    workflow,
    setWorkflow,
    setCurrentView,
    getLightX2VConfig,
    resetView,
    lang
  });

  // Wrapper for generateWorkflowWithAI with loading state
  const generateWorkflowWithAI = useCallback(async (description: string) => {
    setIsGeneratingWorkflow(true);
    try {
      await aiGenerateWorkflow.generateWorkflowWithAI(description);
      modalState.setShowAIGenerateModal(false);
      setAIWorkflowDescription('');
    } catch (error: any) {
      console.error('[AI Workflow] Generation failed:', error);
      setGlobalError({
        message: t('execution_failed'),
        details: error.message || String(error)
      });
    } finally {
      setIsGeneratingWorkflow(false);
    }
  }, [aiGenerateWorkflow, modalState, setAIWorkflowDescription, setGlobalError, t]);

  // Use useAIChatWorkflow Hook
  const aiChatWorkflow = useAIChatWorkflow({
    workflow,
    setWorkflow,
    addNode: nodeManagement.addNode,
    deleteNode: nodeManagement.deleteNode,
    updateNodeData: nodeManagement.updateNodeData,
    replaceNode: nodeManagement.replaceNode,
    addConnection: connectionManagement.addConnection,
    deleteConnection: connectionManagement.deleteConnection,
    getNodeOutputs,
    getReplaceableTools: nodeManagement.getReplaceableTools,
    screenToWorldCoords,
    canvasRef,
    lang,
    lightX2VVoiceList: voiceList.lightX2VVoiceList,
    getCurrentHistoryIndex: undoRedo.getCurrentHistoryIndex,
    undoToIndex: undoRedo.undoToIndex
  });

  // Use useWorkflowExecution Hook
  const workflowExecution = useWorkflowExecution({
    workflow,
    setWorkflow,
    activeOutputs,
    setActiveOutputs,
    isPausedRef,
    setIsPaused,
    runningTaskIdsRef,
    abortControllerRef,
    getLightX2VConfig,
    setValidationErrors,
    setSelectedRunId,
    setGlobalError,
    updateNodeData: nodeManagement.updateNodeData,
    voiceList,
    lang
  });

  // Destructure updateNodeData from nodeManagement (for use in other places)
  const updateNodeData = nodeManagement.updateNodeData;

  // runWorkflow, validateWorkflow, stopWorkflow, and getDescendants are now provided by useWorkflowExecution Hook
  const runWorkflow = workflowExecution.runWorkflow;
  const stopWorkflow = workflowExecution.stopWorkflow;
  const validateWorkflow = workflowExecution.validateWorkflow;
  const getDescendants = workflowExecution.getDescendants;

  // Helper functions for voice selection
  const getLanguageDisplayName = useCallback((langCode: string) => {
    const languageMap: Record<string, string> = {
      'chinese': '中文',
      'en_us': '美式英语',
      'en_gb': '英式英语',
      'en_au': '澳洲英语',
      'es': '西语',
      'ja': '日语'
    };
    return languageMap[langCode] || langCode;
  }, []);

  // filteredVoices, isFemaleVoice are provided by useVoiceList Hook

  const toggleLang = () => {
    const next = lang === 'en' ? 'zh' : 'en';
    setLang(next);
    localStorage.setItem('omniflow_lang', next);
  };

  // Workflow loading is handled by useWorkflow Hook

  // Load workflow from URL hash on mount
  useEffect(() => {
    const loadWorkflowFromHash = async () => {
      // Check if URL hash contains workflow ID
      const hash = window.location.hash;
      const workflowMatch = hash.match(/^#?workflow\/([a-f0-9-]+)$/i);

      if (workflowMatch) {
        const workflowId = workflowMatch[1];
        console.log('[App] Loading workflow from URL hash:', workflowId);

        try {
          // Check if workflow belongs to current user
          const currentUserId = getCurrentUserId();
          const { isPreset, workflow: existingWorkflow } = await checkWorkflowOwnership(workflowId, currentUserId);

          if (existingWorkflow) {
            if (isPreset) {
              // Preset workflow - copy it to user's workflows
              console.log('[App] Workflow is a preset, copying to create user-owned version');

              // Generate new UUID for copied workflow
              const generateUUID = () => {
                return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
                  const r = Math.random() * 16 | 0;
                  const v = c === 'x' ? r : (r & 0x3 | 0x8);
                  return v.toString(16);
                });
              };
              const newWorkflowId = generateUUID();

              // Copy workflow via API
              const copyResponse = await apiRequest(`/api/v1/workflow/${workflowId}/copy`, {
                method: 'POST',
                body: JSON.stringify({
                  workflow_id: newWorkflowId
                })
              });

              if (copyResponse.ok) {
                const copyData = await copyResponse.json();
                const copiedWorkflowId = copyData.workflow_id;
                console.log('[App] Workflow copied successfully, new ID:', copiedWorkflowId);

                // Load the copied workflow
                const loaded = await loadWorkflow(copiedWorkflowId);
                if (loaded) {
                  const config = getLightX2VConfig(null);
                  const newWorkflow = {
                    ...loaded.workflow,
                    env: {
                      lightx2v_url: config.url,
                      lightx2v_token: config.token
                    }
                  };
                  setWorkflow(newWorkflow);
                  setActiveOutputs(loaded.activeOutputs);
                  setCurrentView('EDITOR');

                  // Update URL with new workflow ID
                  if (window.history && window.history.replaceState) {
                    window.history.replaceState(null, '', `#workflow/${copiedWorkflowId}`);
                  }
                }
              } else {
                console.error('[App] Failed to copy preset workflow from URL');
              }
            } else {
              // User's own workflow - load it directly
              const loaded = await loadWorkflow(workflowId);
              if (loaded) {
                const config = getLightX2VConfig(null);
                const newWorkflow = {
                  ...loaded.workflow,
                  env: {
                    lightx2v_url: config.url,
                    lightx2v_token: config.token
                  }
                };
                setWorkflow(newWorkflow);
                setActiveOutputs(loaded.activeOutputs);
                setCurrentView('EDITOR');
              }
            }
          } else {
            console.warn('[App] Workflow not found in database:', workflowId);
          }
        } catch (error) {
          console.error('[App] Failed to load workflow from URL hash:', error);
        }
      }
    };

    // Load workflow from hash on mount
    loadWorkflowFromHash();
  }, []); // Only run on mount - eslint-disable-line react-hooks/exhaustive-deps

  // Also listen for hash changes (when user navigates via browser back/forward)
  useEffect(() => {
    const handleHashChange = async () => {
      const hash = window.location.hash;
      const workflowMatch = hash.match(/^#?workflow\/([a-f0-9-]+)$/i);

      if (workflowMatch) {
        const workflowId = workflowMatch[1];

        // Only load if it's a different workflow
        if (!workflow || workflow.id !== workflowId) {
          console.log('[App] Hash changed, loading workflow:', workflowId);

          try {
            const currentUserId = getCurrentUserId();
            const { isPreset, workflow: existingWorkflow } = await checkWorkflowOwnership(workflowId, currentUserId);

            if (existingWorkflow && !isPreset) {
              // Only load user's own workflows on hash change (preset workflows should be copied first)
              const loaded = await loadWorkflow(workflowId);
              if (loaded) {
                const config = getLightX2VConfig(null);
                const newWorkflow = {
                  ...loaded.workflow,
                  env: {
                    lightx2v_url: config.url,
                    lightx2v_token: config.token
                  }
                };
                setWorkflow(newWorkflow);
                setActiveOutputs(loaded.activeOutputs);
                setCurrentView('EDITOR');
              }
            } else if (isPreset) {
              // For preset workflows, redirect to dashboard
              setCurrentView('DASHBOARD');
              // Clear hash or set to empty
              if (window.history && window.history.replaceState) {
                window.history.replaceState(null, '', '#');
              }
            }
          } catch (error) {
            console.error('[App] Failed to load workflow from hash change:', error);
          }
        }
      } else if (hash === '' || hash === '#') {
        // Hash cleared - go back to dashboard
        if (currentView === 'EDITOR') {
          setCurrentView('DASHBOARD');
        }
      }
    };

    window.addEventListener('hashchange', handleHashChange);
    return () => window.removeEventListener('hashchange', handleHashChange);
  }, [workflow, currentView, loadWorkflow, getLightX2VConfig, setWorkflow, setActiveOutputs, setCurrentView]);

  useEffect(() => {
    let interval: any;
    if (workflow?.isRunning) {
      interval = setInterval(() => setTicker(t => t + 1), 100);
    }
    return () => clearInterval(interval);
  }, [workflow?.isRunning]);

  // Voice list loading is handled by useVoiceList Hook
  // Auto-match resource_id when voice list is loaded
  useEffect(() => {
    if (!voiceList.lightX2VVoiceList?.voices || !workflow) return;

    const voiceData = voiceList.lightX2VVoiceList;
    if (voiceData.voices && voiceData.voices.length > 0) {
          workflow.nodes.forEach(node => {
            const isLightX2V = node.data.model === 'lightx2v' || node.data.model?.startsWith('lightx2v');
            if (node.toolId === 'tts' && isLightX2V && node.data.voiceType) {
              const matchingVoice = voiceData.voices.find((v: any) => v.voice_type === node.data.voiceType);
              if (matchingVoice?.resource_id) {
                // Only update if resourceId is empty, wrong, or missing
                if (!node.data.resourceId || node.data.resourceId !== matchingVoice.resource_id) {
                  updateNodeData(node.id, 'resourceId', matchingVoice.resource_id);
                  console.log(`[LightX2V] Auto-matched resource_id: ${matchingVoice.resource_id} for voice: ${node.data.voiceType}`);
                }
              }
            }
          });
        }
  }, [voiceList.lightX2VVoiceList, workflow]);

  // saveWorkflowToLocal and deleteWorkflow are provided by useWorkflow Hook

  const deleteWorkflow = useCallback((id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (!window.confirm(t('confirm_delete'))) return;
    deleteWorkflowFromHook(id);
  }, [t, deleteWorkflowFromHook]);

  const openWorkflow = useCallback(async (w: WorkflowState) => {
    setSelectedRunId(null);
    setSelectedNodeId(null);
    setSelectedConnectionId(null);
    setValidationErrors([]);
    setActiveOutputs({});
    voiceList.resetVoiceList(); // Reset voice list when switching workflows
    voiceList.resetCloneVoiceList(); // Reset clone voice list

    // Try to load from database if workflow has a database ID
    let workflowToOpen = w;
    let activeOutputsToRestore: Record<string, any> = {};
    let workflowIdToOpen = w.id;

    if (w.id && (w.id.startsWith('workflow-') || w.id.match(/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i))) {
      try {
        // Check if workflow belongs to current user
        const currentUserId = getCurrentUserId();
        const { isPreset } = await checkWorkflowOwnership(w.id, currentUserId);

        if (isPreset) {
          // Workflow belongs to different user (preset workflow), copy it
          console.log('[App] Opening preset workflow, copying to create user-owned version');

          // Generate new UUID for copied workflow
          const generateUUID = () => {
            return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
              const r = Math.random() * 16 | 0;
              const v = c === 'x' ? r : (r & 0x3 | 0x8);
              return v.toString(16);
            });
          };
          const newWorkflowId = generateUUID();

          // Copy workflow via API
          const copyResponse = await apiRequest(`/api/v1/workflow/${w.id}/copy`, {
            method: 'POST',
            body: JSON.stringify({
              workflow_id: newWorkflowId
            })
          });

          if (copyResponse.ok) {
            const copyData = await copyResponse.json();
            workflowIdToOpen = copyData.workflow_id;
            console.log('[App] Workflow copied successfully, new ID:', workflowIdToOpen);

            // Load the copied workflow
            const loaded = await loadWorkflow(workflowIdToOpen);
            if (loaded) {
              workflowToOpen = loaded.workflow;
              activeOutputsToRestore = loaded.activeOutputs;
            }
          } else {
            const errorText = await copyResponse.text();
            console.error('[App] Failed to copy preset workflow:', errorText);
            throw new Error(`Failed to copy preset workflow: ${errorText}`);
          }
        } else {
          // Workflow belongs to current user, load it normally
          const loaded = await loadWorkflow(w.id);
          if (loaded) {
            workflowToOpen = loaded.workflow;
            activeOutputsToRestore = loaded.activeOutputs;
          }
        }
      } catch (error) {
        console.warn('[App] Failed to load workflow from database, using cached version:', error);
      }
    }

    // 获取当前的 LightX2V 配置（从主应用或环境变量）
    const config = getLightX2VConfig(null);

    // 更新工作流的 env 配置，使用动态获取的配置
    const newWorkflow = {
      ...workflowToOpen,
      id: workflowIdToOpen, // Use the correct workflow ID (original or copied)
      isDirty: false,
      isRunning: false,
      env: {
        lightx2v_url: config.url,
        lightx2v_token: config.token
      }
    };
    setWorkflow(newWorkflow);
    setActiveOutputs(activeOutputsToRestore);
    setCurrentView('EDITOR');

    // Update URL to show workflow_id
    if (typeof window !== 'undefined' && window.history && window.history.replaceState) {
      window.history.replaceState(null, '', `#workflow/${workflowIdToOpen}`);
    }

    // History will be automatically initialized by useUndoRedo when workflow changes
  }, [loadWorkflow, getLightX2VConfig, voiceList, setWorkflow, setCurrentView, setSelectedRunId, setSelectedNodeId, setSelectedConnectionId, setValidationErrors, setActiveOutputs]);

  const createNewWorkflow = useCallback(async () => {
    setSelectedRunId(null);
    setSelectedNodeId(null);
    setSelectedConnectionId(null);
    setValidationErrors([]);
    setActiveOutputs({});
    voiceList.resetVoiceList(); // Reset voice list when creating new workflow
    voiceList.resetCloneVoiceList(); // Reset clone voice list

    // 获取当前的 LightX2V 配置（从主应用或环境变量）
    const config = getLightX2VConfig(null);

    // Use temporary ID for new workflow (backend will generate the real workflow_id)
    const tempWorkflowId = `temp-${Date.now()}`;

    // Create workflow in database
    try {
      const tempFlow: WorkflowState = {
        id: tempWorkflowId, // Temporary ID, will be replaced by backend
        name: t('untitled'),
        nodes: [],
        connections: [],
        isDirty: false,
        isRunning: false,
        globalInputs: {},
        env: {
          lightx2v_url: config.url,
          lightx2v_token: config.token
        },
        history: [],
        updatedAt: Date.now(),
        showIntermediateResults: true
      };

      // Create workflow in database (backend will generate workflow_id)
      const finalWorkflowId = await saveWorkflowToDatabase(tempFlow, { name: t('untitled') });

      // Update workflow with backend-generated ID
      const finalFlow = { ...tempFlow, id: finalWorkflowId, isDirty: false };
      setWorkflow(finalFlow);
      setActiveOutputs({});
      setCurrentView('EDITOR');
      // Update URL to show workflow_id
      if (window.history && window.history.replaceState) {
        window.history.replaceState(null, '', `#workflow/${finalWorkflowId}`);
      }
    } catch (error) {
      console.error('[App] Failed to create workflow in database, creating local workflow:', error);
      // Fallback to local workflow (keep temp ID)
      const newFlow: WorkflowState = {
        id: tempWorkflowId,
        name: t('untitled'),
        nodes: [],
        connections: [],
        isDirty: true,
        isRunning: false,
        globalInputs: {},
        env: {
          lightx2v_url: config.url,
          lightx2v_token: config.token
        },
        history: [],
        updatedAt: Date.now(),
        showIntermediateResults: true
      };
      setWorkflow(newFlow);
      setCurrentView('EDITOR');
      if (window.history && window.history.replaceState) {
        window.history.replaceState(null, '', `#workflow/${tempWorkflowId}`);
      }
    }
    // History will be automatically initialized by useUndoRedo when workflow changes
  }, [saveWorkflowToDatabase, loadWorkflow, getLightX2VConfig, t, voiceList, setWorkflow, setCurrentView, setSelectedRunId, setSelectedNodeId, setSelectedConnectionId, setValidationErrors, setActiveOutputs]);

  const selectedNode = useMemo(() => workflow?.nodes.find(n => n.id === selectedNodeId), [workflow, selectedNodeId]);

  // expandedResultData, activeResultsList, and handleManualResultEdit are now provided by useResultManagement Hook
  const expandedResultData = resultManagement.expandedResultData;
  const activeResultsList = resultManagement.activeResultsList;
  const handleManualResultEdit = resultManagement.handleManualResultEdit;


  const handleGlobalInputChange = useCallback((nodeId: string, portId: string, value: any) => {
    if (selectedRunId) setSelectedRunId(null);
    setValidationErrors([]);
    setWorkflow(prev => prev ? ({ ...prev, globalInputs: { ...prev.globalInputs, [`${nodeId}-${portId}`]: value }, isDirty: true }) : null);
  }, [selectedRunId]);

  // addNode and pinOutputToCanvas are now provided by useNodeManagement Hook
  const addNode = nodeManagement.addNode;
  const pinOutputToCanvas = nodeManagement.pinOutputToCanvas;

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Undo/Redo shortcuts (Ctrl+Z / Ctrl+Y or Ctrl+Shift+Z)
      if ((e.ctrlKey || e.metaKey) && !e.shiftKey && e.key === 'z') {
        e.preventDefault();
        if (currentView === 'EDITOR' && workflow) {
          undoRedo.undo();
        }
        return;
      }
      if ((e.ctrlKey || e.metaKey) && (e.key === 'y' || (e.shiftKey && e.key === 'z'))) {
        e.preventDefault();
        if (currentView === 'EDITOR' && workflow) {
          undoRedo.redo();
        }
        return;
      }

      // Delete/Backspace for nodes and connections
      if ((e.key === 'Delete' || e.key === 'Backspace') && !['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement?.tagName || '')) {
        if (selectedNodeId) nodeManagement.deleteNode(selectedNodeId);
        if (selectedConnectionId) connectionManagement.deleteConnection(selectedConnectionId);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [selectedNodeId, selectedConnectionId, nodeManagement, connectionManagement, currentView, workflow, undoRedo]);

  // Close replace menu, output quick add menu, model select, and voice select when clicking outside
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      const target = e.target as HTMLElement;
      if (modalState.showReplaceMenu && !target.closest('.replace-menu-container')) {
        modalState.setShowReplaceMenu(null);
      }
      if (modalState.showOutputQuickAdd && !target.closest('.output-quick-add-menu')) {
        modalState.setShowOutputQuickAdd(null);
      }
      if (modalState.showModelSelect && !target.closest('.model-select-container')) {
        modalState.setShowModelSelect(null);
      }
      if (modalState.showVoiceSelect && !target.closest('.voice-select-container')) {
        modalState.setShowVoiceSelect(null);
      }
    };
    window.addEventListener('click', handleClickOutside);
    return () => window.removeEventListener('click', handleClickOutside);
  }, [modalState.showReplaceMenu, modalState.showOutputQuickAdd, modalState.showModelSelect, modalState.showVoiceSelect]);

  // deleteSelectedNode is now provided by useNodeManagement Hook
  const deleteSelectedNode = useCallback(() => {
    if (!selectedNodeId) return;
    nodeManagement.deleteNode(selectedNodeId);
  }, [selectedNodeId, nodeManagement]);

  // getReplaceableTools is now provided by useNodeManagement Hook
  const getReplaceableTools = nodeManagement.getReplaceableTools;

  // Replace a node with another compatible tool (complex logic with connection mapping)
  const replaceNode = useCallback((nodeId: string, newToolId: string) => {
    if (!workflow) return;
    const node = workflow.nodes.find(n => n.id === nodeId);
    if (!node) return;

    const newTool = TOOLS.find(t => t.id === newToolId);
    if (!newTool) return;

    // Get current node outputs
    const currentNodeOutputs = getNodeOutputs(node);

    // Handle text-generation special case (dynamic outputs)
    let newToolOutputs: Port[] = [];
    let newCustomOutputs: any[] | undefined = undefined;
    if (newToolId === 'text-generation') {
      // If replacing with text-generation, preserve customOutputs if they exist
      if (node.toolId === 'text-generation' && node.data.customOutputs) {
        newToolOutputs = node.data.customOutputs.map((o: any) => ({ ...o, type: DataType.TEXT }));
        newCustomOutputs = node.data.customOutputs;
      } else {
        // When replacing another node with text-generation, create customOutputs based on current node outputs
        // This allows replacement of nodes with outputs
        newCustomOutputs = currentNodeOutputs.map((out, idx) => ({
          id: `out-${idx + 1}`,
          label: out.label || `Output ${idx + 1}`,
          description: out.label || `Output ${idx + 1}`
        }));
        newToolOutputs = newCustomOutputs.map((o: any) => ({ ...o, type: DataType.TEXT }));
      }
    } else {
      newToolOutputs = newTool.outputs;
    }

    // Check if outputs are compatible (for text-generation, we've already created matching outputs)
    if (currentNodeOutputs.length !== newToolOutputs.length) return;
    const isCompatible = currentNodeOutputs.every((out, idx) => {
      if (idx >= newToolOutputs.length) return false;
      return out.type === newToolOutputs[idx].type;
    });
    if (!isCompatible) return;

    // Create a mapping of old output port IDs to new ones
    const outputPortMap: Record<string, string> = {};
    currentNodeOutputs.forEach((oldOut, idx) => {
      if (idx < newToolOutputs.length) {
        outputPortMap[oldOut.id] = newToolOutputs[idx].id;
      }
    });

    // Update the node
    setWorkflow(prev => {
      if (!prev) return null;

      // Create new node with new tool
      const newNode: WorkflowNode = {
        ...node,
        toolId: newToolId,
        data: {
          ...node.data,
          // Reset model if the new tool doesn't have models
          model: newTool.models && newTool.models.length > 0 ? (newTool.models[0].id || node.data.model) : undefined,
          // Preserve customOutputs if replacing with text-generation and current node has them
          customOutputs: newToolId === 'text-generation' ? (newCustomOutputs || node.data.customOutputs) : (newToolId !== 'text-generation' ? node.data.customOutputs : undefined)
        },
        status: NodeStatus.IDLE,
        error: undefined,
        executionTime: undefined,
        startTime: undefined
      };

      // Update connections: map old output port IDs to new ones
      // Special handling for TTS -> Voice Clone replacement
      const isTTSToVoiceClone = node.toolId === 'tts' && newToolId === 'lightx2v-voice-clone';

      const updatedConnections = prev.connections.map(conn => {
        // Handle output connections (source is the replaced node)
        if (conn.sourceNodeId === nodeId) {
          const newSourcePortId = outputPortMap[conn.sourcePortId];
          if (newSourcePortId) {
            return { ...conn, sourcePortId: newSourcePortId };
          }
          // If no mapping found, remove the connection
          return null;
        }

        // Handle input connections (target is the replaced node)
        if (conn.targetNodeId === nodeId) {
          // Special case: TTS -> Voice Clone
          if (isTTSToVoiceClone) {
            // Map in-text to in-tts-text
            if (conn.targetPortId === 'in-text') {
              return { ...conn, targetPortId: 'in-tts-text' };
            }
            // Remove in-context-tone connection
            if (conn.targetPortId === 'in-context-tone') {
              return null;
            }
          }

          // For other replacements, try to map input ports
          const oldTool = TOOLS.find(t => t.id === node.toolId);
          const oldInputs = oldTool?.inputs || [];
          const newInputs = newTool.inputs || [];

          // Try to find matching input port by type and position
          const oldInputIndex = oldInputs.findIndex(inp => inp.id === conn.targetPortId);
          if (oldInputIndex >= 0 && oldInputIndex < newInputs.length) {
            const oldInput = oldInputs[oldInputIndex];
            const newInput = newInputs[oldInputIndex];
            // Only map if types match
            if (oldInput.type === newInput.type) {
              return { ...conn, targetPortId: newInput.id };
            }
          }

          // If no mapping found, check if port ID exists in new tool
          const portExists = newInputs.some(inp => inp.id === conn.targetPortId);
          if (portExists) {
            return conn; // Keep connection if port exists
          }

          // Remove connection if port doesn't exist
          return null;
        }

        return conn;
      }).filter((c): c is Connection => c !== null);

      return {
        ...prev,
        nodes: prev.nodes.map(n => n.id === nodeId ? newNode : n),
        connections: updatedConnections,
        isDirty: true
      };
    });

    modalState.setShowReplaceMenu(null);
  }, [workflow]);

  // deleteSelectedConnection and addConnection are now provided by useConnectionManagement Hook
  const deleteSelectedConnection = useCallback(() => {
    if (!selectedConnectionId) return;
    connectionManagement.deleteConnection(selectedConnectionId);
  }, [selectedConnectionId, connectionManagement]);

  const addConnection = connectionManagement.addConnection;

  // quickAddInput is now provided by useNodeManagement Hook
  const quickAddInput = nodeManagement.quickAddInput;

  // Get tools that can accept a specific output type
  const getCompatibleToolsForOutput = useCallback((outputType: DataType): ToolDefinition[] => {
    return TOOLS.filter(tool => {
      // Skip input nodes
      if (tool.category === 'Input') return false;
      // Find tools that have at least one input port matching the output type
      return tool.inputs.some(input => input.type === outputType);
    });
  }, []);

  // quickAddOutput is now provided by useNodeManagement Hook
  const quickAddOutput = nodeManagement.quickAddOutput;

  const disconnectedInputs = useMemo(() => {
    if (!workflow) return [];
    const list: { nodeId: string; port: Port; toolName: string; isSourceNode?: boolean; dataType: DataType }[] = [];

    workflow.nodes.forEach(node => {
      const tool = TOOLS.find(t => t.id === node.toolId);
      if (!tool || tool.category === 'Input') return;
      tool.inputs.forEach(port => {
        const isConnected = workflow.connections.some(c => c.targetNodeId === node.id && c.targetPortId === port.id);
        if (!isConnected) {
           list.push({ nodeId: node.id, port, toolName: (lang === 'zh' ? tool.name_zh : tool.name), dataType: port.type });
        }
      });
    });

    workflow.nodes.forEach(node => {
      const tool = TOOLS.find(t => t.id === node.toolId);
      if (!tool || tool.category !== 'Input') return;

      const val = node.data.value;
      const isEmpty = (Array.isArray(val) && val.length === 0) || !val;
      if (isEmpty) {
        list.push({
          nodeId: node.id,
          port: tool.outputs[0],
          toolName: (lang === 'zh' ? tool.name_zh : tool.name),
          isSourceNode: true,
          dataType: tool.outputs[0].type
        });
      }
    });

    return list;
  }, [workflow?.nodes, workflow?.connections, lang]);

  // validateWorkflow and runWorkflow are now provided by useWorkflowExecution Hook

  // activeResultsList is now provided by useResultManagement Hook

  // Enhanced mouse handlers that combine canvas Hook with workflow updates
  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    const rect = canvasRef.current?.getBoundingClientRect();
    if (!rect) return;
    const x = e.clientX - rect.left, y = e.clientY - rect.top;

    // Check if mouse is over a node or port
    const target = e.target as HTMLElement;
    const isOverNodeElement = target.closest('.node-element') || target.closest('.port') || target.closest('.connection-path');
    setIsOverNode(!!isOverNodeElement);

    // Call canvas Hook's handleMouseMove for panning
    canvasHandleMouseMove(e);

    // Handle node dragging with workflow update
    if (draggingNode) {
        if (selectedRunId) setSelectedRunId(null);
      const world = screenToWorldCoords(x, y);
      setWorkflow(prev => prev ? ({
        ...prev,
        nodes: prev.nodes.map(n => n.id === draggingNode.id ? {
          ...n,
          x: world.x - draggingNode.offsetX,
          y: world.y - draggingNode.offsetY
        } : n),
        isDirty: true
      }) : null);
    }
  }, [draggingNode, selectedRunId, canvasHandleMouseMove, screenToWorldCoords]);

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    if (e.button !== 0) return;
    // Call canvas Hook's handleMouseDown for panning
    canvasHandleMouseDown(e);
    // Additional logic: clear selection when clicking on canvas
    if (!(e.target as HTMLElement).closest('.node-element') &&
        !(e.target as HTMLElement).closest('.port') &&
        !(e.target as HTMLElement).closest('.connection-path')) {
      setSelectedNodeId(null);
      setSelectedConnectionId(null);
    }
  }, [canvasHandleMouseDown]);

  const handleNodeDragStart = useCallback((nodeId: string, offsetX: number, offsetY: number) => {
    if (selectedRunId) setSelectedRunId(null);
    canvasHandleNodeDragStart(nodeId, offsetX, offsetY);
  }, [selectedRunId, canvasHandleNodeDragStart]);

  const handleNodeDrag = useCallback((nodeId: string, x: number, y: number) => {
    if (!draggingNode || draggingNode.id !== nodeId) return;
    // Call canvas Hook's handleNodeDrag
    canvasHandleNodeDrag(nodeId, x, y);
    // Update workflow with new node position
    setWorkflow(prev => prev ? ({
      ...prev,
      nodes: prev.nodes.map(n => n.id === nodeId ? {
        ...n,
        x: x - draggingNode.offsetX,
        y: y - draggingNode.offsetY
      } : n),
      isDirty: true
    }) : null);
  }, [draggingNode, canvasHandleNodeDrag]);

  const handleNodeDragEnd = useCallback(() => {
    canvasHandleNodeDragEnd();
  }, [canvasHandleNodeDragEnd]);

  const clearSnapshot = useCallback(() => {
    setSelectedRunId(null);
  }, []);

  if (currentView === 'DASHBOARD') {
    return (
      <div className="flex flex-col h-full">
        <Dashboard
          lang={lang}
          myWorkflows={myWorkflows}
          activeTab={activeTab}
          onToggleLang={toggleLang}
          onCreateWorkflow={createNewWorkflow}
          onAIGenerate={() => modalState.setShowAIGenerateModal(true)}
          onOpenWorkflow={openWorkflow}
          onDeleteWorkflow={deleteWorkflow}
          onSetActiveTab={setActiveTab}
        />
        <AIGenerateModal
          lang={lang}
          isOpen={modalState.showAIGenerateModal}
          description={aiWorkflowDescription}
          isGenerating={isGeneratingWorkflow}
          onClose={() => {
            modalState.setShowAIGenerateModal(false);
            setAIWorkflowDescription('');
          }}
          onDescriptionChange={setAIWorkflowDescription}
          onGenerate={() => generateWorkflowWithAI(aiWorkflowDescription)}
        />
      </div>
    );
  }

  if (!workflow) return null;

  const sourceNodes = selectedRunId
    ? (workflow.history.find(r => r.id === selectedRunId)?.nodesSnapshot || [])
    : workflow.nodes;

  const sourceOutputs = selectedRunId
    ? (workflow.history.find(r => r.id === selectedRunId)?.outputs || {})
    : activeOutputs;

  return (
    <div className="flex flex-col h-full bg-slate-950 text-slate-200 selection:bg-indigo-500/30 font-sans overflow-hidden">
      <ExpandedOutputModal
        lang={lang}
        expandedOutput={modalState.expandedOutput}
        expandedResultData={expandedResultData}
        isEditingResult={modalState.isEditingResult}
        tempEditValue={modalState.tempEditValue}
          onClose={() => {
            modalState.setExpandedOutput(null);
            modalState.setIsEditingResult(false);
          }}
          onEditToggle={() => {
            if (!modalState.isEditingResult) {
              modalState.setTempEditValue(
                typeof expandedResultData?.content === 'object'
                  ? JSON.stringify(expandedResultData.content, null, 2)
                  : expandedResultData?.content || ''
              );
            }
            modalState.setIsEditingResult(!modalState.isEditingResult);
          }}
          onSaveEdit={handleManualResultEdit}
          onTempEditValueChange={modalState.setTempEditValue}
      />

      <Editor
        lang={lang}
        workflow={workflow}
        view={view}
        selectedNodeId={selectedNodeId}
        selectedConnectionId={selectedConnectionId}
        selectedRunId={selectedRunId}
        connecting={connecting}
        mousePos={mousePos}
        activeOutputs={activeOutputs}
        nodeHeights={nodeHeightsRef.current}
        sourceNodes={sourceNodes}
        sourceOutputs={sourceOutputs}
        isPaused={isPaused}
        isRunning={workflow.isRunning}
        isSaving={isSavingWorkflow}
        sidebarCollapsed={modalState.sidebarCollapsed}
        validationErrors={validationErrors}
        globalError={globalError}
        canvasRef={canvasRef}
        onBack={() => setCurrentView('DASHBOARD')}
        onWorkflowNameChange={(name) => {
          if (selectedRunId) setSelectedRunId(null);
          setWorkflow(p => p ? ({ ...p, name, isDirty: true }) : null);
        }}
        onZoomIn={zoomIn}
        onZoomOut={zoomOut}
        onResetView={resetView}
        onToggleLang={toggleLang}
        onClearSnapshot={clearSnapshot}
        onSave={async () => {
          if (!workflow || isSavingWorkflow) return; // Prevent multiple saves
          try {
            await saveWorkflowToDatabase(workflow, { name: workflow.name });
            // Also save to localStorage as backup
            saveWorkflowToLocal(workflow);
            setWorkflow(prev => prev ? { ...prev, isDirty: false } : null);
            // 手动保存后重置自动保存计时器
            if (autoSaveHook) {
              autoSaveHook.resetAutoSaveTimer();
            }
          } catch (error) {
            console.error('[App] Failed to save workflow to database:', error);
            // Fallback to localStorage
            saveWorkflowToLocal(workflow);
          }
        }}
        onPause={() => {
                const newPausedState = !isPaused;
                setIsPaused(newPausedState);
                isPausedRef.current = newPausedState;
              }}
        onRun={() => runWorkflow()}
        onStop={async () => {
          if (stopWorkflow) {
            await stopWorkflow();
          }
        }}
          onToggleSidebar={() => modalState.setSidebarCollapsed(!modalState.sidebarCollapsed)}
        canUndo={undoRedo.canUndo}
        canRedo={undoRedo.canRedo}
        onUndo={undoRedo.undo}
        onRedo={undoRedo.redo}
        onAddNode={addNode}
          onMouseMove={handleMouseMove}
          onMouseDown={handleMouseDown}
        onMouseUp={canvasHandleMouseUp}
        onMouseLeave={canvasHandleMouseLeave}
        onWheel={canvasHandleWheel}
        onNodeSelect={setSelectedNodeId}
        onConnectionSelect={setSelectedConnectionId}
        onNodeDragStart={handleNodeDragStart}
        onNodeDrag={handleNodeDrag}
        onNodeDragEnd={handleNodeDragEnd}
        getNodeOutputs={getNodeOutputs}
        isOverNode={isOverNode}
        isPanning={isPanning}
        onCloseValidation={() => setValidationErrors([])}
        onCloseError={() => setGlobalError(null)}
        showReplaceMenu={modalState.showReplaceMenu}
        showOutputQuickAdd={modalState.showOutputQuickAdd}
        showModelSelect={modalState.showModelSelect}
        showVoiceSelect={modalState.showVoiceSelect}
        lightX2VVoiceList={voiceList.lightX2VVoiceList}
        cloneVoiceList={voiceList.cloneVoiceList}
        onUpdateNodeData={updateNodeData}
        onDeleteNode={deleteSelectedNode}
        onReplaceNode={replaceNode}
        onRunWorkflow={runWorkflow}
          onSetReplaceMenu={modalState.setShowReplaceMenu}
          onSetOutputQuickAdd={modalState.setShowOutputQuickAdd}
          onSetModelSelect={modalState.setShowModelSelect}
          onSetVoiceSelect={modalState.setShowVoiceSelect}
          onSetExpandedOutput={modalState.setExpandedOutput}
          onSetShowAudioEditor={modalState.setShowAudioEditor}
        onSetConnecting={setConnecting}
        onAddConnection={addConnection}
        onClearSelectedRunId={() => setSelectedRunId(null)}
        getReplaceableTools={getReplaceableTools}
        getCompatibleToolsForOutput={getCompatibleToolsForOutput}
        quickAddInput={quickAddInput}
        quickAddOutput={quickAddOutput}
        onNodeHeightChange={(nodeId, height) => {
          nodeHeightsRef.current.set(nodeId, height);
          // Force re-render to update connections
          setTicker(prev => prev + 1);
        }}
        disconnectedInputs={disconnectedInputs}
        loadingVoiceList={voiceList.loadingVoiceList}
        voiceSearchQuery={voiceList.voiceSearchQuery}
        setVoiceSearchQuery={voiceList.setVoiceSearchQuery}
        showVoiceFilter={voiceList.showVoiceFilter}
        setShowVoiceFilter={voiceList.setShowVoiceFilter}
        voiceFilterGender={voiceList.voiceFilterGender}
        setVoiceFilterGender={voiceList.setVoiceFilterGender}
        filteredVoices={voiceList.filteredVoices}
        isFemaleVoice={voiceList.isFemaleVoice}
        loadingCloneVoiceList={voiceList.loadingCloneVoiceList}
        onGlobalInputChange={handleGlobalInputChange}
          onShowCloneVoiceModal={() => modalState.setShowCloneVoiceModal(true)}
        resultsCollapsed={modalState.resultsCollapsed}
        activeResultsList={activeResultsList}
          onToggleResultsCollapsed={() => modalState.setResultsCollapsed(!modalState.resultsCollapsed)}
        onSelectRun={setSelectedRunId}
        onToggleShowIntermediate={() => setWorkflow(p => p ? ({ ...p, showIntermediateResults: !p.showIntermediateResults }) : null)}
          onExpandOutput={modalState.setExpandedOutput}
        onPinOutputToCanvas={pinOutputToCanvas}
        isAIChatOpen={modalState.isAIChatOpen}
        isAIChatCollapsed={modalState.isAIChatCollapsed}
        onToggleAIChat={() => {
          // 如果面板已打开但被折叠，则展开它
          if (modalState.isAIChatOpen && modalState.isAIChatCollapsed) {
            modalState.setIsAIChatCollapsed(false);
          } else {
            // 否则切换打开/关闭状态
            const newState = !modalState.isAIChatOpen;
            modalState.setIsAIChatOpen(newState);
            // 如果打开，同时展开面板
            if (newState) {
              modalState.setIsAIChatCollapsed(false);
            }
          }
        }}
        onToggleAIChatCollapse={() => modalState.setIsAIChatCollapsed(!modalState.isAIChatCollapsed)}
        aiChatHistory={aiChatWorkflow.chatHistory}
        isAIProcessing={aiChatWorkflow.isProcessing}
        onAISendMessage={aiChatWorkflow.handleUserInput}
        aiModel={aiChatWorkflow.aiModel}
        onAIModelChange={aiChatWorkflow.setAiModel}
        onAIUndo={(messageId) => {
          // 撤销这次AI对话的所有操作
          aiChatWorkflow.undoMessageOperations(messageId);
        }}
            onAIRetry={(messageId) => {
              // 重试：找到对应的用户消息并重新发送
              const history = aiChatWorkflow.chatHistory;
              const assistantMessageIndex = history.findIndex(m => m.id === messageId);

              if (assistantMessageIndex >= 0) {
                // 找到当前 assistant 消息之前最近的一条用户消息
                let userMessage = null;

                // 先检查前一条消息是否是用户消息（最常见的情况）
                if (assistantMessageIndex > 0) {
                  const prevMessage = history[assistantMessageIndex - 1];
                  if (prevMessage && prevMessage.role === 'user') {
                    userMessage = prevMessage;
                  }
                }

                // 如果前一条不是用户消息，向前查找最近的一条用户消息
                if (!userMessage) {
                  for (let i = assistantMessageIndex - 1; i >= 0; i--) {
                    if (history[i].role === 'user') {
                      userMessage = history[i];
                      break;
                    }
                  }
                }

                // 如果找到了用户消息，重新发送
                if (userMessage) {
                  aiChatWorkflow.handleUserInput(userMessage.content);
                }
              }
            }}
            nodeConfigPanelCollapsed={modalState.nodeConfigPanelCollapsed}
            onToggleNodeConfigPanel={() => modalState.setNodeConfigPanelCollapsed(!modalState.nodeConfigPanelCollapsed)}
            rightPanelSplitRatio={modalState.rightPanelSplitRatio}
            onRightPanelResize={(deltaY: number) => {
              const container = document.querySelector('.flex.flex-col.w-80.border-l');
              if (container) {
                const containerHeight = container.clientHeight;
                const currentRatio = modalState.rightPanelSplitRatio;
                const deltaRatio = deltaY / containerHeight;
                const newRatio = Math.max(0.2, Math.min(0.8, currentRatio + deltaRatio));
                modalState.setRightPanelSplitRatio(newRatio);
                localStorage.setItem('omniflow_right_panel_split_ratio', newRatio.toString());
              }
            }}
            aiChatPanelPosition={modalState.aiChatPanelPosition}
            aiChatPanelSize={modalState.aiChatPanelSize}
            onAiChatPanelPositionChange={(position) => {
              modalState.setAiChatPanelPosition(position);
              localStorage.setItem('omniflow_ai_chat_panel_position', JSON.stringify(position));
            }}
            onAiChatPanelSizeChange={(size) => {
              modalState.setAiChatPanelSize(size);
              localStorage.setItem('omniflow_ai_chat_panel_size', JSON.stringify(size));
            }}
          />

      <style>{`
        @keyframes marching-ants { from { stroke-dashoffset: 40; } to { stroke-dashoffset: 0; } }
        .animate-marching-ants { animation: marching-ants 1.5s linear infinite; }
        .animate-spin-slow { animation: spin 10s linear infinite; }
        .custom-scrollbar::-webkit-scrollbar { width: 5px; height: 5px; }
        .canvas-grid { background-image: radial-gradient(rgba(51, 65, 85, 0.4) 1px, transparent 1px); }
        .connection-path:hover { stroke-opacity: 0.8; stroke-width: 5px; }
      `}</style>

      <AIGenerateModal
        lang={lang}
        isOpen={modalState.showAIGenerateModal}
        description={aiWorkflowDescription}
        isGenerating={isGeneratingWorkflow}
        onClose={() => {
                  modalState.setShowAIGenerateModal(false);
                  setAIWorkflowDescription('');
                }}
        onDescriptionChange={setAIWorkflowDescription}
        onGenerate={() => generateWorkflowWithAI(aiWorkflowDescription)}
      />

        <CloneVoiceModal
          isOpen={modalState.showCloneVoiceModal}
          lightX2VConfig={getLightX2VConfig(workflow)}
                onClose={(newSpeakerId?: string) => {
            modalState.setShowCloneVoiceModal(false);
                  // Reload clone voice list after closing
                  const config = getLightX2VConfig(workflow);
                  if (config.url && config.token) {
            lightX2VGetCloneVoiceList(config.url, config.token)
              .then(voices => {
                voiceList.resetCloneVoiceList();
                      // Auto-select the newly created voice in the current node if applicable
                      if (newSpeakerId && selectedNodeId && workflow) {
                        const node = workflow.nodes.find(n => n.id === selectedNodeId);
                        if (node && node.toolId === 'lightx2v-voice-clone') {
                          updateNodeData(selectedNodeId, 'speakerId', newSpeakerId);
                        }
                      }
              })
              .catch(err => console.error('[LightX2V] Failed to reload clone voice list:', err));
                  }
                }}
              />

      {modalState.showAudioEditor && workflow && (() => {
        const node = workflow.nodes.find(n => n.id === modalState.showAudioEditor);
        if (!node || node.toolId !== 'audio-input' || !node.data.value) return null;
        return (
          <AudioEditorModal
            nodeId={modalState.showAudioEditor}
            audioData={node.data.value}
            onClose={() => modalState.setShowAudioEditor(null)}
            onSave={(trimmedAudio) => {
              updateNodeData(modalState.showAudioEditor!, 'value', trimmedAudio);
              modalState.setShowAudioEditor(null);
            }}
            lang={lang}
          />
        );
      })()}
    </div>
  );
};

export default App;
