import React, { useCallback, useRef } from 'react';
import { WorkflowState, Connection, NodeStatus, NodeHistoryEntry } from '../../types';

const MAX_NODE_HISTORY = 20;
import { TOOLS } from '../../constants';
import {
  geminiText, geminiImage, geminiSpeech, geminiVideo,
  lightX2VTask, lightX2VTTS, lightX2VVoiceCloneTTS,
  deepseekText, doubaoText, ppchatGeminiText,
  getLightX2VConfigForModel,
  lightX2VCancelTask,
  lightX2VTaskQuery
} from '../../services/geminiService';
import { isStandalone } from '../config/runtimeMode';
import { removeGeminiWatermark } from '../../services/watermarkRemover';
import { useTranslation, Language } from '../i18n/useTranslation';
import { saveNodeOutputs, saveInputFileViaOutputSave, getNodeOutputData, getWorkflowFileByFileId } from '../utils/workflowFileManager';
import { apiRequest } from '../utils/apiClient';
import { resolveLightX2VResultRef as resolveLightX2VResultRefUtil } from '../utils/resultRef';
import { workflowSaveQueue } from '../utils/workflowSaveQueue';
import { workflowOfflineQueue } from '../utils/workflowOfflineQueue';
import { checkWorkflowOwnership, getCurrentUserId } from '../utils/workflowUtils';
import { getAssetPath, getAssetBasePath } from '../utils/assetPath';
import { createHistoryEntryFromValue, normalizeHistoryEntries } from '../utils/historyEntry';

/** LightX2V 结果引用：用 task_id + output_name 代替过期 CDN URL，需要时通过 result_url 解析 */
export type LightX2VResultRef = { __type: 'lightx2v_result'; task_id: string; output_name: string; is_cloud: boolean };
export function isLightX2VResultRef(val: any): val is LightX2VResultRef {
  return val != null && typeof val === 'object' && !Array.isArray(val) &&
    (val as any).__type === 'lightx2v_result' &&
    typeof (val as any).task_id === 'string' &&
    typeof (val as any).output_name === 'string';
}
function toLightX2VResultRef(task_id: string, output_name: string, is_cloud: boolean): LightX2VResultRef {
  return { __type: 'lightx2v_result', task_id, output_name, is_cloud };
}

/** 将多路 in-image 输入扁平为一维图片列表（多连接或多图时每路可能是数组，需合并） */
function flattenImageInput(raw: any): string[] {
  if (raw == null) return [];
  if (Array.isArray(raw)) {
    const flat = raw.flatMap((item: any) =>
      Array.isArray(item) ? item : item != null ? [item] : []
    );
    return flat.filter((x: any) => x != null && typeof x === 'string') as string[];
  }
  return typeof raw === 'string' ? [raw] : [];
}

/** 将本地 URL/路径转为 data URL（同源或 /assets），供云端提交使用 */
async function ensureLocalInputAsDataUrl(input: any): Promise<any> {
  if (typeof input !== 'string') return input;
  if (input.startsWith('data:')) return input;
  if (input.startsWith('//')) return input;

  const isHttp = input.startsWith('http');
  const isLocalAsset = input.includes('/assets/task/result') || input.includes('/assets/workflow/input') || input.startsWith('/assets/');
  const isSameOrigin = typeof window !== 'undefined' && isHttp && input.startsWith(window.location.origin);
  const isLocalPath = !isHttp && input.startsWith('/');

  if (!isLocalPath && !isSameOrigin && !isLocalAsset) return input;

  const url = isHttp ? input : getAssetPath(input);
  try {
    const res = await fetch(url);
    const blob = await res.blob();
    return await new Promise<string>((resolve, reject) => {
      const r = new FileReader();
      r.onloadend = () => resolve(r.result as string);
      r.onerror = reject;
      r.readAsDataURL(blob);
    });
  } catch (e) {
    console.error('[WorkflowExecution] Failed to fetch local input for base64:', input, e);
    return input;
  }
}

async function ensureLocalInputsAsDataUrls(inputs: any[]): Promise<any[]> {
  return Promise.all(inputs.map((v) => ensureLocalInputAsDataUrl(v)));
}

/** 将图片 URL/路径转为 data URL（本地资源优先），供需要 base64 的 API 使用 */
async function ensureImageInputsAsDataUrls(imgs: string[]): Promise<string[]> {
  return Promise.all(imgs.map(async (img) => {
    if (typeof img !== 'string') return img;
    if (img.startsWith('data:')) return img;
    if (img.startsWith('//')) return img;
    const isHttp = img.startsWith('http');
    const isLocalAsset = img.includes('/assets/task/result') || img.includes('/assets/workflow/input');
    const isSameOrigin = typeof window !== 'undefined' && isHttp && img.startsWith(window.location.origin);
    if (isHttp && !isSameOrigin && !isLocalAsset) return img;
    if (!isHttp && !img.startsWith('/')) return img;
    const url = isHttp ? img : getAssetPath(img);
    try {
      const res = await fetch(url);
      const blob = await res.blob();
      return await new Promise<string>((resolve, reject) => {
        const r = new FileReader();
        r.onloadend = () => resolve(r.result as string);
        r.onerror = reject;
        r.readAsDataURL(blob);
      });
    } catch (e) {
      console.error('[WorkflowExecution] Failed to fetch image for text-generation:', img, e);
      return img;
    }
  }));
}

interface UseWorkflowExecutionProps {
  workflow: WorkflowState | null;
  setWorkflow: React.Dispatch<React.SetStateAction<WorkflowState | null>>;
  isPausedRef: React.MutableRefObject<boolean>;
  setIsPaused: (paused: boolean) => void;
  runningTaskIdsRef: React.MutableRefObject<Map<string, string>>;
  abortControllerRef: React.MutableRefObject<AbortController | null>;
  getLightX2VConfig: (workflow: WorkflowState | null) => { url: string; token: string };
  setValidationErrors: (errors: { message: string; type: 'ENV' | 'INPUT' }[]) => void;
  setGlobalError: (error: { message: string; details?: string } | null) => void;
  updateNodeData: (nodeId: string, key: string, value: any) => void;
  voiceList: {
    lightX2VVoiceList: any;
  };
  lang: Language;
  /** 纯前端部署时，执行结束后用此回调持久化到本地，避免调用已禁用的后端 API */
  onSaveExecutionToLocal?: (workflowState: WorkflowState) => Promise<void>;
}

function useWorkflowExecutionImpl({
  workflow,
  setWorkflow,
  isPausedRef,
  setIsPaused,
  runningTaskIdsRef,
  abortControllerRef,
  getLightX2VConfig,
  setValidationErrors,
  setGlobalError,
  updateNodeData,
  voiceList,
  lang,
  onSaveExecutionToLocal
}: UseWorkflowExecutionProps) {
  const { t } = useTranslation(lang);

  const getDescendants = useCallback((nodeId: string, connections: Connection[]): Set<string> => {
    const descendants = new Set<string>();
    const stack = [nodeId];
    while (stack.length > 0) {
      const current = stack.pop()!;
      connections.filter(c => c.sourceNodeId === current).forEach(c => {
        if (!descendants.has(c.targetNodeId)) {
          descendants.add(c.targetNodeId);
          stack.push(c.targetNodeId);
        }
      });
    }
    return descendants;
  }, []);

  const validateWorkflow = useCallback((nodesToRunIds: Set<string>): { message: string; type: 'ENV' | 'INPUT' }[] => {
    if (!workflow) return [];
    const errors: { message: string; type: 'ENV' | 'INPUT' }[] = [];

    const usesLightX2V = Array.from(nodesToRunIds).some(id => {
      const node = workflow.nodes.find(n => n.id === id);
      return node && (node.toolId.includes('lightx2v') || node.toolId.includes('video') || node.toolId === 'avatar-gen' || ((node.toolId === 'text-to-image' || node.toolId === 'image-to-image') && node.data.model?.startsWith('Qwen')));
    });

    if (usesLightX2V && !isStandalone()) {
      const config = getLightX2VConfig(workflow);
      const apiClient = (window as any).__API_CLIENT__;
      if (apiClient) {
        if (!config.token?.trim()) {
          errors.push({ message: t('missing_env_msg'), type: 'ENV' });
        }
      } else {
        if (!config.url?.trim() || !config.token?.trim()) {
          errors.push({ message: t('missing_env_msg'), type: 'ENV' });
        }
      }
    }
    // 仅前端模式：不校验 LIGHTX2V 环境，运行到相关节点时由请求结果决定

    workflow.nodes.forEach(node => {
      if (!nodesToRunIds.has(node.id)) return;
      const tool = TOOLS.find(t => t.id === node.toolId);
      if (!tool) return;

      if (tool.category === 'Input') {
        const val = node.data.value;
        const isEmpty = (Array.isArray(val) && val.length === 0) || !val;
        if (isEmpty) {
          errors.push({
            message: `${node.name ?? (lang === 'zh' ? tool.name_zh : tool.name)} (${t('executing')})`,
            type: 'INPUT'
          });
        }
        return;
      }

      tool.inputs.forEach(port => {
        const isOptional = port.label.toLowerCase().includes('optional') || port.label.toLowerCase().includes('(opt)');
        if (isOptional) return;

        const isConnected = workflow.connections.some(c => c.targetNodeId === node.id && c.targetPortId === port.id);
        const hasGlobalVal = !!workflow.globalInputs[`${node.id}-${port.id}`]?.toString().trim();

        if (!isConnected && !hasGlobalVal) {
          errors.push({
            message: `${node.name ?? (lang === 'zh' ? tool.name_zh : tool.name)} -> ${port.label}`,
            type: 'INPUT'
          });
        }
      });

      // Special validation for voice clone nodes
      if (node.toolId === 'lightx2v-voice-clone') {
        if (!node.data.speakerId) {
          errors.push({
            message: `${node.name ?? (lang === 'zh' ? tool.name_zh : tool.name)}: ${t('select_cloned_voice')}`,
            type: 'INPUT'
          });
        }
      }
    });

    return errors;
  }, [workflow, getLightX2VConfig, t, lang]);

  /** Queue item: may be cancelled; each job has id and affectedNodeIds for per-node cancel and pending UI. */
  type QueueJob = { id: string; startNodeId?: string; onlyOne?: boolean; cancelled?: boolean; affectedNodeIds: Set<string> };
  const executionQueueRef = React.useRef<QueueJob[]>([]);
  const runningJobCountRef = React.useRef(0);
  const MAX_CONCURRENT_JOBS = 3;
  const runningJobsByJobIdRef = React.useRef<Map<string, { affectedNodeIds: Set<string>; taskIdsByNodeId: Map<string, string>; abortController: AbortController }>>(new Map());
  const [pendingRunNodeIds, setPendingRunNodeIds] = React.useState<string[]>([]);

  const getAffectedNodeIds = useCallback((startNodeId?: string, onlyOne?: boolean): Set<string> => {
    if (!workflow) return new Set();
    if (startNodeId) {
      if (onlyOne) return new Set([startNodeId]);
      const desc = getDescendants(startNodeId, workflow.connections);
      desc.add(startNodeId);
      return desc;
    }
    return new Set(workflow.nodes.map(n => n.id));
  }, [workflow, getDescendants]);

  const refreshPendingNodeIds = useCallback(() => {
    const fromQueue = executionQueueRef.current
      .filter(j => !j.cancelled)
      .flatMap(j => [...j.affectedNodeIds]);
    const fromRunning = Array.from(runningJobsByJobIdRef.current.values()).flatMap((j: { affectedNodeIds: Set<string> }) => [...j.affectedNodeIds]);
    setPendingRunNodeIds([...new Set([...fromQueue, ...fromRunning])]);
  }, []);

  /** Resolve LightX2V result ref to a fresh URL via result_url (backend or cloud). Uses cache and proxy when standalone. */
  const resolveLightX2VResultRef = useCallback((ref: LightX2VResultRef): Promise<string> => {
    return resolveLightX2VResultRefUtil(ref);
  }, []);

  /** Runs one job (full or single node). Called by processQueue. Does not set isRunning false (processQueue does when queue empty). */
  const executeOneRun = useCallback(async (jobId: string, startNodeId?: string, onlyOne?: boolean) => {
    if (!workflow) return;

    const jobInfo = runningJobsByJobIdRef.current.get(jobId);
    if (!jobInfo) return;
    const jobAbortSignal = jobInfo.abortController.signal;
    const registerTaskId = (nodeId: string, taskId: string) => {
      runningTaskIdsRef.current.set(nodeId, taskId);
      jobInfo.taskIdsByNodeId.set(nodeId, taskId);
    };

    // Reset pause state when starting a new workflow
    setIsPaused(false);
    isPausedRef.current = false;

    const runStartTime = performance.now();
    let nodesToRunIds: Set<string>;
    if (startNodeId) {
      if (onlyOne) nodesToRunIds = new Set([startNodeId]);
      else {
        nodesToRunIds = getDescendants(startNodeId, workflow.connections);
        nodesToRunIds.add(startNodeId);
      }
    } else {
      nodesToRunIds = new Set(workflow.nodes.map(n => n.id));
    }

    // Clear previous errors immediately on rerun
    setWorkflow(prev => prev ? ({
      ...prev,
      nodes: prev.nodes.map(n => nodesToRunIds.has(n.id)
        ? { ...n, status: NodeStatus.IDLE, error: undefined }
        : n)
    }) : null);

    const errors = validateWorkflow(nodesToRunIds);
    if (errors.length > 0) {
      setValidationErrors(errors);
      return;
    }
    setValidationErrors([]);

    const requiresUserApiKey = workflow.nodes
      .filter(n => nodesToRunIds.has(n.id))
      .some(n =>
        n.toolId.includes('video') ||
        n.toolId === 'avatar-gen' ||
        n.data.model === 'gemini-3-pro-image-preview' ||
        n.data.model === 'gemini-2.5-flash-image'
      );

    if (requiresUserApiKey) {
      try {
        if (!(await (window as any).aistudio.hasSelectedApiKey())) {
          await (window as any).aistudio.openSelectKey();
        }
      } catch (err) {}
    }

    setWorkflow(prev => prev ? ({
      ...prev,
      isRunning: true,
      nodes: prev.nodes.map(n => nodesToRunIds.has(n.id) ? { ...n, status: NodeStatus.PENDING, error: undefined, executionTime: undefined, startTime: undefined } : n)
    }) : null);

    const executedInSession = new Set<string>();
    const sessionOutputs: Record<string, any> = {};

    // 本次运行所需的上游节点（含图片输入等，用于「执行时存库」）
    const nodesNeededAsInputs = new Set<string>();
    workflow.connections.forEach(conn => {
      if (nodesToRunIds.has(conn.targetNodeId) && !nodesToRunIds.has(conn.sourceNodeId)) {
        nodesNeededAsInputs.add(conn.sourceNodeId);
      }
    });

    // If running from a specific node, preserve outputs from nodes that won't be re-run
    if (startNodeId) {
      // Preserve outputs from node.outputValue for nodes that won't be re-run
      workflow.nodes.forEach(n => {
        if (!nodesToRunIds.has(n.id) && n.outputValue != null) sessionOutputs[n.id] = n.outputValue;
      });

      // Load intermediate files for nodes needed as inputs
      if (workflow.id && (workflow.id.startsWith('workflow-') || workflow.id.match(/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i))) {
        const loadPromises: Promise<void>[] = [];

        for (const nodeId of nodesNeededAsInputs) {
          const node = workflow.nodes.find(n => n.id === nodeId);
          if (!node) continue;

          // If we already have the output, skip loading
          if (sessionOutputs[nodeId] !== undefined) continue;

          const tool = TOOLS.find(t => t.id === node.toolId);
          if (!tool) continue;

          // Skip Input nodes：由下方「执行时存库」逻辑统一处理（outputs/save + sessionOutputs）
          if (tool.category === 'Input') continue;

          if (!tool.outputs) continue;

          // Load node output data for each output port (using new unified interface)
          for (const port of tool.outputs) {
            loadPromises.push(
              getNodeOutputData(workflow.id, nodeId, port.id)
                .then(dataUrl => {
                  if (dataUrl) {
                    if (tool.outputs.length === 1) {
                      // Single output node
                      sessionOutputs[nodeId] = dataUrl;
                    } else {
                      // Multi-output node
                      if (!sessionOutputs[nodeId] || typeof sessionOutputs[nodeId] !== 'object') {
                        sessionOutputs[nodeId] = {};
                      }
                      sessionOutputs[nodeId][port.id] = dataUrl;
                    }
                  }
                })
                .catch(err => {
                  console.warn(`[WorkflowExecution] Failed to load node output data for ${nodeId}/${port.id}:`, err);
                })
            );
          }
        }

        // Wait for intermediate files to load (with timeout)
        if (loadPromises.length > 0) {
          await Promise.race([
            Promise.all(loadPromises),
            new Promise(resolve => setTimeout(resolve, 3000))
          ]);
        }

        // Fallback: for non-Input nodes still missing, use node.outputValue (e.g. after connection change, standalone mode)
        for (const nodeId of nodesNeededAsInputs) {
          if (sessionOutputs[nodeId] !== undefined) continue;
          const n = workflow.nodes.find(nn => nn.id === nodeId);
          if (n?.outputValue != null) sessionOutputs[nodeId] = n.outputValue;
        }
      }
    }

    // 输入节点与文本输入一致：执行时才存库。此处若为 data URL 则先 save 再写入 sessionOutputs 为 ref
    if (workflow.id && (isStandalone() || workflow.id.startsWith('workflow-') || workflow.id.match(/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i))) {
      const INPUT_PORT_ID: Record<string, string> = {
        'image-input': 'out-image',
        'audio-input': 'out-audio',
        'video-input': 'out-video'
      };
      // 参与本次运行的输入节点 = 在 nodesToRunIds 中 或 作为上游在 nodesNeededAsInputs 中（如图片输入→图生图）
      const inputNodesToProcess = (node: typeof workflow.nodes[0]) => {
        const tool = TOOLS.find(t => t.id === node.toolId);
        if (!tool || tool.category !== 'Input') return false;
        return nodesToRunIds.has(node.id) || nodesNeededAsInputs.has(node.id);
      };
      for (const node of workflow.nodes) {
        if (!inputNodesToProcess(node)) continue;

        const tool = TOOLS.find(t => t.id === node.toolId)!;

        // 文本输入节点：执行时以 data.value 为准，并同步到 outputValue，避免与界面不一致
        if (node.toolId === 'text-input') {
          const textVal = node.data.value ?? '';
          sessionOutputs[node.id] = textVal;
          if (node.outputValue !== textVal) {
            setWorkflow(prev => {
              if (!prev) return null;
              return { ...prev, nodes: prev.nodes.map(n => n.id === node.id ? { ...n, outputValue: textVal } : n) };
            });
          }
          continue;
        }

        const nodeValue = node.outputValue != null ? node.outputValue : node.data.value;
        if (nodeValue == null) continue;

        const portId = INPUT_PORT_ID[node.toolId];
        const isDataUrl = (v: any) => typeof v === 'string' && v.startsWith('data:');
        const isRef = (v: any) => v && typeof v === 'object' && (v.type === 'file' || v.file_id);

        if (portId && Array.isArray(nodeValue) && nodeValue.length > 0) {
          const newOutputValue: any[] = [];
          let hasNewRefs = false;
          for (const item of nodeValue) {
            if (isDataUrl(item)) {
              try {
                const ref = await saveInputFileViaOutputSave(workflow.id, node.id, portId, item);
                if (ref) {
                  newOutputValue.push(ref);
                  hasNewRefs = true;
                } else {
                  newOutputValue.push(item);
                }
              } catch (e) {
                console.warn('[WorkflowExecution] Save input node file at run time failed:', e);
                newOutputValue.push(item);
              }
            } else {
              newOutputValue.push(item);
            }
          }
          if (hasNewRefs) {
            const runTs = Date.now();
            // 多图：每条 ref 一条历史记录，便于节点展示多张缩略图
            const newEntries: NodeHistoryEntry[] = [];
            for (let i = 0; i < newOutputValue.length; i++) {
              const entry = createHistoryEntryFromValue({
                id: `node-${node.id}-${runTs}-${i}`,
                timestamp: runTs,
                value: newOutputValue[i],
                executionTime: 0
              });
              if (entry) newEntries.push(entry);
            }
            setWorkflow(prev => {
              if (!prev) return null;
              const nextNodes = prev.nodes.map(n => n.id === node.id ? { ...n, outputValue: newOutputValue } : n);
              const nextHistory = { ...(prev.nodeOutputHistory || {}) };
              if (newEntries.length > 0) {
                const prevList = normalizeHistoryEntries(nextHistory[node.id] || []);
                nextHistory[node.id] = [...newEntries, ...prevList].slice(0, MAX_NODE_HISTORY);
              }
              return { ...prev, nodes: nextNodes, nodeOutputHistory: nextHistory };
            });
          }
          sessionOutputs[node.id] = newOutputValue;
        } else if (portId && isDataUrl(nodeValue)) {
          try {
            const ref = await saveInputFileViaOutputSave(workflow.id, node.id, portId, nodeValue);
            if (ref) {
              const runTs = Date.now();
              const entry = createHistoryEntryFromValue({
                id: `node-${node.id}-${runTs}`,
                timestamp: runTs,
                value: ref,
                executionTime: 0
              });
              setWorkflow(prev => {
                if (!prev) return null;
                const nextNodes = prev.nodes.map(n => n.id === node.id ? { ...n, outputValue: ref } : n);
                const nextHistory = { ...(prev.nodeOutputHistory || {}) };
                if (entry) {
                  const prevList = normalizeHistoryEntries(nextHistory[node.id] || []);
                  nextHistory[node.id] = [entry, ...prevList].slice(0, MAX_NODE_HISTORY);
                }
                return { ...prev, nodes: nextNodes, nodeOutputHistory: nextHistory };
              });
              sessionOutputs[node.id] = ref;
            } else {
              sessionOutputs[node.id] = nodeValue;
            }
          } catch (e) {
            console.warn('[WorkflowExecution] Save input node file at run time failed:', e);
            sessionOutputs[node.id] = nodeValue;
          }
        } else if (isRef(nodeValue) || (Array.isArray(nodeValue) && nodeValue.every((v: any) => isRef(v)))) {
          sessionOutputs[node.id] = nodeValue;
        } else {
          sessionOutputs[node.id] = nodeValue;
        }
      }
    }

    // Get LightX2V config once at the start of workflow execution
    const lightX2VConfig = getLightX2VConfig(workflow);

      try {
      // Execute nodes in parallel by layer, with max 3 concurrent executions
      const MAX_CONCURRENT = 3;
      // Accumulate nodeId -> duration across all batches for history entries (state updates are async)
      const executionTimeByNodeId: Record<string, number> = {};

      while (executedInSession.size < workflow.nodes.filter(n => nodesToRunIds.has(n.id)).length) {
        // Check if workflow is paused, wait until resumed
        while (isPausedRef.current) {
          await new Promise(resolve => setTimeout(resolve, 100));
          // Check if workflow is still running (might have been stopped)
          const currentWorkflow = workflow;
          if (!currentWorkflow?.isRunning) {
            return;
          }
        }

        // Find all nodes ready to execute (all inputs are ready)
        const readyNodes: typeof workflow.nodes = [];
        for (const node of workflow.nodes) {
          if (!nodesToRunIds.has(node.id) || executedInSession.has(node.id)) continue;
          const tool = TOOLS.find(t => t.id === node.toolId)!;
          const incomingConns = workflow.connections.filter(c => c.targetNodeId === node.id);
          const inputsReady = incomingConns.every(c => !nodesToRunIds.has(c.sourceNodeId) || executedInSession.has(c.sourceNodeId));

          if (inputsReady) {
            readyNodes.push(node);
          }
        }

        // If no nodes are ready, break to avoid infinite loop
        if (readyNodes.length === 0) break;

        // Execute ready nodes in batches of MAX_CONCURRENT
        for (let i = 0; i < readyNodes.length; i += MAX_CONCURRENT) {
          // Check pause state before starting each batch
          while (isPausedRef.current) {
            await new Promise(resolve => setTimeout(resolve, 100));
            const currentWorkflow = workflow;
            if (!currentWorkflow?.isRunning) {
              return;
            }
          }

          const batch = readyNodes.slice(i, i + MAX_CONCURRENT);

          // Execute batch in parallel
          const executionPromises = batch.map(async (node) => {
            const tool = TOOLS.find(t => t.id === node.toolId)!;
            const incomingConns = workflow.connections.filter(c => c.targetNodeId === node.id);
            const nodeStart = performance.now();

            // Update node status to RUNNING
            setWorkflow(prev => prev ? ({ ...prev, nodes: prev.nodes.map(n => n.id === node.id ? { ...n, status: NodeStatus.RUNNING, startTime: nodeStart } : n) }) : null);

            try {
              const nodeInputs: Record<string, any> = {};
              await Promise.all(tool.inputs.map(async (port) => {
                // Check if there's an override value for this port
                if (node.data.inputOverrides && node.data.inputOverrides[port.id] !== undefined) {
                  nodeInputs[port.id] = node.data.inputOverrides[port.id];
                  return;
                }

                const conns = incomingConns.filter(c => c.targetPortId === port.id);
                if (conns.length > 0) {
                  const values = (await Promise.all(conns.map(async (c) => {
                  const sourceNode = workflow.nodes.find(n => n.id === c.sourceNodeId);
                  // 文本生成大模型输出若为对象（字典），传入下游时自动转为 JSON 字符串
                  const ensureStringIfObjectFromTextGen = (raw: any) => {
                    if (sourceNode?.toolId === 'text-generation' && typeof raw === 'object' && raw !== null && !Array.isArray(raw))
                      return JSON.stringify(raw);
                    return raw;
                  };
                  // First check if source node has output in sessionOutputs (incl. input nodes after run-time save)
                  if (sessionOutputs[c.sourceNodeId] !== undefined) {
                    const sourceRes = sessionOutputs[c.sourceNodeId];
                    let raw = (typeof sourceRes === 'object' && sourceRes !== null && c.sourcePortId in sourceRes) ? sourceRes[c.sourcePortId] : sourceRes;
                    if (isLightX2VResultRef(raw)) raw = await resolveLightX2VResultRef(raw);
                    // 输入节点等可能为 file ref：解析为 data URL 再传给下游（flattenImageInput 等只认 string）
                    const isRef = (v: any) => v && typeof v === 'object' && (v.type === 'file' || v.file_id);
                    if (isRef(raw)) {
                      const idToFetch = (raw as { file_url?: string }).file_url?.startsWith('local://')
                        ? (raw as { file_url: string }).file_url
                        : (raw as { file_id: string }).file_id;
                      raw = await getWorkflowFileByFileId(workflow.id, idToFetch) || (raw as { file_url?: string }).file_url || raw;
                    } else if (Array.isArray(raw) && raw.some((v: any) => isRef(v))) {
                      raw = await Promise.all(raw.map(async (v: any) => {
                        if (!isRef(v)) return v;
                        const id = (v as { file_url?: string }).file_url?.startsWith('local://') ? (v as { file_url: string }).file_url : (v as { file_id: string }).file_id;
                        return getWorkflowFileByFileId(workflow.id, id) || (v as { file_url?: string }).file_url || v;
                      }));
                    }
                    return ensureStringIfObjectFromTextGen(raw);
                  }
                  // If not executed yet, check if it's an input node and read from node.data.value
                  // This handles the case where input nodes haven't been executed but their values are needed
                  if (sourceNode) {
                    const sourceTool = TOOLS.find(t => t.id === sourceNode.toolId);
                    if (sourceTool?.category === 'Input') {
                      // For input nodes, read directly from node.data.value
                      let inputValue = sourceNode.data.value;

                      // Convert file paths to base64 data URLs for image and audio inputs
                      if (sourceNode.toolId === 'image-input' && Array.isArray(inputValue) && inputValue.length > 0) {
                        inputValue = await Promise.all(inputValue.map(async (img: string | { type?: string; file_id?: string; file_url?: string }) => {
                          // Backend file reference: resolve file_id to data URL
                          if (img && typeof img === 'object' && (img.type === 'file' || img.file_id)) {
                            const dataUrl = await getWorkflowFileByFileId(workflow.id, (img as { file_id: string }).file_id);
                            return dataUrl || (img as { file_url?: string }).file_url || img;
                          }
                          if (typeof img !== 'string') return img;
                          // If it's already a data URL or base64, return as is
                          if (img.startsWith('data:') || (!img.startsWith('http') && img.includes(','))) {
                            return img;
                          }
                          // Standalone: resolve local:// from IndexedDB
                          if (img.startsWith('local://')) {
                            const dataUrl = await getWorkflowFileByFileId(workflow.id, img);
                            return dataUrl || img;
                          }
                          // If it's a file path (starts with /), load and convert to base64
                          if (img.startsWith('/')) {
                            try {
                              // 修复资源路径：如果在 qiankun 环境，确保路径包含 /canvas/
                              let imagePath = img;
                              const basePath = getAssetBasePath();
                              if (img.startsWith('/assets/') && !img.startsWith('/canvas/')) {
                                imagePath = `${basePath}${img}`;
                              }
                              const response = await fetch(imagePath);
                              const blob = await response.blob();
                              return await new Promise<string>((resolve, reject) => {
                                const reader = new FileReader();
                                reader.onloadend = () => resolve(reader.result as string);
                                reader.onerror = reject;
                                reader.readAsDataURL(blob);
                              });
                            } catch (e) {
                              console.error(`Failed to load image ${img}:`, e);
                              return img; // Return original path if loading fails
                            }
                          }
                          return img;
                        }));
                      } else if (sourceNode.toolId === 'audio-input' && inputValue) {
                        if (inputValue && typeof inputValue === 'object' && (inputValue as { type?: string; file_id?: string }).file_id) {
                          const dataUrl = await getWorkflowFileByFileId(workflow.id, (inputValue as { file_id: string }).file_id);
                          if (dataUrl) inputValue = dataUrl;
                        } else if (typeof inputValue === 'string' && inputValue.startsWith('local://')) {
                          const dataUrl = await getWorkflowFileByFileId(workflow.id, inputValue);
                          if (dataUrl) inputValue = dataUrl;
                        } else if (typeof inputValue === 'string' && inputValue.startsWith('/')) {
                        // Convert audio file path to base64 data URL
                        // 修复资源路径：如果在 qiankun 环境，确保路径包含 /canvas/
                        let audioPath = inputValue;
                        const basePath = getAssetBasePath();
                        if (inputValue.startsWith('/assets/') && !inputValue.startsWith('/canvas/')) {
                          audioPath = `${basePath}${inputValue}`;
                        }
                        try {
                          const response = await fetch(audioPath);
                          const blob = await response.blob();
                          inputValue = await new Promise<string>((resolve, reject) => {
                            const reader = new FileReader();
                            reader.onloadend = () => resolve(reader.result as string);
                            reader.onerror = reject;
                            reader.readAsDataURL(blob);
                          });
                        } catch (e) {
                          console.error(`Failed to load audio ${inputValue}:`, e);
                          // Keep original path if loading fails
                        }
                        }
                      } else if (sourceNode.toolId === 'video-input' && inputValue) {
                        if (inputValue && typeof inputValue === 'object' && (inputValue as { type?: string; file_id?: string }).file_id) {
                          const dataUrl = await getWorkflowFileByFileId(workflow.id, (inputValue as { file_id: string }).file_id);
                          if (dataUrl) inputValue = dataUrl;
                        } else if (typeof inputValue === 'string' && inputValue.startsWith('local://')) {
                          const dataUrl = await getWorkflowFileByFileId(workflow.id, inputValue);
                          if (dataUrl) inputValue = dataUrl;
                        } else if (typeof inputValue === 'string' && inputValue.startsWith('/')) {
                          let videoPath = inputValue;
                          const basePath = getAssetBasePath();
                          if (inputValue.startsWith('/assets/') && !inputValue.startsWith('/canvas/')) {
                            videoPath = `${basePath}${inputValue}`;
                          }
                          try {
                            const response = await fetch(videoPath);
                            const blob = await response.blob();
                            inputValue = await new Promise<string>((resolve, reject) => {
                              const reader = new FileReader();
                              reader.onloadend = () => resolve(reader.result as string);
                              reader.onerror = reject;
                              reader.readAsDataURL(blob);
                            });
                          } catch (e) {
                            console.error(`Failed to load video ${inputValue}:`, e);
                          }
                        }
                      }

                      // Check if this is a multi-output node (like text-generation with customOutputs)
                      if (sourceNode.toolId === 'text-generation' && sourceNode.data.customOutputs && typeof inputValue === 'object' && inputValue !== null) {
                        const raw = c.sourcePortId in inputValue ? inputValue[c.sourcePortId] : inputValue;
                        return ensureStringIfObjectFromTextGen(raw);
                      }
                      return inputValue;
                    }
                    // For other nodes that haven't executed, try node.outputValue (e.g. from load or previous run)
                    const prevOutput = sourceNode?.outputValue;
                    if (prevOutput !== undefined) {
                      const raw = (typeof prevOutput === 'object' && prevOutput !== null && c.sourcePortId in prevOutput) ? prevOutput[c.sourcePortId] : prevOutput;
                      return ensureStringIfObjectFromTextGen(raw);
                    }
              }
              return undefined;
            }))).filter(v => v !== undefined).flat();
              nodeInputs[port.id] = values.length === 1 ? values[0] : values.length > 0 ? values : undefined;
                } else nodeInputs[port.id] = workflow.globalInputs[`${node.id}-${port.id}`];
              }));

              let result: any;
              const model = node.data.model;
              switch (node.toolId) {
                case 'text-input': result = node.data.value || ""; break;
                case 'image-input': {
                  const rawImage = node.outputValue ?? node.data.value;
                  const imageValue = Array.isArray(rawImage) ? rawImage : (rawImage != null ? [rawImage] : []);
                  // For workflow input paths or URLs, use directly (no conversion needed)
                  // Only convert local file paths (starting with /) that are not workflow/task paths
                  if (Array.isArray(imageValue) && imageValue.length > 0) {
                    const convertedImages = await Promise.all(imageValue.map(async (img: string | { type?: string; file_id?: string; file_url?: string }) => {
                      // Backend/standalone file reference: resolve to data URL (file_id 或 local:// 的 file_url)
                      if (img && typeof img === 'object' && (img.type === 'file' || (img as { file_id?: string }).file_id)) {
                        const idToFetch = (img as { file_url?: string }).file_url?.startsWith('local://')
                          ? (img as { file_url: string }).file_url
                          : (img as { file_id: string }).file_id;
                        const dataUrl = await getWorkflowFileByFileId(workflow.id, idToFetch);
                        return dataUrl || (img as { file_url?: string }).file_url || img;
                      }
                      if (typeof img !== 'string') return img;
                      // If it's already a data URL or base64, return as is
                      if (img.startsWith('data:') || (!img.startsWith('http') && img.includes(','))) {
                        return img;
                      }
                      // Standalone: resolve local:// from IndexedDB
                      if (img.startsWith('local://')) {
                        const dataUrl = await getWorkflowFileByFileId(workflow.id, img);
                        return dataUrl || img;
                      }
                      // If it's a workflow input path or task result path, use directly
                      if (img.includes('/assets/workflow/input') || img.includes('/assets/task/') ||
                          img.startsWith('http://') || img.startsWith('https://')) {
                        return img;
                      }
                      // If it's a local file path (starts with /) that's not a workflow/task path, load and convert to base64
                      // This is for backward compatibility with old file paths
                      if (img.startsWith('/')) {
                        try {
                          const response = await fetch(getAssetPath(img));
                          const blob = await response.blob();
                          return await new Promise<string>((resolve, reject) => {
                            const reader = new FileReader();
                            reader.onloadend = () => resolve(reader.result as string);
                            reader.onerror = reject;
                            reader.readAsDataURL(blob);
                          });
                        } catch (e) {
                          console.error(`Failed to load image ${img}:`, e);
                          return img; // Return original path if loading fails
                        }
                      }
                      // For other cases, return as is
                      return img;
                    }));
                    result = convertedImages;
                  } else {
                    result = imageValue;
                  }
                  break;
                }
                case 'audio-input': {
                  const audioValue = node.outputValue ?? node.data.value;
                  if (audioValue && typeof audioValue === 'object' && (audioValue as { file_id?: string }).file_id) {
                    const idToFetch = (audioValue as { file_url?: string }).file_url?.startsWith('local://')
                      ? (audioValue as { file_url: string }).file_url
                      : (audioValue as { file_id: string }).file_id;
                    result = await getWorkflowFileByFileId(workflow.id, idToFetch) || (audioValue as { file_url?: string }).file_url || audioValue;
                  } else if (audioValue && typeof audioValue === 'string') {
                    if (audioValue.startsWith('local://')) {
                      result = await getWorkflowFileByFileId(workflow.id, audioValue) || audioValue;
                    } else if (audioValue.startsWith('data:') || (!audioValue.startsWith('http') && audioValue.includes(','))) {
                      result = audioValue;
                    } else if (audioValue.includes('/assets/workflow/input') || audioValue.includes('/assets/task/') ||
                               audioValue.startsWith('http://') || audioValue.startsWith('https://')) {
                      result = audioValue;
                    } else if (audioValue.startsWith('/')) {
                      // If it's a local file path (starts with /) that's not a workflow/task path, load and convert to base64
                      // This is for backward compatibility with old file paths
                      try {
                        const response = await fetch(getAssetPath(audioValue));
                        const blob = await response.blob();
                        result = await new Promise<string>((resolve, reject) => {
                          const reader = new FileReader();
                          reader.onloadend = () => resolve(reader.result as string);
                          reader.onerror = reject;
                          reader.readAsDataURL(blob);
                        });
                      } catch (e) {
                        console.error(`Failed to load audio ${audioValue}:`, e);
                        result = audioValue; // Return original path if loading fails
                      }
                    } else {
                      // For other cases, return as is
                      result = audioValue;
                    }
                  } else {
                    result = audioValue;
                  }
                  break;
                }
                case 'video-input': {
                  const vidVal = node.outputValue ?? node.data.value;
                  if (vidVal && typeof vidVal === 'object' && (vidVal as { file_id?: string }).file_id) {
                    const idToFetch = (vidVal as { file_url?: string }).file_url?.startsWith('local://')
                      ? (vidVal as { file_url: string }).file_url
                      : (vidVal as { file_id: string }).file_id;
                    result = await getWorkflowFileByFileId(workflow.id, idToFetch) || (vidVal as { file_url?: string }).file_url || vidVal;
                  } else if (typeof vidVal === 'string') {
                    if (vidVal.startsWith('local://')) {
                      result = await getWorkflowFileByFileId(workflow.id, vidVal) || vidVal;
                    } else if (vidVal.startsWith('data:') || (!vidVal.startsWith('http') && vidVal.includes(','))) {
                      result = vidVal;
                    } else if (vidVal.includes('/assets/workflow/input') || vidVal.includes('/assets/task/') ||
                               vidVal.startsWith('http://') || vidVal.startsWith('https://')) {
                      result = vidVal;
                    } else if (vidVal.startsWith('/')) {
                      try {
                        const basePath = getAssetBasePath();
                        const videoPath = vidVal.startsWith('/assets/') && !vidVal.startsWith('/canvas/') ? `${basePath}${vidVal}` : vidVal;
                        const response = await fetch(videoPath);
                        const blob = await response.blob();
                        result = await new Promise<string>((resolve, reject) => {
                          const reader = new FileReader();
                          reader.onloadend = () => resolve(reader.result as string);
                          reader.onerror = reject;
                          reader.readAsDataURL(blob);
                        });
                      } catch (e) {
                        console.error(`Failed to load video ${vidVal}:`, e);
                        result = vidVal;
                      }
                    } else {
                      result = vidVal;
                    }
                  } else {
                    result = vidVal;
                  }
                  break;
                }
                case 'web-search': result = await geminiText(nodeInputs['in-text'] || "Search query", true, 'basic', undefined, model); break;
                case 'text-generation':
                  const outputFields = (node.data.customOutputs || []).map((o: any) => ({ id: o.id, description: o.description || o.label }));
                  const useSearch = node.data.useSearch || false;
                  // Use DeepSeek for deepseek models, Doubao for doubao models, PP Chat for ppchat models, otherwise use Gemini
                  if (model && model.startsWith('deepseek-')) {
                    result = await deepseekText(nodeInputs['in-text'] || "...", node.data.mode, node.data.customInstruction, model, outputFields, useSearch);
                  } else if (model && model.startsWith('doubao-')) {
                    const imageInputRaw = flattenImageInput(nodeInputs['in-image']);
                    const imageInput = imageInputRaw.length > 0 ? await ensureImageInputsAsDataUrls(imageInputRaw) : undefined;
                    result = await doubaoText(nodeInputs['in-text'] || "...", node.data.mode, node.data.customInstruction, model, outputFields, imageInput?.length ? imageInput : undefined, useSearch);
                  } else if (model && model.startsWith('ppchat-')) {
                    const imageInputRaw = flattenImageInput(nodeInputs['in-image']);
                    const imageInput = imageInputRaw.length > 0 ? await ensureImageInputsAsDataUrls(imageInputRaw) : undefined;
                    result = await ppchatGeminiText(nodeInputs['in-text'] || "...", node.data.mode, node.data.customInstruction, model.replace('ppchat-', ''), outputFields, imageInput?.length ? imageInput : undefined);
                  } else {
                    const imageInputRaw = flattenImageInput(nodeInputs['in-image']);
                    const imageInput = imageInputRaw.length > 0 ? await ensureImageInputsAsDataUrls(imageInputRaw) : undefined;
                    result = await geminiText(nodeInputs['in-text'] || "...", false, node.data.mode, node.data.customInstruction, model, outputFields, imageInput?.length ? imageInput : undefined);
                  }
                  break;
                case 'text-to-image':
                  if (model === 'gemini-2.5-flash-image') {
                    result = await geminiImage(nodeInputs['in-text'] || "Artistic portrait", undefined, node.data.aspectRatio, model);
                  } else {
                    // Get config for this specific model (handles -cloud suffix)
                    const t2iModelConfig = getLightX2VConfigForModel(model || 'Qwen-Image-2512');
                    console.log("[LightX2V] Calling lightX2VTask for text-to-image");
                    result = await lightX2VTask(
                      t2iModelConfig.url,
                      t2iModelConfig.token,
                      't2i',
                      model || 'Qwen-Image-2512',
                      nodeInputs['in-text'] || "",
                      undefined, undefined, undefined,
                      'output_image',
                      node.data.aspectRatio,
                      undefined,
                      (taskId) => registerTaskId(node.id, taskId),
                      jobAbortSignal
                    );
                    const t2iTid = jobInfo.taskIdsByNodeId.get(node.id);
                    if (t2iTid) result = toLightX2VResultRef(t2iTid, 'output_image', (model || '').endsWith('-cloud'));
                  }
                  break;
                case 'image-to-image':
                  if (model === 'gemini-2.5-flash-image') {
                    const geminiImgs = flattenImageInput(nodeInputs['in-image']);
                    result = await geminiImage(nodeInputs['in-text'] || "Transform", geminiImgs.length > 0 ? geminiImgs : undefined, node.data.aspectRatio || "1:1", model);
                  } else {
                    // For LightX2V i2i: 多路 in-image 扁平为一维；参考主应用 other.js，多图统一用 base64 提交（后端不接受 type=url 的数组）
                    const i2iImgs = flattenImageInput(nodeInputs['in-image']);
                    const i2iImgsBase64 = await Promise.all(i2iImgs.map(async (img) => {
                      if (typeof img !== 'string') return img;
                      if (img.startsWith('data:') || (!img.startsWith('http') && !img.startsWith('/'))) return img;
                      const url = (img.startsWith('/') && !img.startsWith('//')) ? getAssetPath(img) : img;
                      try {
                        const res = await fetch(url);
                        const blob = await res.blob();
                        return await new Promise<string>((resolve, reject) => {
                          const r = new FileReader();
                          r.onloadend = () => resolve(r.result as string);
                          r.onerror = reject;
                          r.readAsDataURL(blob);
                        });
                      } catch (e) {
                        console.error('[WorkflowExecution] Failed to fetch image for i2i:', img, e);
                        return img;
                      }
                    }));
                    // Backend supports multi-image (single or array of URL/base64); use first image only if backend did not support multi
                    const imageInput = i2iImgsBase64.length === 0 ? undefined : (i2iImgsBase64.length === 1 ? i2iImgsBase64[0] : i2iImgsBase64);
                    // Get config for this specific model (handles -cloud suffix)
                    const i2iModelConfig = getLightX2VConfigForModel(model || 'Qwen-Image-Edit-2511');
                    result = await lightX2VTask(
                      i2iModelConfig.url,
                      i2iModelConfig.token,
                      'i2i',
                      model || 'Qwen-Image-Edit-2511',
                      nodeInputs['in-text'] || "",
                      imageInput,
                      undefined,
                      undefined,
                      'output_image',
                      node.data.aspectRatio,
                      undefined,
                      (taskId) => registerTaskId(node.id, taskId),
                      jobAbortSignal
                    );
                    const i2iTid = jobInfo.taskIdsByNodeId.get(node.id);
                    if (i2iTid) result = toLightX2VResultRef(i2iTid, 'output_image', (model || '').endsWith('-cloud'));
                  }
                  break;
                case 'gemini-watermark-remover':
                  const watermarkImg = Array.isArray(nodeInputs['in-image']) ? nodeInputs['in-image'][0] : nodeInputs['in-image'];
                  if (!watermarkImg) throw new Error("Image input is required for watermark removal");
                  result = await removeGeminiWatermark(watermarkImg);
                  break;
                case 'tts':
                  // Determine which service to use based on model
                  const isLightX2V = model === 'lightx2v' || model?.startsWith('lightx2v');

                  if (isLightX2V) {
                    // Use LightX2V TTS
                    const voiceTypeToUse = node.data.voiceType || 'zh_female_vv_uranus_bigtts';
                    let resourceIdToUse = node.data.resourceId;

                    // Always try to match resource_id from voice list to ensure correctness
                    if (voiceList.lightX2VVoiceList?.voices && voiceList.lightX2VVoiceList.voices.length > 0) {
                      const matchingVoice = voiceList.lightX2VVoiceList.voices.find((v: any) => v.voice_type === voiceTypeToUse);
                      if (matchingVoice?.resource_id) {
                        resourceIdToUse = matchingVoice.resource_id;
                        // Update node data with correct resource_id for future use
                        if (!node.data.resourceId || node.data.resourceId !== resourceIdToUse) {
                          updateNodeData(node.id, 'resourceId', resourceIdToUse);
                          console.log(`[LightX2V] Matched resource_id: ${resourceIdToUse} for voice: ${voiceTypeToUse}`);
                        }
                      } else {
                        console.warn(`[LightX2V] No matching voice found for voice_type: ${voiceTypeToUse}`);
                      }
                    } else {
                      console.warn(`[LightX2V] Voice list not loaded, using stored resourceId: ${resourceIdToUse || 'none'}`);
                    }

                    // Fallback to default if still not found
                    if (!resourceIdToUse) {
                      resourceIdToUse = "seed-tts-1.0";
                      console.warn(`[LightX2V] Using fallback resourceId: ${resourceIdToUse}`);
                    }

                    const contextTone = nodeInputs['in-context-tone'] || "";
                  result = await lightX2VTTS(
                      lightX2VConfig.url,
                      lightX2VConfig.token,
                    nodeInputs['in-text'] || "",
                      voiceTypeToUse,
                      contextTone,
                    node.data.emotion || "",
                    node.data.emotionScale || 3,
                    node.data.speechRate || 0,
                    node.data.pitch || 0,
                    node.data.loudnessRate || 0,
                      resourceIdToUse
                    );
                  } else {
                    // Use Gemini TTS
                    const contextTone = nodeInputs['in-context-tone'] || "";
                    result = await geminiSpeech(
                      nodeInputs['in-text'] || "Script",
                      node.data.voice || "Kore",
                      model || 'gemini-2.5-flash-preview-tts',
                      contextTone
                    );
                  }
                  break;
                case 'lightx2v-voice-clone':
                  // Use selected speaker_id from node data
                  const speakerId = node.data.speakerId;

                  if (!speakerId) {
                    throw new Error("Please select a cloned voice. Use the node settings to choose or create a new cloned voice.");
                  }

                  // Generate TTS with cloned voice
                  const ttsText = nodeInputs['in-tts-text'] || nodeInputs['in-text'] || "";
                  if (!ttsText) throw new Error("TTS text is required");
                  result = await lightX2VVoiceCloneTTS(
                    lightX2VConfig.url,
                    lightX2VConfig.token,
                    ttsText,
                    speakerId,
                    node.data.style || "正常",
                    node.data.speed || 1.0,
                    node.data.volume || 0,
                    node.data.pitch || 0,
                    node.data.language || "ZH_CN"
                  );
                  break;
                case 'video-gen-text':
                  // Get config for this specific model (handles -cloud suffix)
                  const t2vModelConfig = getLightX2VConfigForModel(model || 'Wan2.2_T2V_A14B_distilled');
                  result = await lightX2VTask(
                      t2vModelConfig.url,
                      t2vModelConfig.token,
                    't2v',
                    model || 'Wan2.2_T2V_A14B_distilled',
                    nodeInputs['in-text'] || "",
                    undefined, undefined, undefined,
                    'output_video',
                    node.data.aspectRatio,
                    undefined,
                    (taskId) => registerTaskId(node.id, taskId),
                    jobAbortSignal
                  );
                  const t2vTid = jobInfo.taskIdsByNodeId.get(node.id);
                  if (t2vTid) result = toLightX2VResultRef(t2vTid, 'output_video', (model || '').endsWith('-cloud'));
                  break;
                case 'video-gen-image': {
                  const startImg = Array.isArray(nodeInputs['in-image']) ? nodeInputs['in-image'][0] : nodeInputs['in-image'];
                  // Get config for this specific model (handles -cloud suffix)
                  const i2vModelConfig = getLightX2VConfigForModel(model || 'Wan2.2_I2V_A14B_distilled');
                  const startImgForSubmit = i2vModelConfig.isCloud
                    ? (await ensureLocalInputsAsDataUrls(flattenImageInput(startImg)))[0]
                    : startImg;
                  result = await lightX2VTask(
                    i2vModelConfig.url,
                    i2vModelConfig.token,
                    'i2v',
                    model || 'Wan2.2_I2V_A14B_distilled',
                    nodeInputs['in-text'] || "",
                    startImgForSubmit,
                    undefined, undefined,
                    'output_video',
                    node.data.aspectRatio,
                    undefined,
                    (taskId) => registerTaskId(node.id, taskId),
                    jobAbortSignal
                  );
                  const i2vTid = jobInfo.taskIdsByNodeId.get(node.id);
                  if (i2vTid) result = toLightX2VResultRef(i2vTid, 'output_video', (model || '').endsWith('-cloud'));
                  break;
                }
                case 'video-gen-dual-frame': {
                    const dualStart = Array.isArray(nodeInputs['in-image-start']) ? nodeInputs['in-image-start'][0] : nodeInputs['in-image-start'];
                    const dualEnd = Array.isArray(nodeInputs['in-image-end']) ? nodeInputs['in-image-end'][0] : nodeInputs['in-image-end'];
                    // Get config for this specific model (handles -cloud suffix)
                    const flf2vModelConfig = getLightX2VConfigForModel(model || 'Wan2.2_I2V_A14B_distilled');
                    const dualStartForSubmit = flf2vModelConfig.isCloud
                      ? (await ensureLocalInputsAsDataUrls(flattenImageInput(dualStart)))[0]
                      : dualStart;
                    const dualEndForSubmit = flf2vModelConfig.isCloud
                      ? (await ensureLocalInputsAsDataUrls(flattenImageInput(dualEnd)))[0]
                      : dualEnd;
                    result = await lightX2VTask(
                        flf2vModelConfig.url,
                        flf2vModelConfig.token,
                        'flf2v',
                        model || 'Wan2.2_I2V_A14B_distilled',
                        nodeInputs['in-text'] || "",
                        dualStartForSubmit,
                        undefined,
                        dualEndForSubmit,
                        'output_video',
                        node.data.aspectRatio,
                        undefined,
                        (taskId) => registerTaskId(node.id, taskId),
                        jobAbortSignal
                    );
                    const flf2vTid = jobInfo.taskIdsByNodeId.get(node.id);
                    if (flf2vTid) result = toLightX2VResultRef(flf2vTid, 'output_video', (model || '').endsWith('-cloud'));
                    break;
                }
                case 'character-swap':
                  const swapImg = Array.isArray(nodeInputs['in-image']) ? nodeInputs['in-image'][0] : nodeInputs['in-image'];
                  const swapVid = Array.isArray(nodeInputs['in-video']) ? nodeInputs['in-video'][0] : nodeInputs['in-video'];

                  // Use LightX2V animate task for wan2.2_animate model, otherwise use Gemini
                  if (model === 'wan2.2_animate' || model?.endsWith('-cloud')) {
                    // Get config for this specific model (handles -cloud suffix)
                    const animateModelConfig = getLightX2VConfigForModel(model);
                    const swapImgForSubmit = animateModelConfig.isCloud ? await ensureLocalInputAsDataUrl(swapImg) : swapImg;
                    const swapVidForSubmit = animateModelConfig.isCloud ? await ensureLocalInputAsDataUrl(swapVid) : swapVid;
                    result = await lightX2VTask(
                      animateModelConfig.url,
                      animateModelConfig.token,
                      'animate',
                      model,
                      nodeInputs['in-text'] || "",
                      swapImgForSubmit,
                      undefined, undefined,
                      'output_video',
                      node.data.aspectRatio,
                      swapVidForSubmit,
                      (taskId) => registerTaskId(node.id, taskId),
                      jobAbortSignal
                    );
                    const animateTid = jobInfo.taskIdsByNodeId.get(node.id);
                    if (animateTid) result = toLightX2VResultRef(animateTid, 'output_video', (model || '').endsWith('-cloud'));
                  } else {
                  result = await geminiVideo(nodeInputs['in-text'] || "Swap character", swapImg, "16:9", "720p", swapVid, model);
                  }
                  break;
                case 'avatar-gen':
                  const avatarImg = Array.isArray(nodeInputs['in-image']) ? nodeInputs['in-image'][0] : nodeInputs['in-image'];
                  const avatarAudio = Array.isArray(nodeInputs['in-audio']) ? nodeInputs['in-audio'][0] : nodeInputs['in-audio'];
                  // Get config for this specific model (handles -cloud suffix)
                  const s2vModelConfig = getLightX2VConfigForModel(model || "SekoTalk");
                  const avatarImgForSubmit = s2vModelConfig.isCloud ? await ensureLocalInputAsDataUrl(avatarImg) : avatarImg;
                  const avatarAudioForSubmit = s2vModelConfig.isCloud ? await ensureLocalInputAsDataUrl(avatarAudio) : avatarAudio;
                  result = await lightX2VTask(
                    s2vModelConfig.url,
                    s2vModelConfig.token,
                    's2v',
                    model || "SekoTalk",
                    nodeInputs['in-text'] || "A person talking naturally.",
                    avatarImgForSubmit || "",
                    avatarAudioForSubmit || "",
                    undefined,
                    'output_video',
                    undefined,
                    undefined,
                    (taskId) => registerTaskId(node.id, taskId),
                    jobAbortSignal
                  );
                  const s2vTid = jobInfo.taskIdsByNodeId.get(node.id);
                  if (s2vTid) result = toLightX2VResultRef(s2vTid, 'output_video', (model || '').endsWith('-cloud'));
                  break;
                default: result = "Processed";
              }
              const nodeDuration = performance.now() - nodeStart;

            // Store result in sessionOutputs (need to handle race condition)
            // Use a function to safely update sessionOutputs
            return { nodeId: node.id, result, duration: nodeDuration };
            } catch (err: any) {
              const nodeDuration = performance.now() - nodeStart;
              if (err.message?.includes("Requested entity was not found")) {
                await (window as any).aistudio.openSelectKey();
              }
              setWorkflow(prev => prev ? ({
                ...prev,
                nodes: prev.nodes.map(n => n.id === node.id ? {
                  ...n,
                  status: NodeStatus.ERROR,
                  error: err.message || 'Unknown execution error',
                  executionTime: nodeDuration,
                  completedAt: Date.now()
                } : n)
              }) : null);
            throw { nodeId: node.id, error: err, duration: nodeDuration };
          }
          });

          // Wait for all nodes in this batch to complete
          const results = await Promise.allSettled(executionPromises);

          // Process results and update state - batch updates to reduce re-renders
          const successfulResults: Array<{ nodeId: string; result: any; duration: number }> = [];
          const failedNodes: Array<{ nodeId: string; error: any; duration: number }> = [];

          // Fill nodeId -> duration for this batch (executionTimeByNodeId is declared above the loop)
          results.forEach((settledResult, index) => {
            const node = batch[index];
            if (settledResult.status === 'fulfilled') {
              const { nodeId, result, duration } = settledResult.value;
              executionTimeByNodeId[nodeId] = duration;
              sessionOutputs[nodeId] = result;
              successfulResults.push({ nodeId, result, duration });
              executedInSession.add(nodeId);

              // Note: Immediate save is skipped here - we'll save all outputs together after execution completes
              // This ensures we have the correct outputValue set and can save with runId
            } else {
              const errorInfo = settledResult.reason;
              if (errorInfo && errorInfo.error) {
                // Error was already handled in the catch block, just mark as executed
                executedInSession.add(errorInfo.nodeId);
                executionTimeByNodeId[errorInfo.nodeId] = errorInfo.duration || 0;
                // Ensure error is a string
                const errorMessage = errorInfo.error instanceof Error
                  ? errorInfo.error.message
                  : (typeof errorInfo.error === 'string'
                      ? errorInfo.error
                      : String(errorInfo.error || 'Unknown execution error'));
                failedNodes.push({ nodeId: errorInfo.nodeId, error: errorMessage, duration: errorInfo.duration || 0 });
              } else {
                // Unhandled error
                const nodeDuration = performance.now() - (node.startTime || performance.now());
                executionTimeByNodeId[node.id] = nodeDuration;
                const errorMessage = settledResult.reason instanceof Error
                  ? settledResult.reason.message
                  : (typeof settledResult.reason === 'string'
                      ? settledResult.reason
                      : 'Unknown execution error');
                failedNodes.push({ nodeId: node.id, error: errorMessage, duration: nodeDuration });
                executedInSession.add(node.id);
              }
            }
          });

          // Batch update state for successful results (outputValue is the single source of truth)
          if (successfulResults.length > 0) {
            setWorkflow(prev => {
              if (!prev) return null;
              const updatedNodes = [...prev.nodes];
              successfulResults.forEach(({ nodeId, result, duration }) => {
                const idx = updatedNodes.findIndex(n => n.id === nodeId);
                if (idx >= 0) {
                  updatedNodes[idx] = { ...updatedNodes[idx], status: NodeStatus.SUCCESS, executionTime: duration, outputValue: result, completedAt: Date.now() };
                }
              });
              return { ...prev, nodes: updatedNodes };
            });
          }

          // Batch update state for failed nodes
          if (failedNodes.length > 0) {
            setWorkflow(prev => {
              if (!prev) return null;
              const updatedNodes = [...prev.nodes];
              failedNodes.forEach(({ nodeId, error, duration }) => {
                const index = updatedNodes.findIndex(n => n.id === nodeId);
                if (index >= 0) {
                  updatedNodes[index] = {
                    ...updatedNodes[index],
                    status: NodeStatus.ERROR,
                    error,
                    executionTime: duration,
                    completedAt: updatedNodes[index].completedAt ?? Date.now()
                  };
                }
              });
              return { ...prev, nodes: updatedNodes };
            });
          }
        }
      }
      const runTotalTime = performance.now() - runStartTime;
      const runTimestamp = Date.now();

      // Optimize history storage: only keep essential data
      // Create a lightweight snapshot for nodeOutputHistory entry
      const lightweightNodesSnapshot = workflow.nodes.map(n => ({
        id: n.id,
        toolId: n.toolId,
        x: n.x,
        y: n.y,
        status: n.status,
        data: { ...n.data },
        error: n.error,
        executionTime: n.executionTime,
        completedAt: n.completedAt
      }));

      // Optimize outputs: don't save full base64 in history, save references / small data
      const historyReadyOutputs: Record<string, any> = {};
      for (const [nodeId, output] of Object.entries(sessionOutputs)) {
        if (typeof output === 'string' && output.startsWith('data:')) {
          historyReadyOutputs[nodeId] = {
            type: 'data_url',
            data: output.substring(0, 100) + '...',
            _full_data: output
          };
        } else if (typeof output === 'string') {
          historyReadyOutputs[nodeId] = { type: 'text', data: output };
        } else if (isLightX2VResultRef(output)) {
          // Keep LightX2V ref as-is so createHistoryEntryFromValue produces kind: 'lightx2v_result'
          historyReadyOutputs[nodeId] = output;
        } else if (typeof output === 'object' && output !== null) {
          historyReadyOutputs[nodeId] = { type: 'json', data: output };
        } else {
          historyReadyOutputs[nodeId] = output;
        }
      }

      // Save output files to database (if workflow has a database ID). 纯前端不请求后端，跳过保存避免 ✗ Save returned null 警告
      const hasValidDbId = workflow.id && (workflow.id.startsWith('workflow-') || workflow.id.match(/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i));
      const shouldSaveOutputs = hasValidDbId && !isStandalone();

      // Per-node history: only for nodes in this run set that were actually executed (not upstream nodes whose output was just read as input).
      const nodeHistoryUpdates: Record<string, NodeHistoryEntry[]> = {};
      if (!shouldSaveOutputs) {
        for (const nodeId of executedInSession) {
          if (!nodesToRunIds.has(nodeId)) continue; // Only nodes we chose to run get history
          const node = workflow.nodes.find(n => n.id === nodeId);
          const entry = createHistoryEntryFromValue({
            id: `node-${nodeId}-${runTimestamp}`,
            timestamp: runTimestamp,
            value: historyReadyOutputs[nodeId] ?? sessionOutputs[nodeId],
            executionTime: executionTimeByNodeId[nodeId] ?? node?.executionTime
          });
          if (!entry) continue;
          const prevListRaw = (workflow.nodeOutputHistory && workflow.nodeOutputHistory[nodeId]) || [];
          const prevList = normalizeHistoryEntries(prevListRaw);
          nodeHistoryUpdates[nodeId] = [entry, ...prevList].slice(0, MAX_NODE_HISTORY);
        }
      }
      if (shouldSaveOutputs) {
        try {
          console.log(`[WorkflowExecution] Starting to save outputs for workflow ${workflow.id}, sessionOutputs keys:`, Object.keys(sessionOutputs));
          // Save output files for each node (async, don't block execution)
          const savePromises: Promise<any>[] = [];
          const saveMeta: Array<{ nodeId: string; portId: string; value: any; kind: 'single' | 'multi' | 'array'; index?: number }> = [];

          for (const [nodeId, output] of Object.entries(sessionOutputs)) {
            if (!nodesToRunIds.has(nodeId) || !executedInSession.has(nodeId)) continue; // Only save outputs for nodes we actually ran in this run
            const node = workflow.nodes.find(n => n.id === nodeId);
            if (!node) {
              console.warn(`[WorkflowExecution] Node ${nodeId} not found in workflow.nodes`);
              continue;
            }

            const tool = TOOLS.find(t => t.id === node.toolId);
            if (!tool || !tool.outputs) {
              console.warn(`[WorkflowExecution] Tool ${node.toolId} not found or has no outputs`);
              continue;
            }

            // Use node.outputValue if available (more reliable), otherwise fall back to sessionOutputs
            const outputToSave = node.outputValue !== undefined ? node.outputValue : output;
            const isRef = isLightX2VResultRef(outputToSave);
            console.log(`[WorkflowExecution] Node ${nodeId} (${node.toolId}) outputToSave:`, isRef ? 'LightX2VResultRef' : typeof outputToSave, isRef ? (outputToSave as LightX2VResultRef).task_id : (outputToSave && typeof outputToSave === 'object' ? 'object' : outputToSave ? 'string' : 'empty'));

            // Check if outputToSave is empty or invalid
            if (!outputToSave || (typeof outputToSave === 'string' && outputToSave.length === 0)) {
              console.warn(`[WorkflowExecution] Node ${nodeId} (${node.toolId}) has no output to save, skipping`);
              continue;
            }

            // Handle different output formats - save ALL types of outputs (data URLs, text, JSON, task result refs)
            // LightX2VResultRef must be treated as single object so task_id is persisted; do NOT treat as multi-output
            const scheduleSinglePortSave = (
              portId: string,
              value: any,
              label: string,
              kind: 'single' | 'array' = 'single',
              index?: number,
              previousPromise?: Promise<any>
            ) => {
              const metaEntry: { nodeId: string; portId: string; value: any; kind: 'single' | 'multi' | 'array'; index?: number } = {
                nodeId,
                portId,
                value,
                kind
              };
              if (kind === 'array' && index !== undefined) metaEntry.index = index;
              saveMeta.push(metaEntry);
              const performSave = () =>
                saveNodeOutputs(workflow.id, nodeId, { [portId]: value })
                .then(result => {
                  if (result && result[portId]) {
                    console.log(`[WorkflowExecution] ✓ Saved ${label}:`, result[portId]);
                  } else {
                    console.warn(`[WorkflowExecution] ✗ Save returned null for ${label}`);
                  }
                  return result;
                })
                .catch(err => {
                  console.error(`[WorkflowExecution] ✗ Error saving ${label}:`, err);
                  throw err;
                });
              const promise = previousPromise ? previousPromise.then(() => performSave()) : performSave();
              savePromises.push(promise);
              return promise;
            };

            if (isLightX2VResultRef(outputToSave)) {
              const firstOutputPort = tool.outputs[0];
              if (firstOutputPort) {
                console.log(`[WorkflowExecution] Saving single-output node ${nodeId}/${firstOutputPort.id} (LightX2VResultRef):`, outputToSave.task_id);
                scheduleSinglePortSave(
                  firstOutputPort.id,
                  outputToSave,
                  `node ${nodeId}/${firstOutputPort.id} (LightX2VResultRef)`
                );
              }
            } else if (outputToSave && typeof outputToSave === 'object' && !Array.isArray(outputToSave)) {
              // Multi-output node (e.g. text-generation with customOutputs). Save as single kind:json entry.
              const outputsToSave: Record<string, string | object> = {};
              for (const [portId, value] of Object.entries(outputToSave)) {
                if ((typeof value === 'string' && value.length > 0) || (typeof value === 'object' && value !== null && !Array.isArray(value))) {
                  outputsToSave[portId] = value;
                }
              }
              if (Object.keys(outputsToSave).length > 0) {
                console.log(`[WorkflowExecution] Saving multi-output node ${nodeId} as single JSON (${Object.keys(outputsToSave).length} fields)`);
                saveMeta.push({ nodeId, portId: '__json__', value: outputToSave, kind: 'multi' });
                savePromises.push(saveNodeOutputs(workflow.id, nodeId, outputsToSave));
              }
            } else if (typeof outputToSave === 'string' && outputToSave.length > 0) {
              // Single output node - use first output port (save data URLs, plain text, and task result URLs, including CDN URLs)
              const firstOutputPort = tool.outputs[0];
              if (firstOutputPort) {
                const isUrl = outputToSave.startsWith('http://') || outputToSave.startsWith('https://') || outputToSave.startsWith('./assets/');
                console.log(`[WorkflowExecution] Saving single-output node ${nodeId}/${firstOutputPort.id} (${isUrl ? 'URL' : 'text'}):`, outputToSave.length > 100 ? outputToSave.substring(0, 100) + '...' : outputToSave);
                scheduleSinglePortSave(
                  firstOutputPort.id,
                  outputToSave,
                  `node ${nodeId}/${firstOutputPort.id} (${isUrl ? 'URL' : 'text'})`
                );
              } else {
                console.warn(`[WorkflowExecution] Node ${nodeId} (${node.toolId}) has no output ports defined`);
              }
            } else if (typeof outputToSave === 'object' && outputToSave !== null && !Array.isArray(outputToSave)) {
              // JSON object output
              const firstOutputPort = tool.outputs[0];
              if (firstOutputPort) {
                scheduleSinglePortSave(
                  firstOutputPort.id,
                  outputToSave,
                  `node ${nodeId}/${firstOutputPort.id} (JSON object)`
                );
              }
            } else if (Array.isArray(outputToSave)) {
              // Array output: if each item maps to a distinct port, use saveNodeOutputs; else sequentially save per port
              const firstOutputPort = tool.outputs[0];
              if (firstOutputPort) {
                const canBatch = tool.outputs.length >= outputToSave.length;
                if (canBatch) {
                  const outputsToSave: Record<string, string | object> = {};
                  for (let i = 0; i < outputToSave.length; i++) {
                    const item = outputToSave[i];
                    const portId = tool.outputs.length > 1 ? tool.outputs[i]?.id || firstOutputPort.id : firstOutputPort.id;
                    if ((typeof item === 'string' && item.length > 0) || (typeof item === 'object' && item !== null)) {
                      outputsToSave[portId] = item;
                      saveMeta.push({ nodeId, portId, value: item, kind: 'array', index: i });
                    }
                  }
                  if (Object.keys(outputsToSave).length > 0) {
                    console.log(`[WorkflowExecution] Saving array-output node ${nodeId} (${Object.keys(outputsToSave).length} ports) in one request`);
                    const nodeSavePromise = saveNodeOutputs(workflow.id, nodeId, outputsToSave);
                    for (const _ of Object.keys(outputsToSave)) {
                      savePromises.push(nodeSavePromise);
                    }
                  }
                } else {
                  let arrSeq: Promise<any> = Promise.resolve();
                  for (let i = 0; i < outputToSave.length; i++) {
                    const item = outputToSave[i];
                    const portId = tool.outputs.length > 1 ? tool.outputs[i]?.id || firstOutputPort.id : firstOutputPort.id;
                    if ((typeof item === 'string' && item.length > 0) || (typeof item === 'object' && item !== null)) {
                      arrSeq = scheduleSinglePortSave(
                        portId,
                        item,
                        `node ${nodeId}/${portId} (array item ${i + 1})`,
                        'array',
                        i,
                        arrSeq
                      );
                    }
                  }
                }
              }
            }
          }

          // Wait for all saves to complete and update history outputs with file_id references
          Promise.allSettled(savePromises).then(results => {
            // Check for any failed saves
            const failedSaves: string[] = [];
            results.forEach((result, index) => {
              if (result.status === 'rejected') {
                const errorMsg = result.reason instanceof Error ? result.reason.message : String(result.reason);
                console.error(`[WorkflowExecution] Save promise ${index} failed:`, errorMsg);
                failedSaves.push(`Save ${index + 1}: ${errorMsg}`);
              }
            });

            if (failedSaves.length > 0) {
              setGlobalError({
                message: `Failed to save ${failedSaves.length} node output(s)`,
                details: failedSaves.join('\n')
              });
            }

            // Update history outputs with file_id references (replace data URLs with references)
            const updatedOutputs: Record<string, any> = { ...historyReadyOutputs };
            const outputReplacements: Array<{ nodeId: string; portId: string; value: any; kind: 'single' | 'multi' | 'array'; index?: number }> = [];

            const isDataUrl = (value: any) => typeof value === 'string' && value.startsWith('data:');

            results.forEach((result, idx) => {
              if (result.status !== 'fulfilled' || !result.value) return;
              const meta = saveMeta[idx];
              if (!meta) return;
              const rawResult = result.value;
              // __json__: whole object stored as one entry; use as refOutput for history, no per-port replacement
              if (meta.portId === '__json__') {
                updatedOutputs[meta.nodeId] = meta.value;
                return;
              }
              const perPortResult = typeof rawResult === 'object' && rawResult !== null && !Array.isArray(rawResult)
                ? (rawResult as Record<string, { file_id?: string; file_url?: string; url?: string; ext?: string } | null | undefined>)
                : null;
              let saveResult: { file_id?: string; file_url?: string; url?: string; ext?: string } | null =
                perPortResult?.[meta.portId] ?? null;
              if (!saveResult || !saveResult.file_id) return;
              const fileUrl = saveResult.file_url || saveResult.url;
              // Never replace LightX2VResultRef with URL so task_id stays in node.outputValue for result_url resolution
              const shouldReplaceWithUrl = fileUrl && isDataUrl(meta.value) && !isLightX2VResultRef(meta.value);
              const refPayload = { type: 'file' as const, file_id: saveResult.file_id, ext: saveResult.ext };
              const urlPayload = { type: 'url' as const, data: fileUrl, ext: saveResult.ext };

              if (meta.kind === 'multi') {
                if (!updatedOutputs[meta.nodeId] || typeof updatedOutputs[meta.nodeId] !== 'object') {
                  updatedOutputs[meta.nodeId] = {};
                }
                updatedOutputs[meta.nodeId][meta.portId] = shouldReplaceWithUrl ? urlPayload : refPayload;
              } else if (meta.kind === 'single') {
                if (typeof meta.value === 'string' && meta.value.startsWith('data:')) {
                  updatedOutputs[meta.nodeId] = shouldReplaceWithUrl ? urlPayload : refPayload;
                } else if (typeof meta.value === 'string') {
                  const isTaskResultUrl = meta.value.startsWith('./assets/task/result') ||
                                         meta.value.startsWith('http://') ||
                                         meta.value.startsWith('https://');
                  updatedOutputs[meta.nodeId] = isTaskResultUrl
                    ? { type: 'url', data: meta.value, ext: saveResult.ext }
                    : { type: 'text', data: meta.value };
                } else {
                  updatedOutputs[meta.nodeId] = { type: 'json', data: meta.value };
                }
              } else if (meta.kind === 'array') {
                if (!Array.isArray(updatedOutputs[meta.nodeId])) {
                  updatedOutputs[meta.nodeId] = [];
                }
                const entry = shouldReplaceWithUrl ? urlPayload : refPayload;
                (updatedOutputs[meta.nodeId] as any[])[meta.index ?? 0] = entry;
              }

              if (shouldReplaceWithUrl) {
                outputReplacements.push({
                  nodeId: meta.nodeId,
                  portId: meta.portId,
                  value: fileUrl,
                  kind: meta.kind,
                  index: meta.index
                });
              }
            });

            // Update node outputValues with saved file/url references (no run concept)
            setWorkflow(prev => {
              if (!prev) return null;
              const updatedNodes = prev.nodes.map(node => {
                const updates = outputReplacements.filter(r => r.nodeId === node.id);
                if (updates.length === 0) return node;
                if (node.outputValue && typeof node.outputValue === 'object' && !Array.isArray(node.outputValue)) {
                  const nextOutputValue = { ...node.outputValue };
                  updates.forEach(update => {
                    if (update.kind === 'multi') {
                      nextOutputValue[update.portId] = update.value;
                    }
                  });
                  return { ...node, outputValue: nextOutputValue };
                }
                if (Array.isArray(node.outputValue)) {
                  const nextArray = [...node.outputValue];
                  updates.forEach(update => {
                    if (update.kind === 'array') {
                      nextArray[update.index ?? 0] = update.value;
                    }
                  });
                  return { ...node, outputValue: nextArray };
                }
                const singleUpdate = updates.find(update => update.kind === 'single');
                return singleUpdate ? { ...node, outputValue: singleUpdate.value } : node;
              });
              return { ...prev, nodes: updatedNodes };
            });

            // Add one nodeOutputHistory entry per saved node (ref format); only for nodes that were in this run
            const nodeIdsWithSaves = new Set<string>();
            results.forEach((result, idx) => {
              if (result.status === 'fulfilled' && result.value && saveMeta[idx] && nodesToRunIds.has(saveMeta[idx].nodeId)) {
                nodeIdsWithSaves.add(saveMeta[idx].nodeId);
              }
            });
            if (nodeIdsWithSaves.size > 0) {
              setWorkflow(prev => {
                if (!prev) return prev;
                const nextHistory = { ...(prev.nodeOutputHistory || {}) };
                for (const nodeId of nodeIdsWithSaves) {
                  const refOutput = updatedOutputs[nodeId];
                  if (refOutput == null) continue;
                  const node = workflow.nodes.find(n => n.id === nodeId);
                  const newEntry = createHistoryEntryFromValue({
                    id: `node-${nodeId}-${runTimestamp}`,
                    timestamp: runTimestamp,
                    value: refOutput,
                    executionTime: executionTimeByNodeId[nodeId] ?? node?.executionTime
                  });
                  if (!newEntry) continue;
                  const prevListRaw = nextHistory[nodeId] || [];
                  const prevList = normalizeHistoryEntries(prevListRaw);
                  nextHistory[nodeId] = [newEntry, ...prevList].slice(0, MAX_NODE_HISTORY);
                }
                return { ...prev, nodeOutputHistory: nextHistory };
              });
            }

            if (outputReplacements.length > 0) {
              setWorkflow(prev => {
                if (!prev) return prev;
                return {
                  ...prev,
                  nodes: prev.nodes.map(n => {
                    const update = outputReplacements.find(u => u.nodeId === n.id);
                    if (!update) return n;
                    if (update.kind === 'multi') {
                      const existing = n.outputValue && typeof n.outputValue === 'object' && !Array.isArray(n.outputValue) ? n.outputValue : {};
                      return { ...n, outputValue: { ...existing, [update.portId]: update.value } };
                    }
                    if (update.kind === 'array') {
                      const existing = Array.isArray(n.outputValue) ? [...n.outputValue] : [];
                      existing[update.index ?? 0] = update.value;
                      return { ...n, outputValue: existing };
                    }
                    return { ...n, outputValue: update.value };
                  })
                };
              });
            }
          }).catch(err => {
            const errorMsg = err instanceof Error ? err.message : String(err);
            console.error('[WorkflowExecution] Error processing save results:', errorMsg);
            setGlobalError({
              message: 'Failed to process save results',
              details: errorMsg
            });
            // nodeOutputHistory already updated below; no run to update
          });
        } catch (error) {
          console.error('[WorkflowExecution] Error initiating output file saves:', error);
          setGlobalError({
            message: 'Failed to initiate output file saves',
            details: error instanceof Error ? error.message : String(error)
          });
          // Don't fail the workflow execution if file saving fails
        }
      } else if (!isStandalone() && workflow?.id && !hasValidDbId) {
       console.warn(`[WorkflowExecution] Workflow ID ${workflow.id} is not a valid database ID, skipping save`);
      }

      // Append per-node history (no run); keep legacy history empty for compat
      setWorkflow(prev => prev ? ({
        ...prev,
        nodeOutputHistory: { ...prev.nodeOutputHistory, ...nodeHistoryUpdates }
      }) : null);

      // Save execution state (nodes with status, outputValue) to database or local
      // preset 工作流（仅前端或有后端）都进入此分支，以便分配新 UUID 再保存
      const workflowId = workflow.id;
      const shouldSave = workflowId && (
        workflowId.startsWith('workflow-') ||
        workflowId.match(/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i) ||
        workflowId.startsWith('preset-')
      );

      if (shouldSave) {
        // Capture current state before async operation to avoid closure issues
        // IMPORTANT: Use a function to get the latest workflow state after all setWorkflow calls
        // This ensures we capture the updated node values (including saved file paths for input nodes)
        const getLatestWorkflowState = () => {
          // This will be called after setTimeout, so workflow state should be updated
          // But we need to use a ref or state getter to get the latest value
          // For now, we'll use workflow directly, but note that input node values should be updated
          return workflow;
        };

        // Build nodes state from session execution results
        // IMPORTANT: Use the latest workflow state to ensure input node values are updated
        // Input nodes may have been updated with saved file paths during execution
        // We need to capture the latest state after all setWorkflow calls
        const nodesToSave = workflow.nodes.map(node => {
          const wasExecuted = sessionOutputs[node.id] !== undefined;
          if (wasExecuted) {
            // Node was executed in this session, use the updated state
            // Note: Input nodes may have been updated with saved file paths
            const updatedNode = workflow.nodes.find(n => n.id === node.id);
            return {
              id: node.id,
              toolId: node.toolId,
              x: node.x,
              y: node.y,
              status: updatedNode?.status || node.status,
              data: updatedNode?.data || node.data, // Use updated data (may include saved file paths for input nodes)
              error: updatedNode?.error,
              executionTime: updatedNode?.executionTime,
              outputValue: updatedNode?.outputValue || node.outputValue
            };
          } else {
            // Node was not executed, keep existing state
            return {
              id: node.id,
              toolId: node.toolId,
              x: node.x,
              y: node.y,
              status: node.status,
              data: node.data,
              error: node.error,
              executionTime: node.executionTime,
              outputValue: node.outputValue
            };
          }
        });

        const nextNodeOutputHistory = {
          ...(workflow.nodeOutputHistory || {}),
          ...nodeHistoryUpdates
        };

        // Save execution state via API (using save queue to avoid conflicts)
        // Use setTimeout to defer and ensure state updates are complete
        setTimeout(async () => {
          try {
            const currentUserId = getCurrentUserId();
            const { owned } = await checkWorkflowOwnership(workflowId, currentUserId);

            // Use save queue to avoid conflicts with manual saves
            await workflowSaveQueue.enqueue(workflowId, async () => {
              // 纯前端部署：只持久化到本地，不请求后端
              if (isStandalone() && onSaveExecutionToLocal) {
                // 不拥有（如 preset-*）时分配新 UUID 再保存到本地
                const newId = !owned
                  ? (typeof crypto !== 'undefined' && crypto.randomUUID ? crypto.randomUUID() : `workflow-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`)
                  : workflowId;
                const updatedWorkflow: WorkflowState = {
                  ...workflow,
                  id: newId,
                  nodes: nodesToSave,
                nodeOutputHistory: nextNodeOutputHistory,
                  updatedAt: Date.now()
                };
                await onSaveExecutionToLocal(updatedWorkflow);
                if (newId !== workflowId) {
                  setWorkflow(prev => prev ? { ...prev, id: newId } : null);
                  if (window.history && window.history.replaceState) {
                    window.history.replaceState(null, '', `#workflow/${newId}`);
                  }
                  console.log('[WorkflowExecution] Workflow saved to local with new ID:', newId);
                } else {
                  console.log('[WorkflowExecution] Execution state saved to local (standalone)');
                }
                return;
              }
              if (!owned) {
                // 不拥有（预设/他人/404）：先创建新 UUID 工作流再保存
                try {
                  const newId = typeof crypto !== 'undefined' && crypto.randomUUID
                    ? crypto.randomUUID()
                    : `workflow-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
                  const createData: any = {
                    name: workflow.name,
                    description: workflow.description ?? '',
                    nodes: nodesToSave,
                    connections: workflow.connections,
                    tags: workflow.tags ?? [],
                    node_output_history: nextNodeOutputHistory,
                    workflow_id: newId
                  };

                  const createResponse = await apiRequest('/api/v1/workflow/create', {
                    method: 'POST',
                    body: JSON.stringify(createData)
                  });

                  if (createResponse.ok) {
                    const data = await createResponse.json();
                    const finalWorkflowId = data.workflow_id;
                    console.log('[WorkflowExecution] Created new workflow (not owned):', finalWorkflowId);
                    setWorkflow(prev => prev ? { ...prev, id: finalWorkflowId } : null);
                    if (window.history && window.history.replaceState) {
                      window.history.replaceState(null, '', `#workflow/${finalWorkflowId}`);
                    }
                  } else {
                    const errorText = await createResponse.text();
                    throw new Error(`Failed to create new workflow: ${errorText}`);
                  }
                } catch (error) {
                  console.error('[WorkflowExecution] Failed to create new workflow (not owned):', error);
                  setGlobalError({
                    message: 'Failed to save execution state: workflow not owned but creation failed',
                    details: error instanceof Error ? error.message : String(error)
                  });
                }
              } else {
                // Update existing workflow
                try {
                  const updateResponse = await apiRequest(`/api/v1/workflow/${workflowId}`, {
                    method: 'PUT',
                    body: JSON.stringify({
                      nodes: nodesToSave,
                      connections: workflow.connections, // Save connections
                      node_output_history: nextNodeOutputHistory
                    })
                  });

                  if (!updateResponse.ok) {
                    const errorText = await updateResponse.text();
                    throw new Error(`Failed to update workflow: ${errorText}`);
                  }
                  console.log('[WorkflowExecution] Execution state saved successfully');
                } catch (error) {
                  console.error('[WorkflowExecution] Failed to update workflow:', error);

                  // 检查是否是网络错误，如果是则添加到离线队列
                  const isNetworkError = error instanceof TypeError ||
                                       (error instanceof Error && error.message.includes('fetch')) ||
                                       !navigator.onLine;

                  if (isNetworkError) {
                    // 添加到离线队列以便后续恢复
                    workflowOfflineQueue.addTask(workflowId, {
                      nodes: nodesToSave,
                      connections: workflow.connections,
                      node_output_history: nextNodeOutputHistory
                    });
                    console.log('[WorkflowExecution] Execution state save failed (network error), added to offline queue');
                  }

                  // Show error to user
                  setGlobalError({
                    message: 'Failed to save execution state',
                    details: error instanceof Error ? error.message : String(error)
                  });

                  // 提示用户可以通过手动保存来重试
                  console.warn('[WorkflowExecution] Execution state save failed. User can retry by manually saving the workflow.');
                }
              }
            });
          } catch (err) {
            console.error('[WorkflowExecution] Error saving execution state:', err);

            // 检查是否是网络错误，如果是则添加到离线队列
            const isNetworkError = err instanceof TypeError ||
                                 (err instanceof Error && err.message.includes('fetch')) ||
                                 !navigator.onLine;

            if (isNetworkError) {
              // 添加到离线队列以便后续恢复
              workflowOfflineQueue.addTask(workflowId, {
                nodes: nodesToSave,
                connections: workflow.connections,
                node_output_history: nextNodeOutputHistory
              });
              console.log('[WorkflowExecution] Execution state save failed (network error), added to offline queue');
            }

            // Show error to user
            setGlobalError({
              message: 'Failed to save execution state',
              details: err instanceof Error ? err.message : String(err)
            });

            // 提示用户可以通过手动保存来重试
            console.warn('[WorkflowExecution] Execution state save failed. User can retry by manually saving the workflow.');
          }
        }, 100); // Reduced delay from 300ms to 100ms for faster save
      }
    } catch (e: any) {
      console.error('[Workflow] Execution error:', e);
      const errorMessage = e?.message || e?.toString() || '工作流执行失败';
      setGlobalError({
        message: errorMessage,
        details: e?.stack || (typeof e === 'string' ? e : JSON.stringify(e, null, 2))
      });
    } finally {
      const j = runningJobsByJobIdRef.current.get(jobId);
      if (j) {
        j.taskIdsByNodeId.forEach((taskId, nodeId) => {
          if (runningTaskIdsRef.current.get(nodeId) === taskId) runningTaskIdsRef.current.delete(nodeId);
        });
        runningJobsByJobIdRef.current.delete(jobId);
      }
    }
    // isRunning is cleared by processQueue when queue becomes empty
  }, [
    workflow,
    setWorkflow,
    isPausedRef,
    setIsPaused,
    runningTaskIdsRef,
    runningJobsByJobIdRef,
    getLightX2VConfig,
    getDescendants,
    resolveLightX2VResultRef,
    validateWorkflow,
    setValidationErrors,
    setGlobalError,
    updateNodeData,
    voiceList,
    onSaveExecutionToLocal,
    t
  ]);

  const processQueue = useCallback(() => {
    const queue = executionQueueRef.current;
    while (queue.length > 0 && runningJobCountRef.current < MAX_CONCURRENT_JOBS) {
      const job = queue.shift()!;
      if (job.cancelled) {
        refreshPendingNodeIds();
        continue;
      }
      const jobId = job.id;
      const jobInfo = {
        affectedNodeIds: job.affectedNodeIds,
        taskIdsByNodeId: new Map<string, string>(),
        abortController: new AbortController()
      };
      runningJobsByJobIdRef.current.set(jobId, jobInfo);
      runningJobCountRef.current += 1;
      executeOneRun(jobId, job.startNodeId, job.onlyOne).finally(() => {
        runningJobCountRef.current -= 1;
        refreshPendingNodeIds();
        processQueue();
      });
    }
    if (runningJobCountRef.current === 0 && executionQueueRef.current.length === 0) {
      setWorkflow(prev => prev ? ({ ...prev, isRunning: false }) : null);
    }
    refreshPendingNodeIds();
  }, [executeOneRun, setWorkflow, refreshPendingNodeIds]);

  /** Enqueue a run (full workflow or single/from-node); up to 3 jobs run at once. */
  const runWorkflow = useCallback((startNodeId?: string, onlyOne?: boolean) => {
    if (!workflow) return;
    const affectedNodeIds = getAffectedNodeIds(startNodeId, onlyOne);
    const job: QueueJob = {
      id: `job-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`,
      startNodeId,
      onlyOne,
      cancelled: false,
      affectedNodeIds
    };
    executionQueueRef.current.push(job);
    setWorkflow(prev => prev ? ({ ...prev, isRunning: true }) : null);
    refreshPendingNodeIds();
    processQueue();
  }, [workflow, setWorkflow, processQueue, getAffectedNodeIds, refreshPendingNodeIds]);

  /** Cancel the run for a given node: if queued, mark job cancelled; if running, send cancel to backend and abort. */
  const cancelNodeRun = useCallback((nodeId: string) => {
    const queue = executionQueueRef.current;
    const queuedIndex = queue.findIndex(j => !j.cancelled && j.affectedNodeIds.has(nodeId));
    if (queuedIndex !== -1) {
      queue[queuedIndex].cancelled = true;
      refreshPendingNodeIds();
      return;
    }
    for (const [jid, jobInfo] of runningJobsByJobIdRef.current) {
      if (jobInfo.affectedNodeIds.has(nodeId)) {
        jobInfo.taskIdsByNodeId.forEach((taskId) => {
          (async () => {
            try {
              if (isStandalone()) {
                const cloudUrl = (process.env.LIGHTX2V_CLOUD_URL || 'https://x2v.light-ai.top').trim();
                const cloudToken = (process.env.LIGHTX2V_CLOUD_TOKEN || '').trim();
                await lightX2VCancelTask(cloudUrl, cloudToken, taskId);
              } else {
                await apiRequest(`/api/v1/task/cancel?task_id=${taskId}`, { method: 'GET' });
              }
            } catch (e) {
              console.warn('[WorkflowExecution] Cancel task failed:', taskId, e);
            }
          })();
        });
        jobInfo.abortController.abort();
        const affectedIds = new Set(jobInfo.affectedNodeIds);
        setWorkflow(prev => prev ? {
          ...prev,
          nodes: prev.nodes.map(n => affectedIds.has(n.id) && (n.status === NodeStatus.RUNNING || n.status === NodeStatus.PENDING)
            ? { ...n, status: NodeStatus.IDLE, error: 'Cancelled' }
            : n)
        } : null);
        refreshPendingNodeIds();
        break;
      }
    }
  }, [refreshPendingNodeIds, setWorkflow]);

  // Stop workflow execution by cancelling all running tasks
  const stopWorkflow = useCallback(async () => {
    if (!workflow) {
      console.warn('[WorkflowExecution] Cannot stop: workflow is null');
      return;
    }

    executionQueueRef.current = [];
    runningJobsByJobIdRef.current.forEach((jobInfo) => jobInfo.abortController.abort());
    runningJobsByJobIdRef.current.clear();
    setPendingRunNodeIds([]);
    console.log('[WorkflowExecution] Stopping workflow...');

    // Cancel all running tasks via API
    const taskIds = Array.from(runningTaskIdsRef.current.values());
    console.log(`[WorkflowExecution] Found ${taskIds.length} running tasks to cancel:`, taskIds);

    if (taskIds.length === 0) {
      console.warn('[WorkflowExecution] No running tasks found in runningTaskIdsRef');
      // Still try to abort ongoing requests and stop workflow
    }

    const cancelPromises = taskIds.map(async (taskId) => {
      try {
        console.log(`[WorkflowExecution] Sending cancel request for task ${taskId}...`);
        let response: Response;
        if (isStandalone()) {
          const cloudUrl = (process.env.LIGHTX2V_CLOUD_URL || 'https://x2v.light-ai.top').trim();
          const cloudToken = (process.env.LIGHTX2V_CLOUD_TOKEN || '').trim();
          response = await lightX2VCancelTask(cloudUrl, cloudToken, String(taskId));
        } else {
          response = await apiRequest(`/api/v1/task/cancel?task_id=${taskId}`, {
            method: 'GET'
          });
        }
        if (response.ok) {
          const data = await response.json().catch(() => ({}));
          console.log(`[WorkflowExecution] Task ${taskId} cancelled successfully:`, data);
        } else {
          const errorText = await response.text();
          console.warn(`[WorkflowExecution] Failed to cancel task ${taskId}:`, response.status, errorText);
        }
      } catch (error) {
        console.error(`[WorkflowExecution] Error cancelling task ${taskId}:`, error);
      }
    });

    // Wait for all cancellation requests to complete
    await Promise.all(cancelPromises);

    // Query current task status for each running task (taskId -> nodeId map before clear)
    const taskIdToNodeId = new Map<string, string>();
    runningTaskIdsRef.current.forEach((tid, nid) => taskIdToNodeId.set(tid, nid));

    const cancelledNodeIds = new Set<string>();
    const queryPromises = taskIds.map(async (taskId) => {
      try {
        let status: string;
        if (isStandalone()) {
          const cloudUrl = (process.env.LIGHTX2V_CLOUD_URL || 'https://x2v.light-ai.top').trim();
          const cloudToken = (process.env.LIGHTX2V_CLOUD_TOKEN || '').trim();
          const info = await lightX2VTaskQuery(cloudUrl, cloudToken, String(taskId));
          status = info.status;
        } else {
          const res = await apiRequest(`/api/v1/task/query?task_id=${taskId}`, { method: 'GET' });
          const data = res.ok ? (await res.json().catch(() => ({})) as { status?: string }) : {};
          status = data.status || 'UNKNOWN';
        }
        if (status === 'CANCELLED') {
          const nodeId = taskIdToNodeId.get(String(taskId));
          if (nodeId) cancelledNodeIds.add(nodeId);
        }
      } catch (_) {
        // ignore query errors
      }
    });
    await Promise.all(queryPromises);

    // Clear running task IDs
    runningTaskIdsRef.current.clear();

    // Abort any ongoing fetch requests
    if (abortControllerRef.current) {
      console.log('[WorkflowExecution] Aborting ongoing fetch requests...');
      abortControllerRef.current.abort();
      abortControllerRef.current = new AbortController();
    }

    // Stop workflow execution and update node statuses (cancelled nodes get error message)
    setWorkflow(prev => {
      if (!prev) return null;
      return {
        ...prev,
        isRunning: false,
        nodes: prev.nodes.map(n => {
          if (n.status !== NodeStatus.RUNNING && n.status !== NodeStatus.PENDING) return n;
          const cancelled = cancelledNodeIds.has(n.id);
          return {
            ...n,
            status: NodeStatus.IDLE,
            error: cancelled ? 'Cancelled' : undefined
          };
        })
      };
    });

    console.log('[WorkflowExecution] Workflow stopped');
  }, [workflow, runningTaskIdsRef, abortControllerRef, setWorkflow]);

  return {
    runWorkflow,
    stopWorkflow,
    cancelNodeRun,
    pendingRunNodeIds,
    resolveLightX2VResultRef,
    validateWorkflow,
    getDescendants
  };
}

export const useWorkflowExecution = useWorkflowExecutionImpl;
