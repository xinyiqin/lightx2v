import React, { useCallback, useRef } from 'react';
import { flushSync } from 'react-dom';
import { WorkflowState, Connection, NodeStatus, NodeHistoryEntry } from '../../types';

const MAX_NODE_HISTORY = 20;
import { TOOLS } from '../../constants';
import {
  geminiText, geminiImage, geminiSpeech, geminiVideo,
  lightX2VTask, lightX2VTTS, lightX2VVoiceCloneTTS,
  deepseekText, doubaoText, ppchatGeminiText,
  getLightX2VConfigForModel,
  lightX2VCancelTask,
  lightX2VTaskQuery,
  type WorkflowRefsPayload
} from '../../services/geminiService';
import { isStandalone } from '../config/runtimeMode';
import { removeGeminiWatermark } from '../../services/watermarkRemover';
import { useTranslation, Language } from '../i18n/useTranslation';
import { saveNodeOutputs, saveInputFileViaOutputSave, getNodeOutputData, getWorkflowFileByFileId, getWorkflowFileText, type SaveNodeOutputResult } from '../utils/workflowFileManager';
import { apiRequest } from '../utils/apiClient';
import { resolveLightX2VResultRef as resolveLightX2VResultRefUtil, isLightX2VResultRef as isLightX2VResultRefUtil, toLightX2VResultRef, type LightX2VResultRef } from '../utils/resultRef';
import { getOutputValueByPort, setOutputValueByPort, normalizeOutputValueToPortKeyed, INPUT_PORT_IDS, isPortKeyedOutputValue } from '../utils/outputValuePort';
import { workflowSaveQueue } from '../utils/workflowSaveQueue';
import { workflowOfflineQueue } from '../utils/workflowOfflineQueue';
import { checkWorkflowOwnership, getCurrentUserId } from '../utils/workflowUtils';
import { getAssetPath, getAssetBasePath } from '../utils/assetPath';
import { createHistoryEntryFromValue, createHistoryEntryFromPortKeyedOutputValue, normalizeHistoryEntries } from '../utils/historyEntry';

export type { LightX2VResultRef };
export const isLightX2VResultRef = isLightX2VResultRefUtil;

const EXT_TO_MIME: Record<string, string> = {
  '.txt': 'text/plain', '.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
  '.gif': 'image/gif', '.webp': 'image/webp', '.mp4': 'video/mp4', '.webm': 'video/webm',
  '.mp3': 'audio/mpeg', '.wav': 'audio/wav', '.ogg': 'audio/ogg', '.json': 'application/json',
};
function extToMimeType(ext: string): string {
  const norm = ext?.startsWith('.') ? ext : (ext ? `.${ext}` : '');
  return EXT_TO_MIME[norm] ?? 'application/octet-stream';
}

const isBase64 = (v: any): boolean =>
  typeof v === 'string' && v.startsWith('data:');

/** 从节点 output_value 中移除 base64，返回清理后的值（不修改原对象） */
function stripBase64FromOutputValue(outputValue: any): any {
  if (outputValue == null) return outputValue;
  if (isBase64(outputValue)) return undefined;
  if (Array.isArray(outputValue)) {
    const arr = outputValue.map(stripBase64FromOutputValue).filter(x => x !== undefined);
    return arr.length > 0 ? arr : undefined;
  }
  if (typeof outputValue === 'object') {
    // file ref 持久化 file_id、mime_type、ext、run_id（与后端一致，加载时用 ext/run_id 拼路径）
    if ((outputValue as any).kind === 'file' && (outputValue as any).file_id) {
      const v = outputValue as { file_id: string; mime_type?: string; ext?: string; run_id?: string };
      return {
        kind: 'file' as const,
        file_id: v.file_id,
        ...(v.mime_type != null && { mime_type: v.mime_type }),
        ...(v.ext != null && v.ext !== '' && { ext: v.ext.startsWith('.') ? v.ext : `.${v.ext}` }),
        ...(v.run_id != null && v.run_id !== '' && { run_id: v.run_id }),
      };
    }
    const out: Record<string, any> = {};
    for (const [k, v] of Object.entries(outputValue)) {
      const cleaned = stripBase64FromOutputValue(v);
      if (cleaned !== undefined && cleaned !== '') out[k] = cleaned;
    }
    return Object.keys(out).length > 0 ? out : undefined;
  }
  return outputValue;
}

/** 从 history entry value 中移除 base64（_full_data、dataUrl、data 为 data: 等），避免持久化到 DB */
function stripBase64FromHistoryValue(value: any): any {
  if (value == null) return value;
  if (typeof value === 'string' && value.startsWith('data:')) return undefined;
  if (typeof value !== 'object') return value;
  const v = { ...value };
  if (v._full_data && typeof v._full_data === 'string' && v._full_data.startsWith('data:')) delete v._full_data;
  if (v.dataUrl && typeof v.dataUrl === 'string' && v.dataUrl.startsWith('data:')) delete v.dataUrl;
  if (v.data && typeof v.data === 'string' && v.data.startsWith('data:')) delete v.data;
  return v;
}

const defaultCropBox = () => ({ x: 10, y: 10, w: 80, h: 80 });

/** 将单端口节点的 workflow file URL 转为 port-keyed { [portId]: { kind, file_id, mime_type, ext?, run_id? } }，便于持久化一致 */
function normalizeNodeOutputValueForPersist(outputValue: any, toolId: string): any {
  if (typeof outputValue !== 'string') return outputValue;
  const isOldFileUrl = outputValue.includes('/api/v1/workflow/') && outputValue.includes('/file/');
  const isNewFileUrl = outputValue.includes('/assets/workflow/file');
  if (!isOldFileUrl && !isNewFileUrl) return outputValue;
  const fileIdMatch = isNewFileUrl ? outputValue.match(/[?&]file_id=([^&]+)/) : outputValue.match(/\/file\/([^/?]+)/);
  if (!fileIdMatch) return outputValue;
  const tool = TOOLS.find(t => t.id === toolId);
  const portId = tool?.outputs?.[0]?.id;
  if (!portId) return outputValue;
  let mime_type = 'application/octet-stream';
  const mimeMatch = outputValue.match(/mime_type=([^&]+)/);
  if (mimeMatch) mime_type = decodeURIComponent(mimeMatch[1].replace(/\+/g, ' '));
  const extMatch = outputValue.match(/[?&]ext=([^&]+)/);
  const ext = extMatch ? decodeURIComponent(extMatch[1].replace(/\+/g, ' ')) : undefined;
  const runIdMatch = outputValue.match(/[?&]run_id=([^&]+)/);
  const run_id = runIdMatch ? decodeURIComponent(runIdMatch[1].replace(/\+/g, ' ')) : undefined;
  return {
    [portId]: {
      kind: 'file' as const,
      file_id: fileIdMatch[1],
      mime_type,
      ...(ext != null && ext !== '' && { ext: ext.startsWith('.') ? ext : `.${ext}` }),
      ...(run_id != null && run_id !== '' && { run_id }),
    },
  };
}

/** 从 output_value 中只保留 file/task ref（用于 Input 节点在 strip 后为 undefined 时保留已有 ref） */
function extractRefsOnlyFromOutputValue(val: any): any {
  if (val == null) return val;
  if (typeof val === 'object' && (val as any).kind === 'file' && (val as any).file_id) return val;
  if (typeof val === 'object' && ((val as any).kind === 'task' || (val as any).task_id)) return val;
  if (Array.isArray(val)) {
    const arr = val.map(extractRefsOnlyFromOutputValue).filter(x => x != null);
    return arr.length > 0 ? arr : undefined;
  }
  if (typeof val === 'object') {
    const out: Record<string, any> = {};
    for (const [k, v] of Object.entries(val)) {
      const cleaned = extractRefsOnlyFromOutputValue(v);
      if (cleaned !== undefined && cleaned !== '') out[k] = cleaned;
    }
    return Object.keys(out).length > 0 ? out : undefined;
  }
  return undefined;
}

/** 返回可安全持久化的 nodes 和 node_output_history（不含 base64）；image_edits 只保留 crop_box */
function stripBase64FromWorkflowPayload(nodes: any[], nodeOutputHistory: Record<string, any[]> | undefined): { nodes: any[]; node_output_history: Record<string, any[]> } {
  const cleanedNodes = nodes.map(n => {
    const rawOut = normalizeNodeOutputValueForPersist(n.output_value, n.tool_id);
    let outVal = stripBase64FromOutputValue(rawOut);
    const tool = TOOLS.find(t => t.id === n.tool_id);
    if (tool?.category === 'Input' && outVal === undefined && rawOut != null) {
      const refsOnly = extractRefsOnlyFromOutputValue(rawOut);
      if (refsOnly !== undefined) outVal = refsOnly;
    }
    const dataVal = n.data?.value != null ? stripBase64FromOutputValue(n.data.value) : n.data?.value;
    let data = n.data && (dataVal !== n.data.value || outVal !== n.output_value)
      ? (() => { const { value: _v, ...rest } = n.data || {}; return { ...rest, ...(dataVal !== undefined ? { value: dataVal } : {}) }; })()
      : n.data;
    if (Array.isArray(data?.image_edits)) {
      data = { ...data, image_edits: data.image_edits.map((e: any) => ({ crop_box: e?.crop_box ?? defaultCropBox() })) };
    }
    if (outVal === n.output_value && data === n.data) return n;
    return { ...n, output_value: outVal, data };
  });
  const cleanedHistory: Record<string, any[]> = {};
  if (nodeOutputHistory && typeof nodeOutputHistory === 'object') {
    for (const [nodeId, entries] of Object.entries(nodeOutputHistory)) {
      if (!Array.isArray(entries)) continue;
      cleanedHistory[nodeId] = entries.map(entry => {
        if (!entry || typeof entry !== 'object') return entry;
        let changed = false;
        const next = { ...entry };
        if (entry.value != null) {
          const v = stripBase64FromHistoryValue(entry.value);
          if (v !== entry.value) { next.value = v; changed = true; }
        }
        if ('output_value_port_keyed' in next) {
          delete (next as any).output_value_port_keyed;
          changed = true;
        }
        if (entry.output_value && typeof entry.output_value === 'object') {
          const cleanedOv = stripBase64FromOutputValue(entry.output_value);
          if (cleanedOv !== entry.output_value) {
            next.output_value = cleanedOv;
            changed = true;
          }
        }
        return changed ? next : entry;
      });
    }
  }
  return { nodes: cleanedNodes, node_output_history: cleanedHistory };
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

/** 云端提交前将媒体入参（file ref 或字符串）统一解析为 data URL；无效/空时返回 null 避免提交空 base64 */
async function resolveMediaForCloud(val: any, workflowId: string): Promise<string | null> {
  if (val == null) return null;
  if (typeof val === 'object' && (val as { file_id?: string }).file_id) {
    const ref = val as { file_id: string; mime_type?: string; ext?: string; run_id?: string };
    const dataUrl = await getWorkflowFileByFileId(workflowId, ref.file_id, ref.mime_type, ref.ext, undefined, undefined, ref.run_id);
    if (!dataUrl || typeof dataUrl !== 'string') {
      console.warn('[WorkflowExecution] resolveMediaForCloud: getWorkflowFileByFileId returned no data for file_id=', ref.file_id);
      return null;
    }
    if (dataUrl.startsWith('data:') && (!dataUrl.includes(',') || !dataUrl.split(',')[1]?.trim())) {
      console.warn('[WorkflowExecution] resolveMediaForCloud: data URL has no base64 payload for file_id=', ref.file_id);
      return null;
    }
    const out = await ensureLocalInputAsDataUrl(dataUrl);
    if (typeof out !== 'string' || !out.trim()) return null;
    if (out.startsWith('data:') && (!out.includes(',') || !out.split(',')[1]?.trim())) return null;
    return out;
  }
  if (typeof val === 'string') {
    const out = await ensureLocalInputAsDataUrl(val);
    if (typeof out !== 'string' || !out.trim()) return null;
    if (out.startsWith('data:') && (!out.includes(',') || !out.split(',')[1]?.trim())) return null;
    return out;
  }
  return null;
}

/** 将本地 URL/路径转为 data URL（同源或 /assets），供云端提交使用 */
async function ensureLocalInputAsDataUrl(input: any): Promise<any> {
  if (typeof input !== 'string') return input;
  if (input.startsWith('data:')) return input;
  if (input.startsWith('//')) return input;

  const isHttp = input.startsWith('http');
  const isLocalAsset = input.includes('/assets/task/result') || input.includes('/assets/workflow/file') || input.startsWith('/assets/');
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

/** 从 connections 组装 workflow_output 元组，供本地 x2v 提交使用；端口 -> submit 参数名见文档 */
function buildWorkflowRefs(
  workflowId: string,
  incomingConns: Connection[],
  toolInputs: { id: string }[]
): WorkflowRefsPayload | undefined {
  const refs: WorkflowRefsPayload = { workflowId };
  const tuple = (c: Connection): [string, string, string] => [workflowId, c.source_node_id, c.source_port_id];
  for (const port of toolInputs) {
    const conns = incomingConns.filter(c => c.target_port_id === port.id);
    if (conns.length === 0) continue;
    if (port.id === 'in-text') {
      refs.prompt = tuple(conns[0]);
    } else if (port.id === 'in-image') {
      refs.input_image = conns.length === 1 ? tuple(conns[0]) : conns.map(tuple);
    } else if (port.id === 'in-image-start') {
      refs.input_image = tuple(conns[0]);
    } else if (port.id === 'in-image-end') {
      refs.input_last_frame = tuple(conns[0]);
    } else if (port.id === 'in-audio') {
      refs.input_audio = tuple(conns[0]);
    } else if (port.id === 'in-video') {
      refs.input_video = tuple(conns[0]);
    }
  }
  const hasRefs = refs.prompt !== undefined || refs.input_image !== undefined || refs.input_audio !== undefined || refs.input_video !== undefined || refs.input_last_frame !== undefined;
  return hasRefs ? refs : undefined;
}

/** 将图片 URL/路径转为 data URL（本地资源优先），供需要 base64 的 API 使用 */
async function ensureImageInputsAsDataUrls(imgs: string[]): Promise<string[]> {
  return Promise.all(imgs.map(async (img) => {
    if (typeof img !== 'string') return img;
    if (img.startsWith('data:')) return img;
    if (img.startsWith('//')) return img;
    const isHttp = img.startsWith('http');
    const isLocalAsset = img.includes('/assets/task/result') || img.includes('/assets/workflow/file');
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
  /** 执行前若工作流 isDirty 或为预设，先调用此方法保存到后端，再开始执行。返回保存后的 workflow_id（若新建则与传入 id 可能不同）。 */
  saveWorkflowBeforeRun?: (workflow: WorkflowState) => Promise<string | void>;
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
  onSaveExecutionToLocal,
  saveWorkflowBeforeRun
}: UseWorkflowExecutionProps) {
  const { t } = useTranslation(lang);

  const getDescendants = useCallback((nodeId: string, connections: Connection[]): Set<string> => {
    const descendants = new Set<string>();
    const stack = [nodeId];
    while (stack.length > 0) {
      const current = stack.pop()!;
      connections.filter(c => c.source_node_id === current).forEach(c => {
        if (!descendants.has(c.target_node_id)) {
          descendants.add(c.target_node_id);
          stack.push(c.target_node_id);
        }
      });
    }
    return descendants;
  }, []);

  /** 按拓扑序将待运行节点分成多批：每批内节点无依赖关系，可并行执行。仅考虑 nodesToRunIds 内的边。 */
  const getTopologicalBatches = useCallback((nodesToRunIds: Set<string>, connections: Connection[]): string[][] => {
    const nodeIds = Array.from(nodesToRunIds);
    const inDegree: Record<string, number> = {};
    nodeIds.forEach(id => { inDegree[id] = 0; });
    connections.forEach(c => {
      if (nodesToRunIds.has(c.source_node_id) && nodesToRunIds.has(c.target_node_id) && c.source_node_id !== c.target_node_id) {
        inDegree[c.target_node_id] = (inDegree[c.target_node_id] ?? 0) + 1;
      }
    });
    const batches: string[][] = [];
    const added = new Set<string>();
    while (added.size < nodeIds.length) {
      const layer = nodeIds.filter(id => !added.has(id) && inDegree[id] === 0);
      if (layer.length === 0) break;
      batches.push(layer);
      layer.forEach(id => added.add(id));
      layer.forEach(sourceId => {
        connections.filter(c => c.source_node_id === sourceId).forEach(c => {
          const t = c.target_node_id;
          if (nodesToRunIds.has(t) && inDegree[t] != null) inDegree[t]--;
        });
      });
    }
    if (added.size < nodeIds.length) {
      const remaining = nodeIds.filter(id => !added.has(id));
      batches.push(remaining);
    }
    return batches;
  }, []);

  const validateWorkflow = useCallback((nodesToRunIds: Set<string>): { message: string; type: 'ENV' | 'INPUT' }[] => {
    if (!workflow) return [];
    const errors: { message: string; type: 'ENV' | 'INPUT' }[] = [];

    const usesLightX2V = Array.from(nodesToRunIds).some(id => {
      const node = workflow.nodes.find(n => n.id === id);
      return node && (node.tool_id.includes('lightx2v') || node.tool_id.includes('video') || node.tool_id === 'avatar-gen' || ((node.tool_id === 'text-to-image' || node.tool_id === 'image-to-image') && node.data.model?.startsWith('Qwen')));
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
      const tool = TOOLS.find(t => t.id === node.tool_id);
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

        const isConnected = workflow.connections.some(c => c.target_node_id === node.id && c.target_port_id === port.id);
        const hasGlobalVal = !!workflow.globalInputs[`${node.id}-${port.id}`]?.toString().trim();

        if (!isConnected && !hasGlobalVal) {
          errors.push({
            message: `${node.name ?? (lang === 'zh' ? tool.name_zh : tool.name)} -> ${port.label}`,
            type: 'INPUT'
          });
        }
      });

      // Special validation for voice clone nodes
      if (node.tool_id === 'lightx2v-voice-clone') {
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
  /** 用于 run 完成时合并到“当前最新”workflow，避免后完成的 run 用旧快照覆盖后加入的节点或其它 run 的结果 */
  const workflowRef = React.useRef<WorkflowState | null>(workflow);
  React.useEffect(() => {
    workflowRef.current = workflow;
  }, [workflow]);
  /** 预设运行前会先实体化到 DB，得到新 id；执行阶段用此 id 发 save/chat 等请求，避免用 preset-xxx 导致 404 */
  const effectiveWorkflowIdRef = React.useRef<string | null>(null);

  const getAffectedNodeIds = useCallback((startNodeId?: string, onlyOne?: boolean): Set<string> => {
    if (!workflow) return new Set();
    if (startNodeId) {
      if (onlyOne) return new Set([startNodeId]);
      const desc = getDescendants(startNodeId, workflow.connections);
      desc.add(startNodeId);
      return desc;
    }
    // 运行整个画布时与 executeOneRun 一致：只影响非输入节点
    return new Set(
      workflow.nodes
        .filter(n => TOOLS.find(t => t.id === n.tool_id)?.category !== 'Input')
        .map(n => n.id)
    );
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
    /** 预设运行前已实体化到 DB，effectiveWorkflowIdRef 会带上新 id；整次执行用 wfId 发 save/chat 等请求 */
    const wfId = effectiveWorkflowIdRef.current ?? workflow.id ?? '';
    effectiveWorkflowIdRef.current = null;
    const jobAbortSignal = jobInfo.abortController.signal;
    const registerTaskId = (nodeId: string, taskId: string, isCloud?: boolean) => {
      runningTaskIdsRef.current.set(nodeId, taskId);
      jobInfo.taskIdsByNodeId.set(nodeId, taskId);
      jobInfo.nodeIdToIsCloud.set(nodeId, isCloud ?? false);
      // 与主应用 TaskDetails 一致：轮询 task/query 更新节点 run_state；is_cloud 时用云端接口
      if (!(jobInfo as any).pollIntervalId) {
        (jobInfo as any).pollIntervalId = window.setInterval(async () => {
          const j = runningJobsByJobIdRef.current.get(jobId);
          if (!j) return;
          for (const [nid, tid] of j.taskIdsByNodeId) {
            try {
              let run_state: { status: string; subtasks?: any[] };
              if (isStandalone() || (jobInfo.nodeIdToIsCloud.get(nid) === true)) {
                const cloudUrl = (process.env.LIGHTX2V_CLOUD_URL || 'https://x2v.light-ai.top').trim();
                const cloudToken = (process.env.LIGHTX2V_CLOUD_TOKEN || '').trim();
                const info = await lightX2VTaskQuery(cloudUrl, cloudToken, String(tid));
                run_state = { status: info.status || 'UNKNOWN', subtasks: [] };
              } else {
                const res = await apiRequest(`/api/v1/task/query?task_id=${tid}`, { method: 'GET' });
                const data = res.ok ? (await res.json().catch(() => ({})) as { status?: string; subtasks?: any[] }) : {};
                run_state = { status: data.status || 'UNKNOWN', subtasks: data.subtasks || [] };
              }
              setWorkflow(prev => prev ? ({ ...prev, nodes: prev.nodes.map(n => n.id === nid ? { ...n, run_state } : n) }) : null);
              if (['SUCCEED', 'FAILED', 'CANCEL', 'CANCELLED'].includes(run_state.status)) {
                j.taskIdsByNodeId.delete(nid);
                j.nodeIdToIsCloud.delete(nid);
              }
            } catch (_) { /* ignore */ }
          }
          if (j.taskIdsByNodeId.size === 0 && (j as any).pollIntervalId) {
            clearInterval((j as any).pollIntervalId);
            (j as any).pollIntervalId = null;
          }
        }, 2000);
      }
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
      // 运行整个画布时跳过输入节点，相当于从「除输入节点外的第一层」开始
      nodesToRunIds = new Set(
        workflow.nodes
          .filter(n => TOOLS.find(t => t.id === n.tool_id)?.category !== 'Input')
          .map(n => n.id)
      );
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
        n.tool_id.includes('video') ||
        n.tool_id === 'avatar-gen' ||
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
      nodes: prev.nodes.map(n => nodesToRunIds.has(n.id) ? { ...n, status: NodeStatus.PENDING, error: undefined, execution_time: undefined, start_time: undefined, run_state: undefined } : n)
    }) : null);

    const executedInSession = new Set<string>();
    const sessionOutputs: Record<string, any> = {};

    // 本次运行所需的上游节点（含图片输入等，用于「执行时存库」）
    const nodesNeededAsInputs = new Set<string>();
    workflow.connections.forEach(conn => {
      if (nodesToRunIds.has(conn.target_node_id) && !nodesToRunIds.has(conn.source_node_id)) {
        nodesNeededAsInputs.add(conn.source_node_id);
      }
    });

    // 未参与本次运行的节点（输入节点或上游）需预填 sessionOutputs，供本 run 内节点读入；与「从某节点运行」一致，统一走「先画布再后端拉取并解析 file/task」
    workflow.nodes.forEach(n => {
      if (!nodesToRunIds.has(n.id) && n.output_value != null) sessionOutputs[n.id] = n.output_value;
    });
    for (const nodeId of nodesNeededAsInputs) {
      if (sessionOutputs[nodeId] !== undefined) continue;
      const n = workflow.nodes.find(nn => nn.id === nodeId);
      if (n?.output_value != null) sessionOutputs[nodeId] = n.output_value;
    }

    // 对上游节点从后端拉取端口输出（file/task 由后端 load_workflow_output 等解析后返回），与运行单节点/从某节点运行一致
    if (wfId && (wfId.startsWith('workflow-') || wfId.match(/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i))) {
      const loadPromises: Promise<void>[] = [];
      for (const nodeId of nodesNeededAsInputs) {
        const node = workflow.nodes.find(n => n.id === nodeId);
        if (!node) continue;
        const tool = TOOLS.find(t => t.id === node.tool_id);
        if (!tool?.outputs?.length) continue;
        for (const port of tool.outputs) {
          loadPromises.push(
            getNodeOutputData(wfId, nodeId, port.id)
              .then(resolved => {
                if (resolved != null) {
                  if (tool.outputs!.length === 1) {
                    sessionOutputs[nodeId] = resolved;
                  } else {
                    if (!sessionOutputs[nodeId] || typeof sessionOutputs[nodeId] !== 'object') {
                      sessionOutputs[nodeId] = {};
                    }
                    (sessionOutputs[nodeId] as Record<string, any>)[port.id] = resolved;
                  }
                }
              })
              .catch(err => {
                console.warn(`[WorkflowExecution] Failed to load node output data for ${nodeId}/${port.id}:`, err);
              })
          );
        }
      }
      if (loadPromises.length > 0) {
        await Promise.race([
          Promise.all(loadPromises),
          new Promise(resolve => setTimeout(resolve, 3000))
        ]);
      }
    }

    // 输入节点：仅当「本次值」与「最近一条历史」不同时才记一次历史（修改后执行才产生历史，不是每次执行都产生）
    const inputNodesWithNewValue = new Set<string>();
    const lastHistoryValue = (nodeId: string, portId: string): any => {
      const list = workflow.nodeOutputHistory?.[nodeId];
      if (!Array.isArray(list) || list.length === 0) return undefined;
      const entry = list[0];
      if (!entry) return undefined;
      const ov = (entry as any).output_value ?? (entry as any).value;
      if (ov == null) return undefined;
      if (typeof ov === 'object' && portId in ov) return ov[portId];
      return ov;
    };
    const sameInputValue = (a: any, b: any): boolean => {
      if (a === b) return true;
      if (a == null || b == null) return false;
      if (typeof a === 'string' && typeof b === 'string') return a === b;
      if (typeof a === 'object' && typeof b === 'object') {
        const idA = (a as any).file_id ?? (a as any).file_url;
        const idB = (b as any).file_id ?? (b as any).file_url;
        if (idA != null && idB != null) return idA === idB;
        if (Array.isArray(a) && Array.isArray(b) && a.length === b.length) {
          return a.every((v, i) => sameInputValue(v, b[i]));
        }
      }
      return false;
    };
    for (const node of workflow.nodes) {
      const tool = TOOLS.find(t => t.id === node.tool_id);
      if (!tool || tool.category !== 'Input') continue;
      if (!nodesToRunIds.has(node.id) && !nodesNeededAsInputs.has(node.id)) continue;
      const portId = INPUT_PORT_IDS[node.tool_id];
      if (!portId) continue;
      // 当前值：output_value 优先；text-input 载入后为 file ref，与历史条目的 output_value 一致则视为未变化，不重复保存
      const current = getOutputValueByPort(node, portId) ?? node.data?.value;
      const last = lastHistoryValue(node.id, portId);
      if (!sameInputValue(current, last)) inputNodesWithNewValue.add(node.id);
    }

    // 输入节点：output_value 为 port-keyed；若为 data URL 则先 save 再写入 sessionOutputs 为 ref
    const isRef = (v: any) => v && typeof v === 'object' && ((v as any).kind === 'file' || (v as any).type === 'file' || (v as any).file_id);
    if (wfId && (isStandalone() || wfId.startsWith('workflow-') || wfId.match(/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i))) {
      const inputNodesToProcess = (node: typeof workflow.nodes[0]) => {
        const tool = TOOLS.find(t => t.id === node.tool_id);
        if (!tool || tool.category !== 'Input') return false;
        return nodesToRunIds.has(node.id) || nodesNeededAsInputs.has(node.id);
      };
      for (const node of workflow.nodes) {
        if (!inputNodesToProcess(node)) continue;

        const portId = INPUT_PORT_IDS[node.tool_id];
        if (!portId) continue;

        // 文本输入节点：有 file ref 时展示文件内容且可编辑；若编辑后与文件内容不一致则重新保存为新文件。纯前端模式始终为文本不落库。
        if (node.tool_id === 'text-input') {
          const rawPort = getOutputValueByPort(node, 'out-text');
          const alreadyFileRef = rawPort && typeof rawPort === 'object' && (rawPort as any).file_id && ((rawPort as any).mime_type === 'text/plain' || (rawPort as any).ext === '.txt' || (rawPort as any).ext === 'txt');
          if (alreadyFileRef && !isStandalone()) {
            const displayText = typeof node.data?.value === 'string' && !node.data.value.startsWith('data:') ? node.data.value : null;
            if (displayText != null && wfId) {
              try {
                const fileText = await getWorkflowFileText(wfId, (rawPort as { file_id: string }).file_id, node.id, 'out-text', (rawPort as any).run_id);
                if (fileText === displayText) {
                  sessionOutputs[node.id] = setOutputValueByPort(node.output_value, node.tool_id, 'out-text', rawPort);
                  continue;
                }
                if (fileText == null) {
                  // 轮询用尽仍 404：沿用原 file_id ref，不重复保存为新文件
                  sessionOutputs[node.id] = setOutputValueByPort(node.output_value, node.tool_id, 'out-text', rawPort);
                  continue;
                }
              } catch (_e) {
                // 拉取失败（如 GET /file 404 因库未及时可见）：视为竞态，沿用现有 ref 不重复保存
                sessionOutputs[node.id] = setOutputValueByPort(node.output_value, node.tool_id, 'out-text', rawPort);
                continue;
              }
            } else {
              sessionOutputs[node.id] = setOutputValueByPort(node.output_value, node.tool_id, 'out-text', rawPort);
              continue;
            }
          }
          const textVal = node.data?.value ?? rawPort;
          const str = typeof textVal === 'string' ? textVal : (textVal?.text ?? '');
          let nextPortKeyed = setOutputValueByPort(node.output_value, node.tool_id, 'out-text', str || undefined);
          let valueToStore: any = str || undefined;
          if (str && typeof str === 'string') {
            const dataUrl = `data:text/plain;charset=utf-8;base64,${btoa(unescape(encodeURIComponent(str)))}`;
            try {
              const ref = await saveInputFileViaOutputSave(wfId, node.id, 'out-text', dataUrl);
              if (ref) {
                nextPortKeyed = setOutputValueByPort(node.output_value, node.tool_id, 'out-text', ref);
                valueToStore = ref;
              }
            } catch (e) {
              console.warn('[WorkflowExecution] Save text-input as file failed:', e);
            }
          }
          sessionOutputs[node.id] = nextPortKeyed;
          const runTs = Date.now();
          const newEntries: NodeHistoryEntry[] = [];
          if (valueToStore != null) {
            const portIdForHist = INPUT_PORT_IDS[node.tool_id] || 'out-text';
            const entry = createHistoryEntryFromValue({ id: `node-${node.id}-${runTs}`, timestamp: runTs, value: valueToStore, execution_time: 0, portId: portIdForHist });
            if (entry) newEntries.push(entry);
          }
          setWorkflow(prev => {
            if (!prev) return null;
            const nextNodes = prev.nodes.map(n => n.id === node.id ? { ...n, output_value: nextPortKeyed, data: { ...n.data, value: valueToStore } } : n);
            const nextHistory = { ...(prev.nodeOutputHistory || {}) };
            if (newEntries.length > 0) {
              const prevList = normalizeHistoryEntries(nextHistory[node.id] || []);
              nextHistory[node.id] = [...newEntries, ...prevList].slice(0, MAX_NODE_HISTORY);
            }
            return { ...prev, nodes: nextNodes, nodeOutputHistory: nextHistory };
          });
          continue;
        }

        const nodeValue = getOutputValueByPort(node, portId);
        if (nodeValue == null) continue;

        const isDataUrl = (v: any) => typeof v === 'string' && v.startsWith('data:');
        const isImage = portId === 'out-image';
        const arr = Array.isArray(nodeValue) ? nodeValue : [nodeValue].filter(Boolean);

        if (arr.length > 0 && (isImage || arr.length === 1)) {
          const newOutputValue: any[] = [];
          let hasNewRefs = false;
          for (const item of arr) {
            if (isDataUrl(item)) {
              try {
                const ref = await saveInputFileViaOutputSave(wfId, node.id, portId, item);
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
          const finalVal = isImage ? newOutputValue : newOutputValue[0] ?? null;
          const nextPortKeyed = setOutputValueByPort(node.output_value, node.tool_id, portId, finalVal);
          if (hasNewRefs) {
            const runTs = Date.now();
            const newEntries: NodeHistoryEntry[] = [];
            if (inputNodesWithNewValue.has(node.id)) {
              const toHist = (v: any, i: number) => createHistoryEntryFromValue({
                id: `node-${node.id}-${runTs}-${i}`,
                timestamp: runTs,
                value: v,
                execution_time: 0,
                portId
              });
              if (isImage) newOutputValue.forEach((v, i) => { const e = toHist(v, i); if (e) newEntries.push(e); });
              else { const e = toHist(finalVal, 0); if (e) newEntries.push(e); }
            }
            setWorkflow(prev => {
              if (!prev) return null;
              const nextNodes = prev.nodes.map(n => n.id === node.id
                ? { ...n, output_value: nextPortKeyed, data: { ...n.data, value: finalVal } }
                : n);
              const nextHistory = { ...(prev.nodeOutputHistory || {}) };
              if (newEntries.length > 0) {
                const prevList = normalizeHistoryEntries(nextHistory[node.id] || []);
                nextHistory[node.id] = [...newEntries, ...prevList].slice(0, MAX_NODE_HISTORY);
              }
              return { ...prev, nodes: nextNodes, nodeOutputHistory: nextHistory };
            });
          }
          sessionOutputs[node.id] = nextPortKeyed;
        } else if (!Array.isArray(nodeValue) && isDataUrl(nodeValue)) {
          try {
            const ref = await saveInputFileViaOutputSave(wfId, node.id, portId, nodeValue);
            if (ref) {
              const runTs = Date.now();
              const entry = inputNodesWithNewValue.has(node.id)
                ? createHistoryEntryFromValue({ id: `node-${node.id}-${runTs}`, timestamp: runTs, value: ref, execution_time: 0, params: node?.data ?? {}, portId })
                : null;
              const nextPortKeyed = setOutputValueByPort(node.output_value, node.tool_id, portId, ref);
              setWorkflow(prev => {
                if (!prev) return null;
                const nextNodes = prev.nodes.map(n => n.id === node.id
                  ? { ...n, output_value: nextPortKeyed, data: { ...n.data, value: ref } }
                  : n);
                const nextHistory = { ...(prev.nodeOutputHistory || {}) };
                if (entry) {
                  const prevList = normalizeHistoryEntries(nextHistory[node.id] || []);
                  nextHistory[node.id] = [entry, ...prevList].slice(0, MAX_NODE_HISTORY);
                }
                return { ...prev, nodes: nextNodes, nodeOutputHistory: nextHistory };
              });
              sessionOutputs[node.id] = nextPortKeyed;
            } else {
              sessionOutputs[node.id] = setOutputValueByPort(node.output_value, node.tool_id, portId, nodeValue);
            }
          } catch (e) {
            console.warn('[WorkflowExecution] Save input node file at run time failed:', e);
            sessionOutputs[node.id] = setOutputValueByPort(node.output_value, node.tool_id, portId, nodeValue);
          }
        } else {
          sessionOutputs[node.id] = isPortKeyedOutputValue(node.output_value) ? node.output_value : setOutputValueByPort(node.output_value, node.tool_id, portId, nodeValue);
        }
      }
    }

    // Get LightX2V config once at the start of workflow execution
    const lightX2VConfig = getLightX2VConfig(workflow);

      try {
      // 按拓扑序分多批执行：每批内并行运行单节点（与节点上点击「运行此节点」同一套逻辑），批完成后保存再跑下一批
      // Accumulate nodeId -> duration across all batches for history entries (state updates are async)
      const executionTimeByNodeId: Record<string, number> = {};
      /** 节点失败时，其所有后置依赖加入此集合，不再参与执行 */
      const cancelledByFailure = new Set<string>();
      const hasValidDbId = wfId && (wfId.startsWith('workflow-') || wfId.match(/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i));
      const shouldSaveOutputs = hasValidDbId && !isStandalone();

      /** 为指定节点列表生成保存请求，用于每批完成后先保存再跑下一批，避免下游节点读不到结果 */
      const buildSavePromisesForNodeIds = (
        nodeIds: string[],
        sessionOut: Record<string, any>
      ): { savePromises: Promise<any>[]; saveMeta: Array<{ nodeId: string; portId: string; value: any; kind: 'single' | 'multi' | 'array'; index?: number }> } => {
        const savePromises: Promise<any>[] = [];
        const saveMeta: Array<{ nodeId: string; portId: string; value: any; kind: 'single' | 'multi' | 'array'; index?: number }> = [];
        // 每个节点分配一个 run_id，同一节点的多端口 save 共享同一 run_id
        const nodeRunIds = new Map<string, string>();
        const getRunId = (nodeId: string) => {
          let rid = nodeRunIds.get(nodeId);
          if (!rid) {
            rid = crypto.randomUUID();
            nodeRunIds.set(nodeId, rid);
          }
          return rid;
        };
        const scheduleOne = (
          nodeId: string,
          portId: string,
          value: any,
          _label: string,
          kind: 'single' | 'multi' | 'array' = 'single',
          index?: number,
          previousPromise?: Promise<any>
        ) => {
          saveMeta.push({ nodeId, portId, value, kind, ...(kind === 'array' && index !== undefined ? { index } : {}) });
          const runId = getRunId(nodeId);
          const perform = () =>
            saveNodeOutputs(wfId, nodeId, { [portId]: value }, runId).catch(err => {
              console.error(`[WorkflowExecution] Batch save ${nodeId}/${portId} failed:`, err);
              throw err;
            });
          const p = previousPromise ? previousPromise.then(() => perform()) : perform();
          savePromises.push(p);
          return p;
        };
        for (const nodeId of nodeIds) {
          const output = sessionOut[nodeId];
          const node = workflow.nodes.find(n => n.id === nodeId);
          if (!node) continue;
          const tool = TOOLS.find(t => t.id === node.tool_id);
          if (!tool?.outputs) continue;
          const outputToSave = output !== undefined ? output : (node.output_value !== undefined ? node.output_value : output);
          if (!outputToSave || (typeof outputToSave === 'string' && outputToSave.length === 0)) continue;
          if (isLightX2VResultRef(outputToSave)) {
            const firstOutputPort = tool.outputs[0];
            if (firstOutputPort) scheduleOne(nodeId, firstOutputPort.id, outputToSave, `node ${nodeId}/${firstOutputPort.id} (LightX2VResultRef)`);
          } else if (outputToSave && typeof outputToSave === 'object' && !Array.isArray(outputToSave)) {
            const entries = Object.entries(outputToSave);
            const singlePortTaskRef = entries.length === 1 && isLightX2VResultRef(entries[0][1]);
            if (singlePortTaskRef) {
              scheduleOne(nodeId, entries[0][0], entries[0][1], `node ${nodeId}/${entries[0][0]} (LightX2VResultRef)`);
            } else {
              // 同一节点多端口必须串行保存，否则并发请求各自 fetch workflow 后覆盖写，最后一个完成的会覆盖前面的 ref（如 scene1_prompt 被覆盖回纯文本）
              let prevP: Promise<any> | undefined;
              for (const [portId, value] of entries) {
                if ((typeof value === 'string' && value.length > 0) || (typeof value === 'object' && value !== null)) {
                  prevP = scheduleOne(nodeId, portId, value, `node ${nodeId}/${portId}`, 'multi', undefined, prevP);
                }
              }
            }
          } else if (typeof outputToSave === 'string' && outputToSave.length > 0) {
            const firstOutputPort = tool.outputs[0];
            if (firstOutputPort) scheduleOne(nodeId, firstOutputPort.id, outputToSave, `node ${nodeId}/${firstOutputPort.id}`);
          } else if (typeof outputToSave === 'object' && outputToSave !== null && !Array.isArray(outputToSave)) {
            const firstOutputPort = tool.outputs[0];
            if (firstOutputPort) scheduleOne(nodeId, firstOutputPort.id, outputToSave, `node ${nodeId}/${firstOutputPort.id} (JSON)`);
          } else if (Array.isArray(outputToSave)) {
            // 多图数组：整个 list 一次性传给后端，后端会遍历 list 逐项处理
            const firstOutputPort = tool.outputs[0];
            if (firstOutputPort) {
              const portId = firstOutputPort.id;
              const validItems = outputToSave.filter((item: any) =>
                (typeof item === 'string' && item.length > 0) || (typeof item === 'object' && item !== null)
              );
              if (validItems.length > 0) {
                scheduleOne(nodeId, portId, validItems, `node ${nodeId}/${portId} (array[${validItems.length}])`);
              }
            }
          }
        }
        return { savePromises, saveMeta };
      };

      const topologicalBatches = getTopologicalBatches(nodesToRunIds, workflow.connections);
      for (const batchNodeIds of topologicalBatches) {
        const batch = batchNodeIds
          .filter(id => !cancelledByFailure.has(id))
          .map(id => workflow.nodes.find(n => n.id === id))
          .filter((n): n is NonNullable<typeof workflow.nodes[0]> => n != null);
        if (batch.length === 0) continue;

        while (isPausedRef.current) {
          await new Promise(resolve => setTimeout(resolve, 100));
          const currentWorkflow = workflow;
          if (!currentWorkflow?.isRunning) return;
        }

        // 每批内并行执行各节点（与单节点运行同一套逻辑）
          const executionPromises = batch.map(async (node) => {
            const tool = TOOLS.find(t => t.id === node.tool_id)!;
            const incomingConns = workflow.connections.filter(c => c.target_node_id === node.id);
            const nodeStart = performance.now();

            // Update node status to RUNNING
            setWorkflow(prev => prev ? ({ ...prev, nodes: prev.nodes.map(n => n.id === node.id ? { ...n, status: NodeStatus.RUNNING, start_time: nodeStart } : n) }) : null);

            try {
              const nodeInputs: Record<string, any> = {};
              await Promise.all(tool.inputs.map(async (port) => {
                const conns = incomingConns.filter(c => c.target_port_id === port.id);
                if (conns.length > 0) {
                  const values = (await Promise.all(conns.map(async (c) => {
                  const sourceNode = workflow.nodes.find(n => n.id === c.source_node_id);
                  // 文本生成大模型输出若为对象（字典），传入下游时自动转为 JSON 字符串
                  const ensureStringIfObjectFromTextGen = (raw: any) => {
                    if (sourceNode?.tool_id === 'text-generation' && typeof raw === 'object' && raw !== null && !Array.isArray(raw))
                      return JSON.stringify(raw);
                    return raw;
                  };
                  // First check if source node has output in sessionOutputs (incl. input nodes after run-time save)
                  if (sessionOutputs[c.source_node_id] !== undefined) {
                    const sourceRes = sessionOutputs[c.source_node_id];
                    const fakeNode = sourceNode ? { ...sourceNode, output_value: sourceRes } : { output_value: sourceRes, tool_id: '', data: {} };
                    let raw = getOutputValueByPort(fakeNode, c.source_port_id);
                    if (raw === undefined) raw = (typeof sourceRes === 'object' && sourceRes !== null && c.source_port_id in sourceRes) ? sourceRes[c.source_port_id] : sourceRes;
                    if (isLightX2VResultRef(raw)) raw = await resolveLightX2VResultRef(raw);
                    // 输入节点等可能为 file ref：文本 .txt 解析为字符串，其余解析为 data URL
                    const isRef = (v: any) => v && typeof v === 'object' && (v.type === 'file' || v.kind === 'file' || v.file_id);
                    const isTextFileRef = (v: any) => isRef(v) && ((v as { mime_type?: string }).mime_type === 'text/plain' || (v as { ext?: string }).ext === '.txt' || (v as { ext?: string }).ext === 'txt');
                    const resolveFileRef = async (v: any): Promise<any> => {
                      if (!isRef(v)) return v;
                      const fid = (v as { file_url?: string }).file_url?.startsWith('local://') ? (v as { file_url: string }).file_url : (v as { file_id: string }).file_id;
                      const rid = (v as { run_id?: string }).run_id;
                      if (isTextFileRef(v)) {
                        return await getWorkflowFileText(wfId, fid, c.source_node_id, c.source_port_id, rid) ?? v;
                      }
                      return await getWorkflowFileByFileId(wfId, fid, (v as any).mime_type, (v as any).ext, c.source_node_id, c.source_port_id, rid) || (v as { file_url?: string }).file_url || v;
                    };
                    if (isRef(raw)) {
                      raw = await resolveFileRef(raw);
                    } else if (Array.isArray(raw) && raw.some((v: any) => isRef(v))) {
                      raw = await Promise.all(raw.map(resolveFileRef));
                    }
                    return ensureStringIfObjectFromTextGen(raw);
                  }
                  // If not executed yet, check if it's an input node and read from node.data.value
                  // This handles the case where input nodes haven't been executed but their values are needed
                  if (sourceNode) {
                    const sourceTool = TOOLS.find(t => t.id === sourceNode.tool_id);
                    if (sourceTool?.category === 'Input') {
                      // For input nodes, read from node.data.value 或 output_value（保存后文本在 output_value 中为 file ref）
                      let inputValue = sourceNode.data.value ?? (sourceNode.tool_id === 'text-input' ? getOutputValueByPort(sourceNode, 'out-text') : undefined);
                      if (sourceNode.tool_id === 'text-input' && inputValue && typeof inputValue === 'object' && (inputValue as any).file_id && ((inputValue as any).mime_type === 'text/plain' || (inputValue as any).ext === '.txt' || (inputValue as any).ext === 'txt')) {
                        inputValue = await getWorkflowFileText(wfId, (inputValue as { file_id: string }).file_id, sourceNode.id, c.source_port_id, (inputValue as any).run_id) ?? inputValue;
                      }
                      // Convert file paths to base64 data URLs for image and audio inputs
                      if (sourceNode.tool_id === 'image-input' && Array.isArray(inputValue) && inputValue.length > 0) {
                        inputValue = await Promise.all(inputValue.map(async (img: string | { type?: string; file_id?: string; file_url?: string }) => {
                          // Backend file reference: resolve file_id to data URL
                          if (img && typeof img === 'object' && (img.type === 'file' || img.file_id)) {
                            const dataUrl = await getWorkflowFileByFileId(wfId, (img as { file_id: string }).file_id, (img as any).mime_type, (img as any).ext, sourceNode.id, c.source_port_id, (img as any).run_id);
                            return dataUrl || (img as { file_url?: string }).file_url || img;
                          }
                          if (typeof img !== 'string') return img;
                          // If it's already a data URL or base64, return as is
                          if (img.startsWith('data:') || (!img.startsWith('http') && img.includes(','))) {
                            return img;
                          }
                          // Standalone: resolve local:// from IndexedDB
                          if (img.startsWith('local://')) {
                            const dataUrl = await getWorkflowFileByFileId(wfId, img);
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
                      } else if (sourceNode.tool_id === 'audio-input' && inputValue) {
                        if (inputValue && typeof inputValue === 'object' && (inputValue as { type?: string; file_id?: string }).file_id) {
                          const dataUrl = await getWorkflowFileByFileId(wfId, (inputValue as { file_id: string }).file_id, (inputValue as any).mime_type, (inputValue as any).ext, sourceNode.id, c.source_port_id, (inputValue as any).run_id);
                          if (dataUrl) inputValue = dataUrl;
                        } else if (typeof inputValue === 'string' && inputValue.startsWith('local://')) {
                          const dataUrl = await getWorkflowFileByFileId(wfId, inputValue);
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
                      } else if (sourceNode.tool_id === 'video-input' && inputValue) {
                        if (inputValue && typeof inputValue === 'object' && (inputValue as { type?: string; file_id?: string }).file_id) {
                          const dataUrl = await getWorkflowFileByFileId(wfId, (inputValue as { file_id: string }).file_id, (inputValue as any).mime_type, (inputValue as any).ext, sourceNode.id, c.source_port_id, (inputValue as any).run_id);
                          if (dataUrl) inputValue = dataUrl;
                        } else if (typeof inputValue === 'string' && inputValue.startsWith('local://')) {
                          const dataUrl = await getWorkflowFileByFileId(wfId, inputValue);
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

                      // Check if this is a multi-output node (like text-generation with custom_outputs)
                      if (sourceNode.tool_id === 'text-generation' && sourceNode.data.custom_outputs && typeof inputValue === 'object' && inputValue !== null) {
                        const raw = c.source_port_id in inputValue ? inputValue[c.source_port_id] : inputValue;
                        return ensureStringIfObjectFromTextGen(raw);
                      }
                      return inputValue;
                    }
                    // For other nodes that haven't executed, try node.output_value (e.g. from load or previous run)
                    const prevOutput = sourceNode?.output_value;
                    if (prevOutput !== undefined) {
                      let raw = (typeof prevOutput === 'object' && prevOutput !== null && c.source_port_id in prevOutput) ? prevOutput[c.source_port_id] : prevOutput;
                      if (raw && typeof raw === 'object' && (raw as any).file_id && ((raw as any).mime_type === 'text/plain' || (raw as any).ext === '.txt' || (raw as any).ext === 'txt')) {
                        raw = await getWorkflowFileText(wfId, (raw as { file_id: string }).file_id, c.source_node_id, c.source_port_id, (raw as any).run_id) ?? raw;
                      }
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
              switch (node.tool_id) {
                case 'text-input': result = node.data.value || ""; break;
                case 'image-input': {
                  const rawImage = node.output_value ?? node.data.value;
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
                        const dataUrl = await getWorkflowFileByFileId(wfId, idToFetch, (img as any).mime_type, (img as any).ext, node.id, 'out-image', (img as any).run_id);
                        return dataUrl || (img as { file_url?: string }).file_url || img;
                      }
                      if (typeof img !== 'string') return img;
                      // If it's already a data URL or base64, return as is
                      if (img.startsWith('data:') || (!img.startsWith('http') && img.includes(','))) {
                        return img;
                      }
                      // Standalone: resolve local:// from IndexedDB
                      if (img.startsWith('local://')) {
                        const dataUrl = await getWorkflowFileByFileId(wfId, img);
                        return dataUrl || img;
                      }
                      // If it's a workflow input path or task result path, use directly
                      if (img.includes('/assets/workflow/file') || img.includes('/assets/task/') ||
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
                  const audioValue = node.output_value ?? node.data.value;
                  if (audioValue && typeof audioValue === 'object' && (audioValue as { file_id?: string }).file_id) {
                    const idToFetch = (audioValue as { file_url?: string }).file_url?.startsWith('local://')
                      ? (audioValue as { file_url: string }).file_url
                      : (audioValue as { file_id: string }).file_id;
                    result = await getWorkflowFileByFileId(wfId, idToFetch, (audioValue as any).mime_type, (audioValue as any).ext, node.id, 'out-audio', (audioValue as any).run_id) || (audioValue as { file_url?: string }).file_url || audioValue;
                  } else if (audioValue && typeof audioValue === 'string') {
                    if (audioValue.startsWith('local://')) {
                      result = await getWorkflowFileByFileId(wfId, audioValue) || audioValue;
                    } else if (audioValue.startsWith('data:') || (!audioValue.startsWith('http') && audioValue.includes(','))) {
                      result = audioValue;
                    } else if (audioValue.includes('/assets/workflow/file') || audioValue.includes('/assets/task/') ||
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
                  const vidVal = node.output_value ?? node.data.value;
                  if (vidVal && typeof vidVal === 'object' && (vidVal as { file_id?: string }).file_id) {
                    const idToFetch = (vidVal as { file_url?: string }).file_url?.startsWith('local://')
                      ? (vidVal as { file_url: string }).file_url
                      : (vidVal as { file_id: string }).file_id;
                    result = await getWorkflowFileByFileId(wfId, idToFetch, (vidVal as any).mime_type, (vidVal as any).ext, node.id, 'out-video', (vidVal as any).run_id) || (vidVal as { file_url?: string }).file_url || vidVal;
                  } else if (typeof vidVal === 'string') {
                    if (vidVal.startsWith('local://')) {
                      result = await getWorkflowFileByFileId(wfId, vidVal) || vidVal;
                    } else if (vidVal.startsWith('data:') || (!vidVal.startsWith('http') && vidVal.includes(','))) {
                      result = vidVal;
                    } else if (vidVal.includes('/assets/workflow/file') || vidVal.includes('/assets/task/') ||
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
                case 'text-generation': {
                  const customOutputs = (node.data.custom_outputs || []) as { id: string; label?: string; description?: string }[];
                  const outputFields = customOutputs.map((o: any) => ({ id: o.id, description: o.description || o.label }));
                  const useSearch = node.data.useSearch || false;
                  let rawResult: any;
                  if (model && model.startsWith('deepseek-')) {
                    rawResult = await deepseekText(nodeInputs['in-text'] || "...", node.data.mode, node.data.customInstruction, model, outputFields, useSearch);
                  } else if (model && model.startsWith('doubao-')) {
                    const imageInputRaw = flattenImageInput(nodeInputs['in-image']);
                    const imageInput = imageInputRaw.length > 0 ? await ensureImageInputsAsDataUrls(imageInputRaw) : undefined;
                    rawResult = await doubaoText(nodeInputs['in-text'] || "...", node.data.mode, node.data.customInstruction, model, outputFields, imageInput?.length ? imageInput : undefined, useSearch);
                  } else if (model && model.startsWith('ppchat-')) {
                    const imageInputRaw = flattenImageInput(nodeInputs['in-image']);
                    const imageInput = imageInputRaw.length > 0 ? await ensureImageInputsAsDataUrls(imageInputRaw) : undefined;
                    rawResult = await ppchatGeminiText(nodeInputs['in-text'] || "...", node.data.mode, node.data.customInstruction, model.replace('ppchat-', ''), outputFields, imageInput?.length ? imageInput : undefined);
                  } else {
                    const imageInputRaw = flattenImageInput(nodeInputs['in-image']);
                    const imageInput = imageInputRaw.length > 0 ? await ensureImageInputsAsDataUrls(imageInputRaw) : undefined;
                    rawResult = await geminiText(nodeInputs['in-text'] || "...", false, node.data.mode, node.data.customInstruction, model, outputFields, imageInput?.length ? imageInput : undefined);
                  }
                  if (customOutputs.length > 0 && rawResult != null && typeof rawResult === 'object' && !Array.isArray(rawResult)) {
                    const portIds = customOutputs.map(o => o.id);
                    const portKeyedOnly: Record<string, any> = {};
                    portIds.forEach((portId, i) => {
                      const fromRaw = rawResult[portId];
                      if (fromRaw !== undefined && fromRaw !== null) {
                        portKeyedOnly[portId] = fromRaw;
                      } else {
                        portKeyedOnly[portId] = (node.output_value && typeof node.output_value === 'object' && portId in node.output_value)
                          ? node.output_value[portId]
                          : (i === 0 && typeof rawResult === 'object' && Object.keys(rawResult).length > 0 ? Object.values(rawResult)[0] : '...');
                      }
                    });
                    result = portKeyedOnly;
                  } else {
                    result = rawResult;
                  }
                  break;
                }
                case 'text-to-image':
                  if (model === 'gemini-2.5-flash-image') {
                    result = await geminiImage(nodeInputs['in-text'] || "Artistic portrait", undefined, node.data.aspectRatio, model);
                  } else {
                    // Get config for this specific model (handles -cloud suffix)
                    const t2iModelConfig = getLightX2VConfigForModel(model || 'Qwen-Image-2512');
                    const t2iWorkflowRefs = !t2iModelConfig.isCloud && wfId ? buildWorkflowRefs(wfId, incomingConns, tool.inputs) : undefined;
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
                      (taskId) => registerTaskId(node.id, taskId, t2iModelConfig.isCloud),
                      jobAbortSignal,
                      t2iWorkflowRefs
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
                    const i2iModelConfig = getLightX2VConfigForModel(model || 'Qwen-Image-Edit-2511');
                    let imageInput: string | string[] | undefined;
                    if (i2iModelConfig.isCloud && workflow?.id) {
                      // 云端：file ref 必须解析为 data URL，否则后端无法识别
                      const raw = nodeInputs['in-image'];
                      const resolved = Array.isArray(raw)
                        ? await Promise.all(raw.map((item: any) => resolveMediaForCloud(item, wfId)))
                        : await resolveMediaForCloud(raw, wfId);
                      const list = Array.isArray(resolved)
                        ? resolved.filter((x): x is string => typeof x === 'string')
                        : (typeof resolved === 'string' ? [resolved] : []);
                      imageInput = list.length === 0 ? undefined : (list.length === 1 ? list[0] : list);
                    } else {
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
                      imageInput = i2iImgsBase64.length === 0 ? undefined : (i2iImgsBase64.length === 1 ? i2iImgsBase64[0] : i2iImgsBase64);
                    }
                    const i2iWorkflowRefs = !i2iModelConfig.isCloud && wfId ? buildWorkflowRefs(wfId, incomingConns, tool.inputs) : undefined;
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
                      (taskId) => registerTaskId(node.id, taskId, i2iModelConfig.isCloud),
                      jobAbortSignal,
                      i2iWorkflowRefs
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
                  const t2vWorkflowRefs = !t2vModelConfig.isCloud && wfId ? buildWorkflowRefs(wfId, incomingConns, tool.inputs) : undefined;
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
                    jobAbortSignal,
                    t2vWorkflowRefs
                  );
                  const t2vTid = jobInfo.taskIdsByNodeId.get(node.id);
                  if (t2vTid) result = toLightX2VResultRef(t2vTid, 'output_video', (model || '').endsWith('-cloud'));
                  break;
                case 'video-gen-image': {
                  const startImg = Array.isArray(nodeInputs['in-image']) ? nodeInputs['in-image'][0] : nodeInputs['in-image'];
                  // Get config for this specific model (handles -cloud suffix)
                  const i2vModelConfig = getLightX2VConfigForModel(model || 'Wan2.2_I2V_A14B_distilled');
                  const i2vWorkflowRefs = !i2vModelConfig.isCloud && wfId ? buildWorkflowRefs(wfId, incomingConns, tool.inputs) : undefined;
                  const startImgForSubmit = i2vModelConfig.isCloud && workflow?.id
                    ? (typeof startImg === 'object' && (startImg as any)?.file_id ? await resolveMediaForCloud(startImg, wfId) : (await ensureLocalInputsAsDataUrls(flattenImageInput(startImg)))[0])
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
                    (taskId) => registerTaskId(node.id, taskId, i2vModelConfig.isCloud),
                    jobAbortSignal,
                    i2vWorkflowRefs
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
                    const flf2vWorkflowRefs = !flf2vModelConfig.isCloud && wfId ? buildWorkflowRefs(wfId, incomingConns, tool.inputs) : undefined;
                    const dualStartForSubmit = flf2vModelConfig.isCloud && workflow?.id
                      ? (typeof dualStart === 'object' && (dualStart as any)?.file_id ? await resolveMediaForCloud(dualStart, wfId) : (await ensureLocalInputsAsDataUrls(flattenImageInput(dualStart)))[0])
                      : dualStart;
                    const dualEndForSubmit = flf2vModelConfig.isCloud && workflow?.id
                      ? (typeof dualEnd === 'object' && (dualEnd as any)?.file_id ? await resolveMediaForCloud(dualEnd, wfId) : (await ensureLocalInputsAsDataUrls(flattenImageInput(dualEnd)))[0])
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
                        jobAbortSignal,
                        flf2vWorkflowRefs
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
                    const animateWorkflowRefs = !animateModelConfig.isCloud && wfId ? buildWorkflowRefs(wfId, incomingConns, tool.inputs) : undefined;
                    const swapImgForSubmit = animateModelConfig.isCloud && wfId ? await resolveMediaForCloud(swapImg, wfId) : swapImg;
                    const swapVidForSubmit = animateModelConfig.isCloud && wfId ? await resolveMediaForCloud(swapVid, wfId) : swapVid;
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
                      (taskId) => registerTaskId(node.id, taskId, animateModelConfig.isCloud),
                      jobAbortSignal,
                      animateWorkflowRefs
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
                  const s2vWorkflowRefs = !s2vModelConfig.isCloud && wfId ? buildWorkflowRefs(wfId, incomingConns, tool.inputs) : undefined;
                  const avatarImgForSubmit = s2vModelConfig.isCloud && wfId ? await resolveMediaForCloud(avatarImg, wfId) : avatarImg;
                  const avatarAudioForSubmit = s2vModelConfig.isCloud && wfId ? await resolveMediaForCloud(avatarAudio, wfId) : avatarAudio;
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
                    (taskId) => registerTaskId(node.id, taskId, s2vModelConfig.isCloud),
                    jobAbortSignal,
                    s2vWorkflowRefs
                  );
                  const s2vTid = jobInfo.taskIdsByNodeId.get(node.id);
                  if (s2vTid) result = toLightX2VResultRef(s2vTid, 'output_video', (model || '').endsWith('-cloud'));
                  break;
                default: result = "Processed";
              }
              const nodeDuration = performance.now() - nodeStart;

            // Normalize result to valueToStore (port-keyed file ref for workflow file URL, etc.)
            let valueToStore: any = result;
            const isOldUrl = typeof result === 'string' && result.includes('/api/v1/workflow/') && result.includes('/file/');
            const isNewUrl = typeof result === 'string' && result.includes('/assets/workflow/file');
            if (isOldUrl || isNewUrl) {
              const fileIdMatch = isNewUrl ? result.match(/[?&]file_id=([^&]+)/) : result.match(/\/file\/([^/?]+)/);
              if (fileIdMatch) {
                const fileId = fileIdMatch[1];
                let mime = 'application/octet-stream';
                const mimeMatch = result.match(/mime_type=([^&]+)/);
                if (mimeMatch) mime = decodeURIComponent(mimeMatch[1].replace(/\+/g, ' '));
                let ext: string | undefined;
                const extMatch = result.match(/[?&]ext=([^&]+)/);
                if (extMatch) ext = decodeURIComponent(extMatch[1]).replace(/\+/g, ' ');
                const portId = tool?.outputs?.[0]?.id;
                if (portId) valueToStore = { [portId]: { kind: 'file', file_id: fileId, mime_type: mime, ...(ext && { ext: ext.startsWith('.') ? ext : `.${ext}` }) } };
              }
            }
            // 单结果值（task/file ref 等）统一为 port_keyed { "out-xxx": value }，与后端 /output/{port_id}/url 等约定一致
            if (tool?.outputs?.length && !isPortKeyedOutputValue(valueToStore)) {
              const portId = tool.outputs[0].id;
              valueToStore = { [portId]: valueToStore };
            }
            sessionOutputs[node.id] = valueToStore;
            // Update this node in UI as soon as it completes (don't wait for whole batch)
            setWorkflow(prev => {
              if (!prev) return null;
              const idx = prev.nodes.findIndex(n => n.id === node.id);
              if (idx < 0) return prev;
              const updated = [...prev.nodes];
              updated[idx] = { ...updated[idx], status: NodeStatus.SUCCESS, execution_time: nodeDuration, output_value: valueToStore, completed_at: Date.now() };
              return { ...prev, nodes: updated };
            });
            return { nodeId: node.id, result: valueToStore, duration: nodeDuration };
            } catch (err: any) {
              const nodeDuration = performance.now() - nodeStart;
              if (err?.message?.includes?.("Requested entity was not found")) {
                await (window as any).aistudio.openSelectKey();
              }
              const errorMessage = err?.message ?? (typeof err === 'string' ? err : String(err ?? 'Unknown execution error'));
              // 立即同步更新节点为 ERROR，避免状态停留在 RUNNING
              flushSync(() => {
                setWorkflow(prev => prev ? ({
                  ...prev,
                  nodes: prev.nodes.map(n => n.id === node.id ? {
                    ...n,
                    status: NodeStatus.ERROR,
                    error: errorMessage,
                    execution_time: nodeDuration,
                    completed_at: Date.now()
                  } : n)
                }) : null);
              });
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
              // sessionOutputs and UI already updated per-node when promise resolved
              successfulResults.push({ nodeId, result, duration });
              executedInSession.add(nodeId);
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
                const nodeDuration = performance.now() - (node.start_time || performance.now());
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

          // Successful nodes already updated in UI when each promise resolved; successfulResults still used for batch save below

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
                    execution_time: duration,
                    completed_at: updatedNodes[index].completed_at ?? Date.now()
                  };
                }
              });
              return { ...prev, nodes: updatedNodes };
            });
            // 失败节点的所有后置依赖不再执行：加入 cancelledByFailure 与 executedInSession，并置为 IDLE
            const downstreamToCancel = new Set<string>();
            failedNodes.forEach(({ nodeId }) => {
              const desc = getDescendants(nodeId, workflow.connections);
              desc.forEach(d => {
                if (nodesToRunIds.has(d) && !executedInSession.has(d)) downstreamToCancel.add(d);
              });
            });
            downstreamToCancel.forEach(id => {
              cancelledByFailure.add(id);
              executedInSession.add(id);
            });
            if (downstreamToCancel.size > 0) {
              setWorkflow(prev => {
                if (!prev) return null;
                return {
                  ...prev,
                  nodes: prev.nodes.map(n => downstreamToCancel.has(n.id)
                    ? { ...n, status: NodeStatus.IDLE, error: undefined, execution_time: undefined, start_time: undefined }
                    : n)
                };
              });
            }
          }
          // 每批完成后立即保存到后端并等待，再跑下一批，避免下游节点或 task 提交时后端读不到结果
          if (shouldSaveOutputs && successfulResults.length > 0) {
            const batchNodeIds = successfulResults.map(r => r.nodeId);
            const { savePromises: batchSavePromises, saveMeta: batchSaveMeta } = buildSavePromisesForNodeIds(batchNodeIds, sessionOutputs);
            if (batchSavePromises.length > 0) {
              const batchResults = await Promise.allSettled(batchSavePromises);
              const batchUpdatedOutputs: Record<string, any> = {};
              batchResults.forEach((settled, idx) => {
                if (settled.status !== 'fulfilled' || !settled.value) return;
                const meta = batchSaveMeta[idx];
                if (!meta) return;
                const rawResult = settled.value;
                const perPortResult = typeof rawResult === 'object' && rawResult !== null && !Array.isArray(rawResult)
                  ? (rawResult as Record<string, any>)
                  : null;
                const saveResultRaw = perPortResult?.[meta.portId] ?? null;
                if (!saveResultRaw || typeof saveResultRaw !== 'object') return;

                /** 将 saveNodeOutputs 返回的单条结果转为 output_value 使用的 ref payload */
                const toRefPayload = (sr: any): any => {
                  if (!sr || typeof sr !== 'object') return null;
                  const kind = sr.kind;
                  const srRunId = sr.run_id;
                  if (kind === 'file' && sr.file_id) {
                    const mimeType = sr.mime_type ?? (sr.ext ? extToMimeType(sr.ext) : undefined);
                    const extNorm = sr.ext != null ? (sr.ext.startsWith('.') ? sr.ext : `.${sr.ext}`) : undefined;
                    return { kind: 'file' as const, file_id: sr.file_id, mime_type: mimeType, ...(extNorm != null && { ext: extNorm }), ...(srRunId && { run_id: srRunId }) };
                  }
                  if (kind === 'task') {
                    return { kind: 'task' as const, task_id: sr.task_id, output_name: sr.output_name, is_cloud: sr.is_cloud };
                  }
                  if (kind === 'url') {
                    return { kind: 'url' as const, url: sr.url };
                  }
                  // fallback: file_id 存在时仍然作为 file ref
                  if (sr.file_id) {
                    const mimeType = sr.mime_type ?? (sr.ext ? extToMimeType(sr.ext) : undefined);
                    const extNorm = sr.ext != null ? (sr.ext.startsWith('.') ? sr.ext : `.${sr.ext}`) : undefined;
                    return { kind: 'file' as const, file_id: sr.file_id, mime_type: mimeType, ...(extNorm != null && { ext: extNorm }), ...(srRunId && { run_id: srRunId }) };
                  }
                  return null;
                };

                // 处理 file_list（多图 entries 数组）
                if (saveResultRaw.kind === 'file_list' && Array.isArray(saveResultRaw.entries)) {
                  const refList = saveResultRaw.entries.map((e: any) => toRefPayload(e)).filter((r: any) => r != null);
                  if (refList.length === 0) return;
                  const refPayload = refList.length === 1 ? refList[0] : refList;
                  if (!batchUpdatedOutputs[meta.nodeId]) batchUpdatedOutputs[meta.nodeId] = { [meta.portId]: refPayload };
                  else (batchUpdatedOutputs[meta.nodeId] as Record<string, any>)[meta.portId] = refPayload;
                  return;
                }

                const refPayload = toRefPayload(saveResultRaw);
                if (!refPayload) return;
                // 单端口节点（如 gemini-watermark-remover）也按端口字典写入，与图生图等一致
                if (!batchUpdatedOutputs[meta.nodeId]) batchUpdatedOutputs[meta.nodeId] = meta.kind === 'multi' ? {} : meta.kind === 'array' ? [] : { [meta.portId]: refPayload };
                else if (meta.kind === 'multi') (batchUpdatedOutputs[meta.nodeId] as Record<string, any>)[meta.portId] = refPayload;
                else if (meta.kind === 'array') (batchUpdatedOutputs[meta.nodeId] as any[])[meta.index ?? 0] = refPayload;
                else (batchUpdatedOutputs[meta.nodeId] as Record<string, any>)[meta.portId] = refPayload;
              });
              if (Object.keys(batchUpdatedOutputs).length > 0) {
                for (const [nodeId, refOutput] of Object.entries(batchUpdatedOutputs)) {
                  const existing = sessionOutputs[nodeId];
                  // 多端口节点：只保存了部分端口为 file ref，合并到现有 output，避免覆盖未保存的端口（如纯文本）
                  if (typeof refOutput === 'object' && refOutput !== null && !Array.isArray(refOutput) && typeof existing === 'object' && existing !== null && !Array.isArray(existing) && !(existing as { kind?: string }).kind) {
                    sessionOutputs[nodeId] = { ...existing, ...refOutput };
                  } else {
                    sessionOutputs[nodeId] = refOutput;
                  }
                }
                setWorkflow(prev => {
                  if (!prev) return null;
                  return {
                    ...prev,
                    nodes: prev.nodes.map(n => {
                      const refOutput = batchUpdatedOutputs[n.id];
                      if (!refOutput) return n;
                      const tool = TOOLS.find(t => t.id === n.tool_id);
                      const portId = tool?.outputs?.[0]?.id;
                      // 单端口 + 单个 ref（file/task/url）
                      const isSingleRef = typeof refOutput === 'object' && !Array.isArray(refOutput) && typeof (refOutput as any).kind === 'string';
                      if (portId && isSingleRef) {
                        const ov = n.output_value;
                        const nextOv = (ov && typeof ov === 'object' && !Array.isArray(ov)) ? { ...ov } : {};
                        (nextOv as Record<string, any>)[portId] = refOutput;
                        return { ...n, output_value: nextOv };
                      }
                      // 多端口字典（port_id -> ref）
                      if (typeof refOutput === 'object' && !Array.isArray(refOutput) && isPortKeyedOutputValue(refOutput)) {
                        const ov = n.output_value;
                        const nextOv = (ov && typeof ov === 'object' && !Array.isArray(ov)) ? { ...ov, ...refOutput } : { ...refOutput };
                        return { ...n, output_value: nextOv };
                      }
                      // 数组（多图 ref list）
                      if (Array.isArray(refOutput)) return { ...n, output_value: refOutput };
                      return n;
                    })
                  };
                });
              }
            }
          }
        }
      const runTotalTime = performance.now() - runStartTime;
      const runTimestamp = Date.now();

      // Optimize history storage: only keep essential data
      // Create a lightweight snapshot for nodeOutputHistory entry
      const lightweightNodesSnapshot = workflow.nodes.map(n => ({
        id: n.id,
        tool_id: n.tool_id,
        x: n.x,
        y: n.y,
        status: n.status,
        data: { ...n.data },
        error: n.error,
        execution_time: n.execution_time,
        completed_at: n.completed_at
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
          historyReadyOutputs[nodeId] = { kind: 'file', data: output };
        } else if (isLightX2VResultRef(output)) {
          // Keep task ref as-is so createHistoryEntryFromValue produces kind: 'task'
          historyReadyOutputs[nodeId] = output;
        } else if (typeof output === 'object' && output !== null) {
          historyReadyOutputs[nodeId] = output;
        } else {
          historyReadyOutputs[nodeId] = output;
        }
      }

      // Per-node history: only for nodes in this run set that were actually executed (not upstream nodes whose output was just read as input).
      const nodeHistoryUpdates: Record<string, NodeHistoryEntry[]> = {};
      if (!shouldSaveOutputs) {
        for (const nodeId of executedInSession) {
          if (!nodesToRunIds.has(nodeId)) continue; // Only nodes we chose to run get history
          const node = workflow.nodes.find(n => n.id === nodeId);
          let output = sessionOutputs[nodeId];
          if (output != null && typeof output === 'object' && !Array.isArray(output) && output.output_value != null && typeof output.output_value === 'object' && Object.keys(output.output_value).some((k: string) => k.startsWith('out-'))) {
            output = output.output_value as Record<string, any>;
          }
          const isPortKeyed = typeof output === 'object' && output !== null && !Array.isArray(output) &&
            Object.keys(output).some((k: string) => k.startsWith('out-'));
          const entry = isPortKeyed
            ? createHistoryEntryFromPortKeyedOutputValue({
                id: `node-${nodeId}-${runTimestamp}`,
                timestamp: runTimestamp,
                output_value: output as Record<string, any>,
                execution_time: executionTimeByNodeId[nodeId] ?? node?.execution_time,
                params: node?.data ?? {}
              })
            : (() => {
                const tool = node ? TOOLS.find(t => t.id === node.tool_id) : undefined;
                const portIdForHist = node ? (INPUT_PORT_IDS[node.tool_id] ?? tool?.outputs?.[0]?.id) : undefined;
                return createHistoryEntryFromValue({
                  id: `node-${nodeId}-${runTimestamp}`,
                  timestamp: runTimestamp,
                  value: historyReadyOutputs[nodeId] ?? output,
                  execution_time: executionTimeByNodeId[nodeId] ?? node?.execution_time,
                  portId: portIdForHist
                });
              })();
          if (!entry) continue;
          const prevListRaw = (workflow.nodeOutputHistory && workflow.nodeOutputHistory[nodeId]) || [];
          const prevList = normalizeHistoryEntries(prevListRaw);
          nodeHistoryUpdates[nodeId] = [entry, ...prevList].slice(0, MAX_NODE_HISTORY);
        }
      }
      /** Persist execution state once; strips base64 and drops Input data.value. Used after save outputs (single PUT) or when no output save (setTimeout). */
      const doPersistExecutionState = async (wfId: string, wf: WorkflowState, nodes: any[], nodeOutputHistory: Record<string, any[]>) => {
        const { nodes: safeNodes, node_output_history: safeHistory } = stripBase64FromWorkflowPayload(nodes, nodeOutputHistory);
        const nodesForPayload = safeNodes.map((n: any) => {
          const tool = TOOLS.find(t => t.id === n.tool_id);
          if (tool?.category === 'Input') {
            const { value: _dropped, ...restData } = n.data || {};
            return { ...n, data: restData };
          }
          return n;
        });
        try {
          const currentUserId = getCurrentUserId();
          const { owned } = await checkWorkflowOwnership(wfId, currentUserId);
          await workflowSaveQueue.enqueue(wfId, async () => {
            if (isStandalone() && onSaveExecutionToLocal) {
              const newId = !owned
                ? (typeof crypto !== 'undefined' && crypto.randomUUID ? crypto.randomUUID() : `workflow-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`)
                : wfId;
              const updatedWorkflow: WorkflowState = {
                ...wf,
                id: newId,
                nodes: nodesForPayload,
                nodeOutputHistory: safeHistory,
                updatedAt: Date.now()
              };
              await onSaveExecutionToLocal(updatedWorkflow);
              if (newId !== wfId) {
                setWorkflow(prev => prev ? { ...prev, id: newId } : null);
                if (window.history?.replaceState) window.history.replaceState(null, '', `#workflow/${newId}`);
              }
              return;
            }
            if (!owned) {
              try {
                const newId = typeof crypto !== 'undefined' && crypto.randomUUID ? crypto.randomUUID() : `workflow-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
                const createResponse = await apiRequest('/api/v1/workflow/create', {
                  method: 'POST',
                  body: JSON.stringify({
                    name: wf.name,
                    description: wf.description ?? '',
                    nodes: nodesForPayload,
                    connections: wf.connections,
                    tags: wf.tags ?? [],
                    node_output_history: safeHistory,
                    workflow_id: newId
                  })
                });
                if (createResponse.ok) {
                  const data = await createResponse.json();
                  const finalWorkflowId = data.workflow_id;
                  setWorkflow(prev => prev ? { ...prev, id: finalWorkflowId } : null);
                  if (window.history?.replaceState) window.history.replaceState(null, '', `#workflow/${finalWorkflowId}`);
                } else {
                  throw new Error(await createResponse.text());
                }
              } catch (error) {
                setGlobalError({ message: 'Failed to save execution state', details: error instanceof Error ? error.message : String(error) });
              }
              return;
            }
            try {
              const updateResponse = await apiRequest(`/api/v1/workflow/${wfId}/update`, {
                method: 'POST',
                body: JSON.stringify({ nodes: nodesForPayload, connections: wf.connections, node_output_history: safeHistory })
              });
              if (!updateResponse.ok) throw new Error(await updateResponse.text());
              console.log('[WorkflowExecution] Execution state saved successfully');
            } catch (err) {
              const isNetworkError = err instanceof TypeError || (err instanceof Error && err.message.includes('fetch')) || !navigator.onLine;
              if (isNetworkError) workflowOfflineQueue.addTask(wfId, { nodes: nodesForPayload, connections: wf.connections, node_output_history: safeHistory });
              setGlobalError({ message: 'Failed to save execution state', details: err instanceof Error ? err.message : String(err) });
            }
          });
        } catch (err) {
          setGlobalError({ message: 'Failed to save execution state', details: err instanceof Error ? err.message : String(err) });
        }
      };

      if (shouldSaveOutputs) {
        // 已在每批完成后保存并等待，此处仅合并历史并持久化 workflow 状态
        // 用 sessionOutputs 覆盖节点 output_value；若为 Input 且覆盖值会 strip 成 undefined（如仍是 data URL），则保留 baseNodes 中已有 ref，避免图像输入等被清空
        try {
          const latest = workflowRef.current;
          const baseNodes = latest?.nodes ?? workflow.nodes;
          const mergedNodes = baseNodes.map((n: any) => {
            const so = sessionOutputs[n.id];
            if (so === undefined) return n;
            const tool = TOOLS.find(t => t.id === n.tool_id);
            const wouldStripToUndefined = (v: any) => {
              if (v == null) return true;
              const stripped = stripBase64FromOutputValue(normalizeNodeOutputValueForPersist(v, n.tool_id));
              return stripped === undefined;
            };
            if (tool?.category === 'Input' && wouldStripToUndefined(so) && n.output_value != null && !wouldStripToUndefined(n.output_value))
              return n;
            return { ...n, output_value: so };
          });
          const mergedHistory: Record<string, NodeHistoryEntry[]> = { ...(latest?.nodeOutputHistory ?? workflow.nodeOutputHistory ?? {}) };
          for (const nodeId of executedInSession) {
            if (!nodesToRunIds.has(nodeId)) continue;
            const node = mergedNodes.find((n: any) => n.id === nodeId);
            if (!node) continue;
            let outputValue = node.output_value;
            if (outputValue != null && typeof outputValue === 'object' && !Array.isArray(outputValue) && Object.keys(outputValue).some((k: string) => k.startsWith('out-'))) {
              const entry = createHistoryEntryFromPortKeyedOutputValue({
                id: `node-${nodeId}-${runTimestamp}`,
                timestamp: runTimestamp,
                output_value: outputValue as Record<string, any>,
                execution_time: executionTimeByNodeId[nodeId] ?? node.execution_time,
                params: node?.data ?? {}
              });
              const prevList = normalizeHistoryEntries(mergedHistory[nodeId] ?? []);
              mergedHistory[nodeId] = [entry, ...prevList].slice(0, MAX_NODE_HISTORY);
            } else if (outputValue != null) {
              const tool = TOOLS.find(t => t.id === node.tool_id);
              const portId = node ? (INPUT_PORT_IDS[node.tool_id] ?? tool?.outputs?.[0]?.id) : undefined;
              const entry = createHistoryEntryFromValue({
                id: `node-${nodeId}-${runTimestamp}`,
                timestamp: runTimestamp,
                value: outputValue,
                execution_time: executionTimeByNodeId[nodeId] ?? node.execution_time,
                params: node?.data ?? {},
                portId
              });
              if (entry) {
                const prevList = normalizeHistoryEntries(mergedHistory[nodeId] ?? []);
                mergedHistory[nodeId] = [entry, ...prevList].slice(0, MAX_NODE_HISTORY);
              }
            }
          }
          // 输入节点不进入 executedInSession，但本次运行若被处理并写入了 sessionOutputs，需在合并历史时补一条，否则只有第一次运行会通过 setWorkflow 写入历史，持久化时可能读到旧 ref 而丢条
          const sameOutputValue = (a: any, b: any): boolean => {
            if (a == null || b == null) return a === b;
            const firstFileId = (x: any) => (x?.file_id) ?? (typeof x === 'object' && x && Object.keys(x).some(k => k.startsWith('out-')) ? (Object.values(x).find((v: any) => v?.file_id) as { file_id?: string } | undefined)?.file_id : undefined);
            const fa = firstFileId(a);
            const fb = firstFileId(b);
            if (fa != null && fb != null) return fa === fb;
            return JSON.stringify(a) === JSON.stringify(b);
          };
          for (const node of mergedNodes) {
            const tool = TOOLS.find(t => t.id === node.tool_id);
            if (!tool || tool.category !== 'Input') continue;
            if (sessionOutputs[node.id] === undefined) continue;
            if (executedInSession.has(node.id)) continue;
            const outputValue = node.output_value;
            if (outputValue == null) continue;
            const prevList = normalizeHistoryEntries(mergedHistory[node.id] ?? []);
            const head = prevList[0];
            if (head?.output_value != null && sameOutputValue(outputValue, head.output_value)) continue; // 已有同一次运行的条目，避免重复
            const portId = INPUT_PORT_IDS[node.tool_id] ?? tool?.outputs?.[0]?.id;
            if (typeof outputValue === 'object' && !Array.isArray(outputValue) && Object.keys(outputValue).some((k: string) => k.startsWith('out-'))) {
              const entry = createHistoryEntryFromPortKeyedOutputValue({
                id: `node-${node.id}-${runTimestamp}`,
                timestamp: runTimestamp,
                output_value: outputValue as Record<string, any>,
                execution_time: 0,
                params: node?.data ?? {}
              });
              mergedHistory[node.id] = [entry, ...prevList].slice(0, MAX_NODE_HISTORY);
            } else {
              const entry = createHistoryEntryFromValue({
                id: `node-${node.id}-${runTimestamp}`,
                timestamp: runTimestamp,
                value: outputValue,
                execution_time: 0,
                params: node?.data ?? {},
                portId
              });
              if (entry) mergedHistory[node.id] = [entry, ...prevList].slice(0, MAX_NODE_HISTORY);
            }
          }
          // 输入/节点输出已通过 /save 写入后端，此处只同步前端 state，不再发 POST /update，避免覆盖后端多端口等数据
          setWorkflow(prev => prev ? { ...prev, nodes: mergedNodes, nodeOutputHistory: mergedHistory } : null);
        } catch (err) {
          const errorMsg = err instanceof Error ? err.message : String(err);
          console.error('[WorkflowExecution] Error merging state after run:', err);
          setGlobalError({ message: 'Failed to merge execution state', details: errorMsg });
        }
      } else if (!isStandalone() && workflow?.id && !hasValidDbId) {
       console.warn(`[WorkflowExecution] Workflow ID ${wfId} is not a valid database ID, skipping save`);
      }

      // Append per-node history (no run); keep legacy history empty for compat
      setWorkflow(prev => prev ? ({
        ...prev,
        nodeOutputHistory: { ...prev.nodeOutputHistory, ...nodeHistoryUpdates }
      }) : null);

      // Save execution state (nodes with status, output_value) to database or local
      const workflowId = wfId;
      const shouldSave = workflowId && (
        workflowId.startsWith('workflow-') ||
        workflowId.match(/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i) ||
        workflowId.startsWith('preset-')
      );

      if (shouldSave) {
        // When shouldSaveOutputs, persist is done once inside savePromises.then (single PUT). Otherwise defer one persist (no base64).
        // 在 setTimeout 内用 workflowRef.current 合并出要持久化的 state，避免后完成的 run 覆盖后加入的节点或其它 run 的结果
        if (!shouldSaveOutputs) {
          setTimeout(() => {
            const latest = workflowRef.current;
            const baseNodes = latest?.nodes ?? workflow.nodes;
            const nodesToSave = baseNodes.map((node: any) => {
              const tool = TOOLS.find(t => t.id === node.tool_id);
              if (tool?.category === 'Input' && sessionOutputs[node.id] != null) {
                return { ...node, output_value: sessionOutputs[node.id] };
              }
              const wasExecuted = sessionOutputs[node.id] !== undefined;
              if (wasExecuted) {
                const updatedNode = workflow.nodes.find((n: any) => n.id === node.id);
                return {
                  id: node.id,
                  tool_id: node.tool_id,
                  x: node.x,
                  y: node.y,
                  status: updatedNode?.status ?? node.status,
                  data: updatedNode?.data ?? node.data,
                  error: updatedNode?.error,
                  execution_time: updatedNode?.execution_time,
                  output_value: updatedNode?.output_value ?? node.output_value ?? sessionOutputs[node.id]
                };
              }
              return { ...node };
            });
            const nextNodeOutputHistory = {
              ...(latest?.nodeOutputHistory ?? workflow.nodeOutputHistory ?? {}),
              ...nodeHistoryUpdates
            };
            doPersistExecutionState(workflowId, latest ?? workflow, nodesToSave, nextNodeOutputHistory);
          }, 100);
        }
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
        const pid = (j as any).pollIntervalId;
        if (pid) {
          clearInterval(pid);
          (j as any).pollIntervalId = null;
        }
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
        nodeIdToIsCloud: new Map<string, boolean>(),
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

  /** Enqueue a run (full workflow or single/from-node); up to 3 jobs run at once. 若 isDirty 或为预设则先保存/实体化再执行。 */
  const runWorkflow = useCallback(async (startNodeId?: string, onlyOne?: boolean) => {
    if (!workflow) return;
    const isPreset = workflow.id?.startsWith('preset-');
    if ((workflow.isDirty || isPreset) && saveWorkflowBeforeRun) {
      try {
        const savedId = await saveWorkflowBeforeRun(workflow);
        setWorkflow(prev => prev ? ({ ...prev, isDirty: false, ...(savedId && savedId !== prev.id ? { id: savedId } : {}) }) : null);
        if (savedId && savedId !== workflow.id) {
          effectiveWorkflowIdRef.current = savedId;
        }
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        setGlobalError({ message: '保存工作流失败，无法执行', details: msg });
        return;
      }
    }
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
  }, [workflow, setWorkflow, processQueue, getAffectedNodeIds, refreshPendingNodeIds, saveWorkflowBeforeRun, setGlobalError]);

  /** Cancel the run for a given node: if queued, mark job cancelled; if running, send cancel to backend and abort. */
  const cancelNodeRun = useCallback((nodeId: string) => {
    const queue = executionQueueRef.current;
    const queuedIndex = queue.findIndex(j => !j.cancelled && j.affectedNodeIds.has(nodeId));
    if (queuedIndex !== -1) {
      queue[queuedIndex].cancelled = true;
      refreshPendingNodeIds();
      return;
    }
    const cloudUrl = (process.env.LIGHTX2V_CLOUD_URL || 'https://x2v.light-ai.top').trim();
    const cloudToken = (process.env.LIGHTX2V_CLOUD_TOKEN || '').trim();
    for (const [jid, jobInfo] of runningJobsByJobIdRef.current) {
      if (jobInfo.affectedNodeIds.has(nodeId)) {
        jobInfo.taskIdsByNodeId.forEach((taskId, nid) => {
          const isCloud = jobInfo.nodeIdToIsCloud.get(nid) ?? false;
          (async () => {
            try {
              if (isStandalone() || isCloud) {
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
    // Build taskId -> nodeId and taskId -> isCloud before clearing jobs
    const taskIds = Array.from(runningTaskIdsRef.current.values());
    const taskIdToNodeId = new Map<string, string>();
    runningTaskIdsRef.current.forEach((tid, nid) => taskIdToNodeId.set(tid, nid));
    const taskIdToIsCloud = new Map<string, boolean>();
    runningJobsByJobIdRef.current.forEach((j) => {
      j.taskIdsByNodeId.forEach((tid, nid) => {
        taskIdToIsCloud.set(tid, j.nodeIdToIsCloud.get(nid) ?? false);
      });
    });
    runningJobsByJobIdRef.current.clear();
    setPendingRunNodeIds([]);
    console.log('[WorkflowExecution] Stopping workflow...');

    console.log(`[WorkflowExecution] Found ${taskIds.length} running tasks to cancel:`, taskIds);

    if (taskIds.length === 0) {
      console.warn('[WorkflowExecution] No running tasks found in runningTaskIdsRef');
      // Still try to abort ongoing requests and stop workflow
    }

    const cloudUrlStop = (process.env.LIGHTX2V_CLOUD_URL || 'https://x2v.light-ai.top').trim();
    const cloudTokenStop = (process.env.LIGHTX2V_CLOUD_TOKEN || '').trim();
    const cancelPromises = taskIds.map(async (taskId) => {
      try {
        console.log(`[WorkflowExecution] Sending cancel request for task ${taskId}...`);
        let response: Response;
        if (isStandalone() || (taskIdToIsCloud.get(taskId) ?? false)) {
          response = await lightX2VCancelTask(cloudUrlStop, cloudTokenStop, String(taskId));
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

    const cancelledNodeIds = new Set<string>();
    const queryPromises = taskIds.map(async (taskId) => {
      try {
        let status: string;
        if (isStandalone() || (taskIdToIsCloud.get(taskId) ?? false)) {
          const info = await lightX2VTaskQuery(cloudUrlStop, cloudTokenStop, String(taskId));
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
