/**
 * 工作流文件管理工具
 * 使用新的统一接口（基于 data_store.outputs）
 *
 * 仅前端（standalone）时 IndexedDB 用法：
 * - 写入：uploadNodeInputFile(文件上传) 或 persistDataUrlToLocal(data: URL) → 存 Blob，得到 local://key
 * - 读取：getLocalFileDataUrl(local://key) → 得到 data URL，用于 <img>/预览/执行入参
 * - 列表缩略图：WorkflowCard 里 previewImage 为 local:// 时用 getLocalFileDataUrl 异步解析后显示
 */

import { apiRequest, getAccessToken } from './apiClient';
import { isStandalone } from '../config/runtimeMode';

const LOCAL_FILES_DB = 'canvas_local_files';
const LOCAL_FILES_STORE = 'files';

function openLocalDB(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(LOCAL_FILES_DB, 1);
    req.onerror = () => reject(req.error);
    req.onsuccess = () => resolve(req.result);
    req.onupgradeneeded = (e) => {
      const db = (e.target as IDBOpenDBRequest).result;
      if (!db.objectStoreNames.contains(LOCAL_FILES_STORE)) {
        db.createObjectStore(LOCAL_FILES_STORE);
      }
    };
  });
}

/**
 * 纯前端模式：从 IndexedDB 读取 local://key 对应的文件，返回 data URL
 */
export async function getLocalFileDataUrl(key: string): Promise<string | null> {
  if (!key) return null;
  const rawKey = key.startsWith('local://') ? key.slice(8) : key;
  try {
    const db = await openLocalDB();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(LOCAL_FILES_STORE, 'readonly');
      const store = tx.objectStore(LOCAL_FILES_STORE);
      const req = store.get(rawKey);
      req.onerror = () => { db.close(); reject(req.error); };
      req.onsuccess = () => {
        db.close();
        const blob = req.result as Blob | undefined;
        if (!blob) {
          resolve(null);
          return;
        }
        const reader = new FileReader();
        reader.onloadend = () => resolve(reader.result as string);
        reader.onerror = () => resolve(null);
        reader.readAsDataURL(blob);
      };
    });
  } catch (e) {
    console.warn('[WorkflowFileManager] getLocalFileDataUrl failed:', rawKey, e);
    return null;
  }
}

/**
 * 纯前端模式：将 Blob 存入 IndexedDB，返回 local://key
 */
export async function saveLocalFile(key: string, blob: Blob): Promise<void> {
  try {
    const db = await openLocalDB();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(LOCAL_FILES_STORE, 'readwrite');
      const store = tx.objectStore(LOCAL_FILES_STORE);
      store.put(blob, key);
      tx.oncomplete = () => { db.close(); resolve(); };
      tx.onerror = () => { db.close(); reject(tx.error); };
    });
  } catch (e) {
    console.error('[WorkflowFileManager] saveLocalFile failed:', key, e);
    throw e;
  }
}

/**
 * 从 base64 data URL 提取 MIME 类型和扩展名
 */
function getMimeTypeAndExt(dataUrl: string): { mimeType: string; ext: string } {
  const match = dataUrl.match(/data:([^;]+);base64,/);
  if (match) {
    const mimeType = match[1];
    const extMap: Record<string, string> = {
      'image/png': '.png',
      'image/jpeg': '.jpg',
      'image/jpg': '.jpg',
      'image/gif': '.gif',
      'image/webp': '.webp',
      'video/mp4': '.mp4',
      'video/webm': '.webm',
      'audio/mpeg': '.mp3',
      'audio/wav': '.wav',
      'audio/ogg': '.ogg',
      'text/plain': '.txt',
      'application/json': '.json'
    };
    const ext = extMap[mimeType] || '.bin';
    return { mimeType, ext };
  }
  return { mimeType: 'application/octet-stream', ext: '.bin' };
}

/**
 * 将 base64 data URL 转换为 Blob
 */
function dataURLToBlob(dataUrl: string): Blob {
  const arr = dataUrl.split(',');
  const mimeMatch = arr[0].match(/:(.*?);/);
  const mimeType = mimeMatch ? mimeMatch[1] : 'application/octet-stream';
  const bstr = atob(arr[1]);
  let n = bstr.length;
  const u8arr = new Uint8Array(n);
  while (n--) {
    u8arr[n] = bstr.charCodeAt(n);
  }
  return new Blob([u8arr], { type: mimeType });
}

/**
 * 仅前端模式：将 data: URL 存入 IndexedDB，返回 local://key，供列表缩略图等使用。
 * 有后端时直接返回原 dataUrl（不写入 IndexedDB）。
 */
export async function persistDataUrlToLocal(dataUrl: string, keyPrefix: string): Promise<string> {
  if (!dataUrl || typeof dataUrl !== 'string' || !dataUrl.startsWith('data:')) return dataUrl;
  if (!isStandalone()) return dataUrl;
  try {
    const blob = dataURLToBlob(dataUrl);
    const key = `${keyPrefix}_${crypto.randomUUID()}`;
    await saveLocalFile(key, blob);
    return `local://${key}`;
  } catch (e) {
    console.warn('[WorkflowFileManager] persistDataUrlToLocal failed:', e);
    return dataUrl;
  }
}

export type SaveNodeOutputResult = { file_id?: string; file_url?: string; url?: string; ext?: string } | null;

const JSON_PORT_ID = '__json__';

/**
 * 以节点为单位保存多端口输出。对于对象型多端口（如文本生成 customOutputs），
 * 将整个 JSON 存为一条 kind:json 历史记录，读取时按字段提取。
 */
export async function saveNodeOutputs(
  workflowId: string,
  nodeId: string,
  outputs: Record<string, string | object>,
  runId?: string
): Promise<Record<string, SaveNodeOutputResult> | null> {
  if (isStandalone()) return null;
  if (!outputs || Object.keys(outputs).length === 0) return null;
  try {
    // 对象型多端口（如文本生成 customOutputs）：整存整取，存成一条 kind:json 记录
    // 排除 data URL、Blob、大对象，仅对纯文本/小 JSON 使用
    const keys = Object.keys(outputs);
    const isObjectOutput =
      keys.length >= 1 &&
      keys.every(k => {
        const v = outputs[k];
        if (v == null || v instanceof Blob) return false;
        if (typeof v === 'string') return !v.startsWith('data:');
        return typeof v === 'object' && !Array.isArray(v);
      });
    const payload: Record<string, string | object> = isObjectOutput
      ? { [JSON_PORT_ID]: outputs }
      : {};

    if (!isObjectOutput) {
      for (const [portId, value] of Object.entries(outputs)) {
        if (value === undefined || value === null) continue;
        if (typeof value === 'string' && value.length === 0) continue;
        if (typeof value === 'object' && !Array.isArray(value) && Object.keys(value as object).length === 0) continue;
        let outputData: string | object;
        if (typeof value === 'string') {
          outputData = value;
        } else if (value instanceof Blob) {
          outputData = await new Promise<string>((resolve, reject) => {
            const reader = new FileReader();
            reader.onloadend = () => resolve(reader.result as string);
            reader.onerror = reject;
            reader.readAsDataURL(value);
          });
        } else {
          outputData = value as object;
        }
        payload[portId] = outputData;
      }
    }
    if (Object.keys(payload).length === 0) return null;

    const response = await apiRequest(`/api/v1/workflow/${workflowId}/node/${nodeId}/outputs/save`, {
      method: 'POST',
      body: JSON.stringify({ outputs: payload, run_id: runId })
    });

    if (!response.ok) {
      const contentType = response.headers.get('content-type') || '';
      let errorMessage = `Failed to save node outputs: ${response.status} ${response.statusText}`;
      if (contentType.includes('application/json')) {
        try {
          const errorData = await response.json();
          errorMessage = errorData.message || errorMessage;
        } catch (_e) { /* ignore */ }
      }
      throw new Error(errorMessage);
    }

    const result = (await response.json()) as { results?: Record<string, { file_id?: string; file_url?: string; url?: string; ext?: string }> };
    const results = result.results ?? {};
    const out: Record<string, SaveNodeOutputResult> = {};
    for (const [portId, r] of Object.entries(results)) {
      out[portId] = r ? { file_id: r.file_id, file_url: r.file_url ?? r.url, url: r.url ?? r.file_url, ext: r.ext } : null;
    }
    return out;
  } catch (error) {
    const msg = error instanceof Error ? error.message : String(error);
    console.error(`[WorkflowFileManager] Error saving node outputs for ${nodeId}:`, msg);
    throw error;
  }
}

/**
 * 获取节点输出数据（当前输出）。后端按节点返回所有端口，此处取指定 portId。
 */
export async function getNodeOutputData(
  workflowId: string,
  nodeId: string,
  portId: string
): Promise<any | null> {
  if (isStandalone()) return null;
  try {
    const response = await apiRequest(`/api/v1/workflow/${workflowId}/node/${nodeId}/output`);

    if (!response.ok) {
      const contentType = response.headers.get('content-type') || '';
      let errorMessage = `Failed to get node output: ${response.status} ${response.statusText}`;
      if (contentType.includes('application/json')) {
        try {
          const errorData = await response.json();
          errorMessage = errorData.message || errorMessage;
        } catch (e) {
          // ignore
        }
      } else {
        try {
          const text = await response.text();
          if (text.includes('<!doctype') || text.includes('<html')) {
            errorMessage = `Server returned HTML error page (${response.status}). The node output may not exist yet.`;
          } else {
            errorMessage = text.substring(0, 200);
          }
        } catch (e) {
          // ignore
        }
      }
      console.error(`[WorkflowFileManager] Error getting node output data for ${nodeId}/${portId}:`, errorMessage);
      if (response.status === 404) return null;
      throw new Error(errorMessage);
    }

    const contentType = response.headers.get('content-type') || '';
    if (!contentType.includes('application/json')) {
      const text = await response.text();
      throw new Error(`Expected JSON but got ${contentType}. Response: ${text.substring(0, 200)}`);
    }

    const result = await response.json();
    const portOutput = result.outputs?.[portId];
    if (!portOutput?.data) return null;

    const dataRef = portOutput.data;
    const resultUrl = portOutput.url;

    if (dataRef.data_type === 'url') {
      return resultUrl || dataRef.url_value;
    }
    if (dataRef.data_type === 'file' && dataRef.file_path) {
      const filePath = dataRef.file_path;
      const filePaths = Array.isArray(filePath) ? filePath : [filePath];
      const filePromises = filePaths.map(async (path: string) => {
        const match = path.match(/workflows\/[^_]+_(.+)\.(.+)$/);
        if (match) {
          const fileId = match[1];
          return await getWorkflowFileByFileId(workflowId, fileId);
        }
        return null;
      });
      const results = await Promise.all(filePromises);
      const validResults = results.filter(r => r !== null);
      return validResults.length === 1 ? validResults[0] : validResults;
    }
    if (dataRef.data_type === 'text') return dataRef.text_value;
    if (dataRef.data_type === 'json') return dataRef.json_value;
    return dataRef;
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : String(error);
    console.error(`[WorkflowFileManager] Error getting node output data for ${nodeId}/${portId}:`, errorMessage);
    throw error;
  }
}

/**
 * 获取工作流对话历史（chat 已拆到独立存储）
 */
export async function getWorkflowChat(workflowId: string): Promise<{ messages: any[]; updated_at?: number } | null> {
  try {
    const response = await apiRequest(`/api/v1/workflow/${workflowId}/chat`);
    if (!response.ok) return null;
    const data = await response.json();
    return { messages: data.messages || [], updated_at: data.updated_at };
  } catch (error) {
    console.error('[WorkflowFileManager] Error getting workflow chat:', error);
    return null;
  }
}

/**
 * 替换工作流对话历史（PUT 全量）
 */
export async function putWorkflowChat(workflowId: string, messages: any[]): Promise<boolean> {
  try {
    const response = await apiRequest(`/api/v1/workflow/${workflowId}/chat`, {
      method: 'PUT',
      body: JSON.stringify({ messages })
    });
    return response.ok;
  } catch (error) {
    console.error('[WorkflowFileManager] Error putting workflow chat:', error);
    return false;
  }
}

/**
 * 获取节点历史记录（按节点拉取，再按 portId 过滤）
 */
export async function getNodeOutputHistory(
  workflowId: string,
  nodeId: string,
  portId: string
): Promise<any[]> {
  try {
    const response = await apiRequest(`/api/v1/workflow/${workflowId}/node/${nodeId}/output/history`);
    if (!response.ok) return [];
    const result = await response.json();
    const list = result.history || [];
    const normalizedPort = portId || 'output';
    return list.filter((e: any) => (e?.metadata?.port_id ?? 'output') === normalizedPort);
  } catch (error) {
    console.error('[WorkflowFileManager] Error getting node output history:', error);
    return [];
  }
}

/**
 * 重用历史记录
 */
export async function reuseNodeOutputHistory(
  workflowId: string,
  nodeId: string,
  portId: string,
  historyIndex: number
): Promise<any | null> {
  try {
    const response = await apiRequest(`/api/v1/workflow/${workflowId}/node/${nodeId}/output/reuse`, {
      method: 'POST',
      body: JSON.stringify({ port_id: portId, history_index: historyIndex })
    });
    if (!response.ok) return null;
    const result = await response.json();
    return result?.data ?? null;
  } catch (error) {
    console.error('[WorkflowFileManager] Error reusing node output history:', error);
    return null;
  }
}

/**
 * 将输入节点的文件通过 node output/save 存到数据库，与节点生成文件时的保存逻辑一致。
 * 返回 file 类型引用 { type: 'file', file_id, file_url, ext }，用于写入 node.outputValue。
 */
export async function saveInputFileViaOutputSave(
  workflowId: string,
  nodeId: string,
  portId: string,
  fileOrDataUrl: File | string
): Promise<{ type: 'file'; file_id: string; file_url: string; ext?: string } | null> {
  try {
    let dataUrl: string;
    let ext = '.bin';
    if (typeof fileOrDataUrl === 'string') {
      dataUrl = fileOrDataUrl;
      if (dataUrl.startsWith('data:')) {
        const header = dataUrl.split(',')[0] || '';
        const mime = header.split(':')[1]?.split(';')[0] || '';
        const extMap: Record<string, string> = {
          'image/png': '.png', 'image/jpeg': '.jpg', 'image/gif': '.gif', 'image/webp': '.webp',
          'video/mp4': '.mp4', 'video/webm': '.webm', 'audio/mpeg': '.mp3', 'audio/wav': '.wav', 'audio/ogg': '.ogg'
        };
        ext = extMap[mime] || '.bin';
      }
    } else {
      dataUrl = await new Promise<string>((resolve, reject) => {
        const reader = new FileReader();
        reader.onloadend = () => resolve(reader.result as string);
        reader.onerror = reject;
        reader.readAsDataURL(fileOrDataUrl);
      });
      const name = fileOrDataUrl.name || '';
      if (name.includes('.')) ext = '.' + name.split('.').pop()!.toLowerCase();
    }

    if (isStandalone()) {
      const key = `localFile_${workflowId}_${nodeId}_${portId}_${Date.now()}_${crypto.randomUUID()}`;
      const file = typeof fileOrDataUrl === 'string'
        ? await (async () => {
            const res = await fetch(fileOrDataUrl);
            const blob = await res.blob();
            return new File([blob], 'file' + ext, { type: blob.type || 'application/octet-stream' });
          })()
        : fileOrDataUrl;
      await saveLocalFile(key, file);
      const localRef = `local://${key}`;
      return { type: 'file', file_id: key, file_url: localRef, ext };
    }

    const result = await saveNodeOutputs(workflowId, nodeId, { [portId]: dataUrl });
    const r = result?.[portId];
    if (!r?.file_id) return null;
    const file_url = r.file_url ?? r.url ?? `/api/v1/workflow/${workflowId}/file/${r.file_id}`;
    return { type: 'file', file_id: r.file_id, file_url, ext: r.ext };
  } catch (error) {
    const msg = error instanceof Error ? error.message : String(error);
    console.error(`[WorkflowFileManager] saveInputFileViaOutputSave failed:`, msg);
    throw error;
  }
}

/**
 * 直接上传文件到工作流节点输出（用于输入节点）
 * 有后端时通过 output/save 保存；纯前端时保存到 IndexedDB。
 * @deprecated 新逻辑请用 saveInputFileViaOutputSave，并写入 node.outputValue。
 */
export async function uploadNodeInputFile(
  workflowId: string,
  nodeId: string,
  portId: string,
  file: File,
  index: number = 0
): Promise<{ file_id: string; file_path: string; file_url: string } | null> {
  const ref = await saveInputFileViaOutputSave(workflowId, nodeId, portId, file);
  if (!ref) return null;
  return { file_id: ref.file_id, file_path: ref.file_url, file_url: ref.file_url };
}

/**
 * 判断是否为“需上传的本地资源 URL”（如 /assets/girl.png），
 * 即不以 workflow/input、task/、http 等已保存路径为前缀。
 */
export function isLocalAssetUrlToUpload(url: string): boolean {
  if (typeof url !== 'string' || !url) return false;
  const u = url.replace(/^\.\//, '/');
  if (!u.startsWith('/assets/')) return false;
  if (u.startsWith('/assets/workflow/input') || u.startsWith('/assets/task/')) return false;
  if (u.startsWith('http://') || u.startsWith('https://')) return false;
  return true;
}

/**
 * 将本地资源 URL（如 /assets/girl.png）拉取为 File 并上传到节点输出，
 * 返回后端 file 引用，便于数据库以 file 类型存储而非 base64。
 * 有后端时使用；standalone 下不调用。
 */
export async function uploadLocalUrlAsNodeOutput(
  workflowId: string,
  nodeId: string,
  portId: string,
  localUrl: string,
  index: number = 0
): Promise<{ type: 'file'; file_id: string; file_url: string; ext?: string } | null> {
  try {
    const path = localUrl.startsWith('./') ? localUrl.slice(1) : localUrl;
    const urlToFetch = path.startsWith('http') ? path : (typeof window !== 'undefined' ? window.location.origin : '') + path;
    const res = await fetch(urlToFetch);
    if (!res.ok) return null;
    const blob = await res.blob();
    const name = path.split('/').pop() || 'file';
    const file = new File([blob], name, { type: blob.type || 'application/octet-stream' });
    const result = await uploadNodeInputFile(workflowId, nodeId, portId, file, index);
    if (!result) return null;
    const ext = name.includes('.') ? '.' + name.split('.').pop()!.toLowerCase() : undefined;
    return {
      type: 'file',
      file_id: result.file_id,
      file_url: result.file_url,
      ext: ext || undefined
    };
  } catch (e) {
    console.error('[WorkflowFileManager] uploadLocalUrlAsNodeOutput failed:', localUrl, e);
    return null;
  }
}

/**
 * 根据 file_id 获取工作流文件（新格式）
 * 纯前端模式：fileId 为 local://key 时从 IndexedDB 解析
 */
export async function getWorkflowFileByFileId(
  workflowId: string,
  fileId: string
): Promise<string | null> {
  try {
    if (fileId.startsWith('local://')) {
      return getLocalFileDataUrl(fileId);
    }
    if (isStandalone()) {
      return null;
    }
    // 与上传接口一致：显式带 Authorization，保证与 POST /output/upload 相同的 token 方式
    const token = getAccessToken();
    const url = `/api/v1/workflow/${workflowId}/file/${fileId}`;
    const response = await fetch(url, {
      method: 'GET',
      headers: token ? { Authorization: `Bearer ${token}` } : {},
    });

    if (!response.ok) {
      console.error(`[WorkflowFileManager] Failed to fetch file: ${response.status}`);
      return null;
    }

    const blob = await response.blob();
    return await new Promise<string>((resolve, reject) => {
      const reader = new FileReader();
      reader.onloadend = () => resolve(reader.result as string);
      reader.onerror = reject;
      reader.readAsDataURL(blob);
    });
  } catch (error) {
    console.error('[WorkflowFileManager] Error getting workflow file by file_id:', error);
    return null;
  }
}

/**
 * 获取工作流文件 URL（用于直接访问，如 <img> 标签）
 * 新格式：使用 file_id
 */
export function getWorkflowFileUrl(
  workflowId: string,
  fileId: string
): string {
  const token = localStorage.getItem('accessToken');
  const tokenParam = token ? `?token=${encodeURIComponent(token)}` : '';

  return `/api/v1/workflow/${workflowId}/file/${fileId}${tokenParam}`;
}
