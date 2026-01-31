/**
 * 工作流文件管理工具
 * 使用新的统一接口（基于 data_store.outputs）
 *
 * 仅前端（standalone）时 IndexedDB 用法：
 * - 写入：uploadNodeInputFile(文件上传) 或 persistDataUrlToLocal(data: URL) → 存 Blob，得到 local://key
 * - 读取：getLocalFileDataUrl(local://key) → 得到 data URL，用于 <img>/预览/执行入参
 * - 列表缩略图：WorkflowCard 里 previewImage 为 local:// 时用 getLocalFileDataUrl 异步解析后显示
 */

import { apiRequest } from './apiClient';
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

/**
 * 保存节点输出数据（统一接口，替代旧的 input/intermediate/output 区分）
 */
export async function saveNodeOutputData(
  workflowId: string,
  nodeId: string,
  portId: string,
  data: string | Blob | object,
  runId?: string
): Promise<{ file_id?: string; data_id: string; file_url?: string; url?: string } | null> {
  if (isStandalone()) return null;
  try {
    let outputData: string | object;
    let fileInfo: any = null;

    if (typeof data === 'string') {
      if (data.startsWith('data:')) {
        // Base64 data URL
        outputData = data;
        // 后端会自动从 data URL 中提取 file_info
      } else if (data.startsWith('http://') || data.startsWith('https://')) {
        // CDN URL or HTTP/HTTPS URL - save as URL type
        outputData = data;
        console.log(`[WorkflowFileManager] Saving URL output: ${data.substring(0, 100)}...`);
      } else {
        // 纯文本
        outputData = data;
      }
    } else if (data instanceof Blob) {
      // Convert Blob to data URL
      outputData = await new Promise<string>((resolve, reject) => {
        const reader = new FileReader();
        reader.onloadend = () => resolve(reader.result as string);
        reader.onerror = reject;
        reader.readAsDataURL(data);
      });
    } else {
      // JSON 对象
      outputData = data;
    }

    const response = await apiRequest(`/api/v1/workflow/${workflowId}/node/${nodeId}/output/${portId}/save`, {
      method: 'POST',
      body: JSON.stringify({
        output_data: outputData,
        file_info: fileInfo,  // 可选，后端会自动生成
        run_id: runId  // 可选，用于关联工作流历史记录
      })
    });

    if (!response.ok) {
      // 检查响应内容类型
      const contentType = response.headers.get('content-type') || '';
      let errorMessage = `Failed to save node output: ${response.status} ${response.statusText}`;

      if (contentType.includes('application/json')) {
        try {
          const errorData = await response.json();
          errorMessage = errorData.message || errorMessage;
        } catch (e) {
          // 如果 JSON 解析失败，使用默认错误消息
        }
      } else {
        // 如果返回的是 HTML（错误页面），尝试读取文本
        try {
          const text = await response.text();
          if (text.includes('<!doctype') || text.includes('<html')) {
            errorMessage = `Server returned HTML error page (${response.status}). Check server logs for details.`;
          } else {
            errorMessage = text.substring(0, 200); // 限制长度
          }
        } catch (e) {
          // 如果读取失败，使用默认错误消息
        }
      }

      console.error(`[WorkflowFileManager] Error saving node output data for ${nodeId}/${portId}:`, errorMessage);

      // 抛出错误，让调用者可以处理（如显示错误提示）
      throw new Error(errorMessage);
    }

    // 检查响应内容类型
    const contentType = response.headers.get('content-type') || '';
    if (!contentType.includes('application/json')) {
      const text = await response.text();
      throw new Error(`Expected JSON but got ${contentType}. Response: ${text.substring(0, 200)}`);
    }

    const result = await response.json();
    // 返回保存成功的信息（后端会返回 data_id，如果是文件还会返回 file_id / file_url）
    return {
      data_id: result.data_id,
      file_id: result.file_id,
      file_url: result.file_url,
      url: result.url
    };
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : String(error);
    console.error(`[WorkflowFileManager] Error saving node output data for ${nodeId}/${portId}:`, errorMessage);

    // 重新抛出错误，让调用者可以处理（如显示错误提示）
    throw error;
  }
}

/**
 * 获取节点输出数据（当前输出）
 */
export async function getNodeOutputData(
  workflowId: string,
  nodeId: string,
  portId: string
): Promise<any | null> {
  if (isStandalone()) return null;
  try {
    const response = await apiRequest(`/api/v1/workflow/${workflowId}/node/${nodeId}/output/${portId}`);

    if (!response.ok) {
      // 检查响应内容类型
      const contentType = response.headers.get('content-type') || '';
      let errorMessage = `Failed to get node output: ${response.status} ${response.statusText}`;

      if (contentType.includes('application/json')) {
        try {
          const errorData = await response.json();
          errorMessage = errorData.message || errorMessage;
        } catch (e) {
          // 如果 JSON 解析失败，使用默认错误消息
        }
      } else {
        // 如果返回的是 HTML（错误页面），尝试读取文本
        try {
          const text = await response.text();
          if (text.includes('<!doctype') || text.includes('<html')) {
            errorMessage = `Server returned HTML error page (${response.status}). The node output may not exist yet.`;
          } else {
            errorMessage = text.substring(0, 200); // 限制长度
          }
        } catch (e) {
          // 如果读取失败，使用默认错误消息
        }
      }

      console.error(`[WorkflowFileManager] Error getting node output data for ${nodeId}/${portId}:`, errorMessage);

      // 如果是 404，可能是节点输出还不存在，这是正常的，不抛出错误
      if (response.status === 404) {
        return null;
      }

      // 对于其他错误，抛出异常以便上层处理
      throw new Error(errorMessage);
    }

    // 检查响应内容类型
    const contentType = response.headers.get('content-type') || '';
    if (!contentType.includes('application/json')) {
      const text = await response.text();
      throw new Error(`Expected JSON but got ${contentType}. Response: ${text.substring(0, 200)}`);
    }

    const result = await response.json();
    const dataRef = result.data;

    if (!dataRef) {
      return null;
    }

    // Handle different data types
    if (dataRef.data_type === 'url') {
      // Task result URL (local path or CDN URL) - return directly
      return result.url || dataRef.url_value;
    } else if (dataRef.data_type === 'file' && dataRef.file_path) {
      // File type - use file_path directly
      // file_path 现在是数组格式（即使是单个文件）
      const filePath = dataRef.file_path;

      // 确保是数组格式
      const filePaths = Array.isArray(filePath) ? filePath : [filePath];

      // 返回数组格式的 data URLs
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

      // 如果只有一个文件，返回单个值（兼容旧代码）；否则返回数组
      return validResults.length === 1 ? validResults[0] : validResults;
    } else if (dataRef.data_type === 'text') {
      // Text output
      return dataRef.text_value;
    } else if (dataRef.data_type === 'json') {
      // JSON output
      return dataRef.json_value;
    }

    return dataRef;
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : String(error);
    console.error(`[WorkflowFileManager] Error getting node output data for ${nodeId}/${portId}:`, errorMessage);

    // 重新抛出错误，让调用者可以处理（如显示错误提示）
    throw error;
  }
}

/**
 * 获取节点历史记录
 */
export async function getNodeOutputHistory(
  workflowId: string,
  nodeId: string,
  portId: string
): Promise<any[]> {
  try {
    const response = await apiRequest(`/api/v1/workflow/${workflowId}/node/${nodeId}/output/${portId}/history`);

    if (response.ok) {
      const result = await response.json();
      return result.history || [];
    }

    return [];
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
    const response = await apiRequest(`/api/v1/workflow/${workflowId}/node/${nodeId}/output/${portId}/reuse`, {
      method: 'POST',
      body: JSON.stringify({ history_index: historyIndex })
    });

    if (response.ok) {
      const result = await response.json();
      return result.data;
    }

    return null;
  } catch (error) {
    console.error('[WorkflowFileManager] Error reusing node output history:', error);
    return null;
  }
}

/**
 * 直接上传文件到工作流节点输出（用于输入节点）
 * 有后端时保存到服务器；纯前端时保存到 IndexedDB，返回 local://key
 */
export async function uploadNodeInputFile(
  workflowId: string,
  nodeId: string,
  portId: string,
  file: File,
  index: number = 0
): Promise<{ file_id: string; file_path: string; file_url: string } | null> {
  try {
    if (isStandalone()) {
      const key = `localFile_${workflowId}_${nodeId}_${portId}_${index}_${crypto.randomUUID()}`;
      await saveLocalFile(key, file);
      const localRef = `local://${key}`;
      return {
        file_id: key,
        file_path: localRef,
        file_url: localRef
      };
    }

    const formData = new FormData();
    formData.append('file', file);

    const token = localStorage.getItem('accessToken');
    const url = `/api/v1/workflow/${workflowId}/node/${nodeId}/output/${portId}/upload${token ? `?token=${encodeURIComponent(token)}` : ''}`;

    const response = await fetch(url, {
      method: 'POST',
      body: formData,
      headers: token ? {
        'Authorization': `Bearer ${token}`
      } : {}
    });

    if (!response.ok) {
      const contentType = response.headers.get('content-type') || '';
      let errorMessage = `Failed to upload file: ${response.status} ${response.statusText}`;

      if (contentType.includes('application/json')) {
        try {
          const errorData = await response.json();
          errorMessage = errorData.message || errorMessage;
        } catch (e) {
          // Ignore
        }
      }

      console.error(`[WorkflowFileManager] Error uploading file for ${nodeId}/${portId}:`, errorMessage);
      throw new Error(errorMessage);
    }

    const result = await response.json();
    return {
      file_id: result.file_id,
      file_path: result.file_path,
      file_url: result.file_url
    };
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : String(error);
    console.error(`[WorkflowFileManager] Error uploading file for ${nodeId}/${portId}:`, errorMessage);
    throw error;
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
    // Use apiRequest to ensure Authorization header is set
    const url = `/api/v1/workflow/${workflowId}/file/${fileId}`;

    // Get token for Authorization header
    const sharedStore = (window as any).__SHARED_STORE__;
    const token = sharedStore ? sharedStore.getState('token') : localStorage.getItem('accessToken');

    const headers: Record<string, string> = {};
    if (token) {
      headers['Authorization'] = `Bearer ${token}`;
    }

    const response = await fetch(url, {
      headers
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
