import { NodeHistoryEntry, NodeHistoryEntryKind, NodeHistoryValue } from '../../types';
import { isLightX2VResultRef, type LightX2VResultRef } from './resultRef';

const FILE_STRING_PREFIXES = ['http://', 'https://', './', '/assets/', 'local://', 'blob:'];

const cloneJson = (value: any) => {
  if (Array.isArray(value)) return value.map(cloneJson);
  if (value && typeof value === 'object') return Object.fromEntries(Object.entries(value).map(([k, v]) => [k, cloneJson(v)]));
  return value;
};

const looksLikeFileString = (str: string) => {
  if (str.startsWith('data:')) return true;
  return FILE_STRING_PREFIXES.some(prefix => str.startsWith(prefix));
};

const MIME_TO_EXT: Record<string, string> = {
  'image/png': '.png', 'image/jpeg': '.jpg', 'image/jpg': '.jpg', 'image/gif': '.gif', 'image/webp': '.webp',
  'video/mp4': '.mp4', 'video/webm': '.webm', 'audio/mpeg': '.mp3', 'audio/wav': '.wav', 'audio/ogg': '.ogg',
  'text/plain': '.txt', 'application/json': '.json'
};

function inferExtFromDataUrl(dataUrl: string): string {
  const m = dataUrl.match(/data:([^;]+);/);
  const mime = m ? m[1] : '';
  return MIME_TO_EXT[mime] || '.bin';
}

function inferExtFromUrl(url: string): string | undefined {
  const m = url.match(/\.([a-z0-9]+)(?:\?|$)/i);
  return m ? `.${m[1].toLowerCase()}` : undefined;
}

const toLightX2VValue = (ref: LightX2VResultRef): { kind: NodeHistoryEntryKind; value: NodeHistoryValue } => ({
  kind: 'lightx2v_result',
  value: {
    taskId: ref.task_id,
    outputName: ref.output_name || 'output',
    isCloud: !!ref.is_cloud
  }
});

type HistoryValueBuilderResult = { kind: NodeHistoryEntryKind; value: NodeHistoryValue };

export const buildHistoryValue = (value: any): HistoryValueBuilderResult | null => {
  if (value == null) return null;

  if (isLightX2VResultRef(value)) {
    return toLightX2VValue(value);
  }

  if (typeof value === 'string') {
    if (value.startsWith('data:')) {
      return { kind: 'file', value: { dataUrl: value, ext: inferExtFromDataUrl(value) } };
    }
    if (looksLikeFileString(value)) {
      return { kind: 'file', value: { url: value, ext: inferExtFromUrl(value) } };
    }
    return { kind: 'text', value: { text: value } };
  }

  if (typeof value === 'number' || typeof value === 'boolean') {
    return { kind: 'text', value: { text: String(value) } };
  }

  if (Array.isArray(value)) {
    return { kind: 'json', value: { json: cloneJson(value) } };
  }

  if (typeof value === 'object') {
    if ('json_value' in value) {
      return buildHistoryValue((value as any).json_value);
    }
    if ('text_value' in value) {
      return { kind: 'text', value: { text: String((value as any).text_value ?? '') } };
    }
    if (value.__type === 'lightx2v_result') {
      return toLightX2VValue(value as LightX2VResultRef);
    }
    if (value.type === 'lightx2v_result') {
      return toLightX2VValue({
        __type: 'lightx2v_result',
        task_id: value.task_id,
        output_name: value.output_name,
        is_cloud: value.is_cloud
      });
    }
    if (value.type === 'text' && 'data' in value) {
      return { kind: 'text', value: { text: String(value.data ?? '') } };
    }
    if (value.type === 'json' && 'data' in value) {
      return { kind: 'json', value: { json: cloneJson(value.data) } };
    }
    if (value.type === 'data_url') {
      const dataUrl = typeof value._full_data === 'string' ? value._full_data : value.data;
      const ext = typeof dataUrl === 'string' && dataUrl.startsWith('data:') ? inferExtFromDataUrl(dataUrl) : undefined;
      return {
        kind: 'file',
        value: { dataUrl: typeof dataUrl === 'string' ? dataUrl : undefined, ext }
      };
    }
    if (value.type === 'url') {
      const url = value.data ?? value.url;
      const ext = typeof url === 'string' ? inferExtFromUrl(url) : value.ext;
      return {
        kind: 'file',
        value: {
          url: typeof url === 'string' ? url : undefined,
          ext: ext ?? value.ext
        }
      };
    }
    // Backend-stored file: persist fileId (and optional url, ext); backend resolves by fileId
    if (value.type === 'reference' || value.file_id) {
      return {
        kind: 'file',
        value: {
          fileId: value.file_id,
          url: value.file_url || value.url || undefined,
          ext: value.ext
        }
      };
    }
    if ('file_id' in value || 'file_url' in value) {
      return {
        kind: 'file',
        value: {
          fileId: value.file_id,
          url: value.file_url || undefined,
          ext: value.ext
        }
      };
    }

    return { kind: 'json', value: { json: cloneJson(value) } };
  }

  return null;
};

export const createHistoryEntryFromValue = (params: {
  id: string;
  timestamp: number;
  value: any;
  executionTime?: number;
  metadata?: Record<string, any>;
  valueOverride?: HistoryValueBuilderResult | null;
}): NodeHistoryEntry | null => {
  const { id, timestamp, value, executionTime, metadata, valueOverride } = params;
  const payload = valueOverride ?? buildHistoryValue(value);
  if (!payload) return null;

  const entry: NodeHistoryEntry = {
    id,
    timestamp,
    kind: payload.kind,
    value: payload.value,
    executionTime,
    metadata
  };
  return entry;
};

const normalizeValueForKind = (
  kind: NodeHistoryEntryKind,
  value: any,
  legacy?: any
): NodeHistoryValue => {
  switch (kind) {
    case 'text': {
      if (value && typeof value === 'object' && 'text' in value) {
        return { text: String(value.text ?? '') };
      }
      if (typeof value === 'string') return { text: value };
      if (legacy && typeof legacy.text === 'string') return { text: legacy.text };
      return { text: value != null ? String(value) : '' };
    }
    case 'json': {
      if (value && typeof value === 'object' && 'json' in value) {
        return { json: cloneJson(value.json) };
      }
      if (value !== undefined) return { json: cloneJson(value) };
      if (legacy && legacy.value !== undefined) return { json: cloneJson(legacy.value) };
      return { json: null };
    }
    case 'file': {
      const source = { ...(value || {}), ...(legacy || {}) };
      // 兼容旧数据：dataId 可当作 fileId 使用（历史格式迁移）
      const fileId = source.fileId ?? source.file_id ?? source.dataId ?? source.data_id;
      return {
        fileId: fileId || undefined,
        url: source.url ?? source.file_url,
        dataUrl: source.dataUrl,
        ext: source.ext
      };
    }
    case 'lightx2v_result': {
      const source = value || legacy || {};
      return {
        taskId: source.taskId ?? source.task_id ?? '',
        outputName: source.outputName ?? source.output_name ?? 'output',
        isCloud: !!(source.isCloud ?? source.is_cloud)
      };
    }
    default:
      return { json: cloneJson(value) };
  }
};

export const normalizeHistoryEntry = (raw: any): NodeHistoryEntry | null => {
  if (!raw) return null;
  if (typeof raw === 'object' && raw.kind) {
    let kind = raw.kind;
    let value = raw.value;
    // 兼容旧数据：kind 为 json 但 value.json 是 lightx2v_result 时，规范为 kind: lightx2v_result
    if (kind === 'json' && value && typeof value === 'object' && value.json && typeof value.json === 'object' && value.json.__type === 'lightx2v_result') {
      kind = 'lightx2v_result';
      value = {
        taskId: value.json.task_id ?? value.json.taskId ?? '',
        outputName: value.json.output_name ?? value.json.outputName ?? 'output',
        isCloud: !!(value.json.is_cloud ?? value.json.isCloud)
      };
    }
    return {
      id: raw.id || `entry-${Date.now()}`,
      timestamp: typeof raw.timestamp === 'number' ? raw.timestamp : Date.now(),
      executionTime: raw.executionTime,
      metadata: raw.metadata,
      kind,
      value: normalizeValueForKind(kind, value, raw)
    };
  }
  if (typeof raw === 'object' && 'output' in raw) {
    return createHistoryEntryFromValue({
      id: raw.id || `legacy-${Date.now()}`,
      timestamp: typeof raw.timestamp === 'number' ? raw.timestamp : Date.now(),
      value: (raw as any).output,
      executionTime: raw.executionTime,
      metadata: raw.metadata
    });
  }
  return null;
};

export const normalizeHistoryEntries = (list: any[]): NodeHistoryEntry[] => {
  if (!Array.isArray(list)) return [];
  return list
    .map(normalizeHistoryEntry)
    .filter((entry): entry is NodeHistoryEntry => !!entry);
};

export const normalizeHistoryMap = (history?: Record<string, any[]>): Record<string, NodeHistoryEntry[]> => {
  if (!history) return {};
  return Object.fromEntries(
    Object.entries(history).map(([nodeId, entries]) => [nodeId, normalizeHistoryEntries(entries as any[])])
  );
};

export const historyEntryToDisplayValue = (entry: NodeHistoryEntry): any => {
  const value = entry.value || {};
  switch (entry.kind) {
    case 'text':
      return (value as { text?: string }).text ?? '';
    case 'json':
      return cloneJson((value as { json?: any }).json ?? value);
    case 'file': {
      const fileValue = value as {
        fileId?: string;
        url?: string;
        dataUrl?: string;
        ext?: string;
      };
      if (fileValue.dataUrl) return fileValue.dataUrl;
      if (fileValue.url) return { type: 'url', data: fileValue.url, ext: fileValue.ext };
      if (fileValue.fileId) {
        return { type: 'reference', file_id: fileValue.fileId, ext: fileValue.ext };
      }
      return null;
    }
    case 'lightx2v_result':
      return {
        __type: 'lightx2v_result',
        task_id: (value as { taskId?: string }).taskId || '',
        output_name: (value as { outputName?: string }).outputName || 'output',
        is_cloud: !!(value as { isCloud?: boolean }).isCloud
      };
    default:
      return null;
  }
};
