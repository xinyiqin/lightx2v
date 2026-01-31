import { apiRequest } from './apiClient';
import { getAssetPath } from './assetPath';
import { lightX2VResultUrl } from '../../services/geminiService';
import { isStandalone } from '../config/runtimeMode';

/** LightX2V 结果引用：用 task_id + output_name 代替过期 CDN URL，需要时通过 result_url 解析 */
export type LightX2VResultRef = { __type: 'lightx2v_result'; task_id: string; output_name: string; is_cloud: boolean };

export function isLightX2VResultRef(val: any): val is LightX2VResultRef {
  return val != null && typeof val === 'object' && !Array.isArray(val) &&
    (val as any).__type === 'lightx2v_result' &&
    typeof (val as any).task_id === 'string' &&
    typeof (val as any).output_name === 'string';
}

export function toLightX2VResultRef(task_id: string, output_name: string, is_cloud: boolean): LightX2VResultRef {
  return { __type: 'lightx2v_result', task_id, output_name, is_cloud };
}

/** Collect all LightX2V result refs from a value (handles nested objects/arrays) */
export function collectLightX2VResultRefs(val: any): LightX2VResultRef[] {
  if (val == null) return [];
  if (isLightX2VResultRef(val)) return [val];
  if (Array.isArray(val)) return val.flatMap(collectLightX2VResultRefs);
  if (typeof val === 'object') return Object.values(val).flatMap(collectLightX2VResultRefs);
  return [];
}

/** 缓存 result_url 解析结果，避免同一 ref 反复请求（presigned URL 有效期较长，缓存 1 小时） */
const LIGHTX2V_RESULT_URL_CACHE_TTL_MS = 60 * 60 * 1000;
const lightX2VResultUrlCache = new Map<string, { url: string; ts: number }>();

function getCachedResultUrl(cacheKey: string): string | null {
  const entry = lightX2VResultUrlCache.get(cacheKey);
  if (!entry || Date.now() - entry.ts > LIGHTX2V_RESULT_URL_CACHE_TTL_MS) return null;
  return entry.url;
}

function setCachedResultUrl(cacheKey: string, url: string): void {
  lightX2VResultUrlCache.set(cacheKey, { url, ts: Date.now() });
}

/**
 * Fetch LightX2V result URL via canvas proxy API (task_id + output_name).
 * Used when canvas runs standalone; proxy forwards to LightX2V cloud or local backend.
 */
export async function fetchLightX2VResultUrl(
  taskId: string,
  outputName: string,
  isCloud: boolean
): Promise<string> {
  const params = new URLSearchParams({
    task_id: taskId,
    output_name: outputName,
    is_cloud: String(isCloud)
  });
  const res = await fetch(`/api/lightx2v/result_url?${params.toString()}`, { method: 'GET' });
  if (!res.ok) {
    const err = await res.text();
    throw new Error(`result_url failed: ${res.status} ${err}`);
  }
  const data = (await res.json().catch(() => ({}))) as { url?: string };
  if (!data.url) throw new Error('result_url missing url');
  return data.url;
}

/**
 * Resolve LightX2V result ref to a displayable URL via result_url (backend or cloud).
 * Uses cache to avoid repeated calls. For backend asset paths, normalizes to absolute URL with token for <img>.
 */
export async function resolveLightX2VResultRef(ref: LightX2VResultRef): Promise<string> {
  const cacheKey = `${ref.is_cloud}:${ref.task_id}:${ref.output_name}`;
  const cached = getCachedResultUrl(cacheKey);
  if (cached != null) return cached;

  let url: string;
  if (isStandalone()) {
    // Use canvas proxy API (same-origin), which forwards to LightX2V cloud or local
    url = await fetchLightX2VResultUrl(ref.task_id, ref.output_name, ref.is_cloud);
  } else if (ref.is_cloud) {
    const cloudUrl = (process.env.LIGHTX2V_CLOUD_URL || 'https://x2v.light-ai.top').trim();
    const cloudToken = (process.env.LIGHTX2V_CLOUD_TOKEN || '').trim();
    url = await lightX2VResultUrl(cloudUrl, cloudToken, ref.task_id, ref.output_name);
  } else {
    const res = await apiRequest(`/api/v1/task/result_url?task_id=${encodeURIComponent(ref.task_id)}&name=${encodeURIComponent(ref.output_name)}`, { method: 'GET' });
    if (!res.ok) throw new Error(`result_url failed: ${res.status}`);
    const data = await res.json().catch(() => ({})) as { url?: string };
    if (!data.url) throw new Error('result_url missing url');
    url = data.url;
    // Backend may return relative path — under /canvas that would 404; normalize to root path and add token for <img>
    if (url.startsWith('./')) url = url.slice(1);
    if (!url.startsWith('/') && !url.startsWith('http')) url = '/' + url;
    if (url.includes('/assets/task/result') || url.includes('/assets/workflow/input')) {
      url = getAssetPath(url);
      if (typeof window !== 'undefined' && url.startsWith('/')) url = window.location.origin + url;
    }
  }
  setCachedResultUrl(cacheKey, url);
  return url;
}
