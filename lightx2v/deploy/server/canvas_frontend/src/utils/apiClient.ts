
/**
 * 初始化 LIGHTX2V_TOKEN
 * - 如果使用环境变量 URL (LIGHTX2V_URL 存在)，则必须使用环境变量的 LIGHTX2V_TOKEN，不存在则报错
 * - 否则，如果用户已登录，使用用户的 accessToken
 */
export function initLightX2VToken(): void {
  const envUrl = process.env.LIGHTX2V_URL;
  const envToken = process.env.LIGHTX2V_TOKEN;

  // 如果使用环境变量 URL，必须使用环境变量的 TOKEN
  if (envUrl && envUrl.trim()) {
    if (!envToken || !envToken.trim()) {
      throw new Error('LIGHTX2V_URL 已设置，但 LIGHTX2V_TOKEN 未设置。请设置 LIGHTX2V_TOKEN 环境变量。');
    }
    // 使用环境变量的 TOKEN，不同步 localStorage
    console.log('[LightX2V] 使用环境变量 URL，使用环境变量 LIGHTX2V_TOKEN');
    console.log('[LightX2V] process.env.LIGHTX2V_TOKEN:', envToken ? `${envToken.substring(0, 10)}...` : 'empty');
    return;
  }

  // 不使用环境变量 URL 时，使用原来的逻辑
  const accessToken = localStorage.getItem('accessToken');
  if (accessToken) {
    // 用户已登录，直接设置 process.env.LIGHTX2V_TOKEN 为用户的 accessToken
    (process.env as any).LIGHTX2V_TOKEN = accessToken;
    console.log('[LightX2V] 已将 process.env.LIGHTX2V_TOKEN 设置为用户登录的 accessToken');
    console.log('[LightX2V] process.env.LIGHTX2V_TOKEN:', accessToken ? `${accessToken.substring(0, 10)}...` : 'empty');
  } else {
    // 用户未登录，保持使用环境变量的值
    console.log('[LightX2V] 用户未登录，使用环境变量 LIGHTX2V_TOKEN');
    console.log('[LightX2V] process.env.LIGHTX2V_TOKEN:', envToken ? `${envToken.substring(0, 10)}...` : 'empty');
  }
}

/**
 * 获取 API 客户端实例
 */
export function getApiClient(): any {
  return (window as any).__API_CLIENT__ || null;
}

/**
 * 获取共享存储实例
 */
export function getSharedStore(): any {
  return (window as any).__SHARED_STORE__ || null;
}

/**
 * 统一的 API 请求函数
 * 优先使用主应用的 apiClient，否则使用直接 fetch
 */
/**
 * 获取认证 headers（参考主应用的 getAuthHeaders）
 */
function getAuthHeaders(): Record<string, string> {
  const headers: Record<string, string> = {
    'Content-Type': 'application/json'
  };

  // 优先从 sharedStore 获取 token，否则从 localStorage 获取
  const sharedStore = getSharedStore();
  const token = sharedStore ? sharedStore.getState('token') : localStorage.getItem('accessToken');

  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
    console.log('[apiRequest] 使用Token进行认证:', token.substring(0, 20) + '...');
  } else {
    console.warn('[apiRequest] 没有找到accessToken');
  }

  return headers;
}

export async function apiRequest(url: string, options: RequestInit = {}): Promise<Response> {
  const apiClient = getApiClient();

  // 参考主应用的做法：总是获取认证 headers，然后合并 options.headers
  // options.headers 会覆盖 getAuthHeaders() 中的 headers
  const authHeaders = getAuthHeaders();
  const incomingHeaders = (options.headers as Record<string, string>) || {};

  // 合并 headers：先使用 authHeaders，然后 options.headers 会覆盖
  const headers: Record<string, string> = {
    ...authHeaders,
    ...incomingHeaders
  };

  // 如果有 apiClient，使用它
  if (apiClient && typeof apiClient.request === 'function') {
    // apiClient 使用相对路径，但需要完整 URL
    // 获取 baseURL
    const baseURL = apiClient.baseURL || '';
    const fullUrl = url.startsWith('http') ? url : `${baseURL}${url}`;

    // 使用 apiClient 的 request 方法
    // 但 apiClient.request 返回的是解析后的 JSON，我们需要 Response 对象
    // 所以需要直接使用 fetch，但使用 apiClient 的配置
    return fetch(fullUrl, {
      ...options,
      headers
    });
  }

  // 回退到直接 fetch（使用相对路径，浏览器会自动使用当前域名）
  return fetch(url, {
    ...options,
    headers
  });
}

/**
 * 获取后端基础 URL
 */
export function getApiBaseUrl(): string {
  const apiClient = getApiClient();
  if (apiClient && apiClient.baseURL) {
    return apiClient.baseURL;
  }
  // 回退到当前域名
  return typeof window !== 'undefined'
    ? `${window.location.protocol}//${window.location.host}`
    : 'http://localhost:8082';
}

/**
 * 获取访问令牌
 * - 如果使用环境变量 URL (LIGHTX2V_URL 存在)，则只使用环境变量的 LIGHTX2V_TOKEN
 * - 否则，优先使用用户登录的 accessToken，否则使用 process.env.LIGHTX2V_TOKEN
 */
// 缓存 token，避免重复读取 localStorage
let tokenCache: { token: string; timestamp: number } | null = null;
const TOKEN_CACHE_TTL = 1000; // 1秒缓存，避免频繁读取

export function getAccessToken(): string {
  const envUrl = process.env.LIGHTX2V_URL;

  // 如果使用环境变量 URL，只使用环境变量的 TOKEN
  if (envUrl && envUrl.trim()) {
    const envToken = (process.env.LIGHTX2V_TOKEN || '').trim();
    if (!envToken) {
      throw new Error('LIGHTX2V_URL 已设置，但 LIGHTX2V_TOKEN 未设置。请设置 LIGHTX2V_TOKEN 环境变量。');
    }
    // 只在开发环境或首次调用时打印日志
    if (!tokenCache || Date.now() - tokenCache.timestamp > TOKEN_CACHE_TTL) {
      // 减少日志输出
    }
    return envToken;
  }

  // 检查缓存
  const now = Date.now();
  if (tokenCache && now - tokenCache.timestamp < TOKEN_CACHE_TTL) {
    return tokenCache.token;
  }

  // 不使用环境变量 URL 时，使用原来的逻辑
  const localToken = localStorage.getItem('accessToken');

  // 只在首次调用或缓存过期时打印日志
  if (!tokenCache || now - tokenCache.timestamp > TOKEN_CACHE_TTL) {
    // 减少日志输出，只在必要时打印
  }

  let token: string;
  if (localToken) {
    // 如果获取到了 token，同时更新 process.env.LIGHTX2V_TOKEN
    (process.env as any).LIGHTX2V_TOKEN = localToken;
    token = localToken;
  } else {
    // 如果没有登录，使用环境变量
    token = (process.env.LIGHTX2V_TOKEN || '').trim();
  }

  // 更新缓存
  tokenCache = { token, timestamp: now };

  return token;
}
