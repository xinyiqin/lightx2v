
export function getAssetPath(path: string | null | undefined): string {
  // 处理 null、undefined 或非字符串类型
  if (!path || typeof path !== 'string') {
    return '';
  }
  
  // 如果已经是完整路径或 data URL，直接返回
  if (path.startsWith('http') || path.startsWith('data:') || path.startsWith('blob:')) {
    return path;
  }
  
  if (path.startsWith('/api/v1/workflow/')) {
    return path;
  }
  
  if (path.includes('/assets/task/result') || path.includes('/assets/workflow/input')) {
    // 获取主应用的 JWT token
    const sharedStore = (window as any).__SHARED_STORE__;
    const token = sharedStore ? sharedStore.getState('token') : localStorage.getItem('accessToken');
    
    if (token) {
      const separator = path.includes('?') ? '&' : '?';
      return `${path}${separator}token=${encodeURIComponent(token)}`;
    }
  }
  
  if (path.startsWith('/assets/')) {
    const basePath = (window as any).__ASSET_BASE_PATH__;
    if (basePath && !path.startsWith(`${basePath}/`)) {
      return `${basePath}${path}`;
    }
  }
  
  return path;
}

export function setAssetBasePath(basePath: string = '/canvas'): void {
  (window as any).__ASSET_BASE_PATH__ = basePath;
}

export function getAssetBasePath(): string {
  return (window as any).__ASSET_BASE_PATH__ || '/canvas';
}

