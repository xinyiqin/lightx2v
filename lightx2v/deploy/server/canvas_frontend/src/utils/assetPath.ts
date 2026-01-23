/**
 * 获取正确的静态资源路径
 * 在 qiankun 环境中，资源路径需要加上 /canvas/ 前缀
 * 对于需要认证的结果 URL，添加 token 查询参数
 */
export function getAssetPath(path: string): string {
  // 如果已经是完整路径或 data URL，直接返回
  if (path.startsWith('http') || path.startsWith('data:') || path.startsWith('blob:')) {
    return path;
  }
  
  // 如果路径是相对路径的结果 URL（如 ./assets/task/result 或 /assets/task/result），需要添加 token
  // 后端 verify_user_access_from_query 可以从查询参数或 Authorization header 中获取 token
  if (path.includes('/assets/task/result') || path.includes('./assets/task/result')) {
    // 获取主应用的 JWT token
    const sharedStore = (window as any).__SHARED_STORE__;
    const token = sharedStore ? sharedStore.getState('token') : localStorage.getItem('accessToken');
    
    if (token) {
      // 检查 URL 是否已经有查询参数
      const separator = path.includes('?') ? '&' : '?';
      return `${path}${separator}token=${encodeURIComponent(token)}`;
    }
  }
  
  // 如果路径以 /assets/ 开头，需要添加基础路径前缀
  // 在 qiankun 环境中为 /canvas，独立运行时为空字符串
  if (path.startsWith('/assets/')) {
    const basePath = (window as any).__ASSET_BASE_PATH__;
    // 如果设置了基础路径且路径不包含它，则添加
    if (basePath && !path.startsWith(`${basePath}/`)) {
      return `${basePath}${path}`;
    }
  }
  
  return path;
}

/**
 * 设置全局的资源基础路径
 * 在 mount 时调用，确保所有资源路径正确
 */
export function setAssetBasePath(basePath: string = '/canvas'): void {
  (window as any).__ASSET_BASE_PATH__ = basePath;
}

/**
 * 获取全局的资源基础路径
 */
export function getAssetBasePath(): string {
  return (window as any).__ASSET_BASE_PATH__ || '/canvas';
}

