import { apiRequest } from './apiClient';

/**
 * 检查工作流是否为预设工作流（不属于当前用户）
 * @param workflowId 工作流ID
 * @param currentUserId 当前用户ID
 * @returns Promise<{isPreset: boolean, workflow: any | null}> 如果是预设工作流，返回true和null；否则返回false和工作流对象
 */
export async function checkWorkflowOwnership(
  workflowId: string,
  currentUserId: string | null
): Promise<{ isPreset: boolean; workflow: any | null }> {
  if (!currentUserId || !workflowId) {
    return { isPreset: false, workflow: null };
  }

  try {
    const checkResponse = await apiRequest(`/api/v1/workflow/${workflowId}`);
    
    if (checkResponse.ok) {
      const existingWorkflow = await checkResponse.json();
      // 如果工作流存在但不属于当前用户，则是预设工作流
      if (existingWorkflow.user_id && existingWorkflow.user_id !== currentUserId) {
        return { isPreset: true, workflow: null };
      }
      return { isPreset: false, workflow: existingWorkflow };
    } else {
      // 工作流不存在
      return { isPreset: false, workflow: null };
    }
  } catch (error) {
    console.warn('[WorkflowUtils] Failed to check workflow ownership:', error);
    // 出错时假设不是预设工作流，继续尝试更新
    return { isPreset: false, workflow: null };
  }
}

/**
 * 获取当前用户ID
 * @returns 当前用户ID或null
 */
export function getCurrentUserId(): string | null {
  try {
    const sharedStore = (window as any).__SHARED_STORE__;
    if (sharedStore) {
      const user = sharedStore.getState('user');
      return user?.user_id || null;
    }
    
    // Fallback to localStorage
    try {
      const userStr = localStorage.getItem('currentUser');
      if (userStr) {
        const user = JSON.parse(userStr);
        return user?.user_id || null;
      }
    } catch (e) {
      // Ignore
    }
    
    return null;
  } catch (error) {
    console.warn('[WorkflowUtils] Failed to get current user ID:', error);
    return null;
  }
}
