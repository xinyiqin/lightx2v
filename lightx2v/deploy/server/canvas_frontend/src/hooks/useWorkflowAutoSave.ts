import { useEffect, useRef, useCallback } from 'react';
import { WorkflowState } from '../../types';
import { apiRequest } from '../utils/apiClient';
import { workflowSaveQueue } from '../utils/workflowSaveQueue';

const AUTO_SAVE_INTERVAL = 30000; // 30 seconds

interface UseWorkflowAutoSaveProps {
  workflow: WorkflowState | null;
  onSave?: (workflow: WorkflowState) => Promise<void>;
}

export interface UseWorkflowAutoSaveReturn {
  resetAutoSaveTimer: () => void; // 重置自动保存计时器（手动保存后调用）
}

/**
 * Hook for auto-saving workflows
 * Automatically saves workflow structure (nodes, connections, global_inputs) every 30 seconds
 * if the workflow is dirty (has unsaved changes)
 * Now uses save queue to avoid conflicts with manual saves
 */
export const useWorkflowAutoSave = ({
  workflow,
  onSave
}: UseWorkflowAutoSaveProps): UseWorkflowAutoSaveReturn => {
  const autoSaveTimerRef = useRef<NodeJS.Timeout | null>(null);
  const lastSavedRef = useRef<number>(0);

  // 重置自动保存计时器（手动保存后调用）
  const resetAutoSaveTimer = useCallback(() => {
    // 清除当前计时器
    if (autoSaveTimerRef.current) {
      clearTimeout(autoSaveTimerRef.current);
      autoSaveTimerRef.current = null;
    }
    // 更新最后保存时间，防止立即触发自动保存
    lastSavedRef.current = Date.now();
    console.log('[Workflow AutoSave] Timer reset after manual save');
  }, []);

  useEffect(() => {
    if (!workflow || !workflow.isDirty || !workflow.id) {
      // Clear timer if workflow is not dirty or doesn't have an ID
      if (autoSaveTimerRef.current) {
        clearTimeout(autoSaveTimerRef.current);
        autoSaveTimerRef.current = null;
      }
      return;
    }

    // Clear previous timer
    if (autoSaveTimerRef.current) {
      clearTimeout(autoSaveTimerRef.current);
    }

    // Set up auto-save timer
    autoSaveTimerRef.current = setTimeout(async () => {
      try {
        // Only auto-save if workflow hasn't been saved in the last 30 seconds
        const now = Date.now();
        if (now - lastSavedRef.current < AUTO_SAVE_INTERVAL) {
          return;
        }

        // Use save queue for auto-save (lower priority, but still queued)
        await workflowSaveQueue.enqueue(workflow.id, async () => {
          // 重新获取最新的工作流状态（从传入的 workflow，但应该是最新的）
          // 自动保存时也保存执行状态，确保数据一致性
          // 自动保存通过 URL 路径传递 workflow_id，后端会验证该 workflow_id 是否属于当前用户
          const response = await apiRequest(`/api/v1/workflow/${workflow.id}/autosave`, {
            method: 'POST',
            body: JSON.stringify({
              workflow_id: workflow.id, // 显式传递 workflow_id 用于验证
              name: workflow.name,
              nodes: workflow.nodes, // 包含执行状态
              connections: workflow.connections,
              global_inputs: workflow.globalInputs
            })
          });

          if (response.ok) {
            const data = await response.json();
            lastSavedRef.current = now;
            console.log('[Workflow AutoSave] Workflow autosaved:', workflow.id, data);

            // Call optional onSave callback
            if (onSave) {
              await onSave(workflow);
            }
          } else {
            console.warn('[Workflow AutoSave] Failed to autosave workflow:', workflow.id);
          }
        });
      } catch (error) {
        console.error('[Workflow AutoSave] Error autosaving workflow:', error);
      }
    }, AUTO_SAVE_INTERVAL);

    // Cleanup on unmount or when dependencies change
    return () => {
      if (autoSaveTimerRef.current) {
        clearTimeout(autoSaveTimerRef.current);
        autoSaveTimerRef.current = null;
      }
    };
  }, [workflow?.isDirty, workflow?.id, workflow?.name, workflow?.nodes, workflow?.connections, workflow?.globalInputs, onSave]);

  return {
    resetAutoSaveTimer
  };
};
