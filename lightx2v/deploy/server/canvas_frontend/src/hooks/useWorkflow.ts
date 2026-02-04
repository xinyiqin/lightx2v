import React, { useState, useCallback, useEffect, useRef } from 'react';
import { WorkflowState, WorkflowNode, NodeHistoryEntry } from '../../types';
import { apiRequest } from '../utils/apiClient';
import { getWorkflowFileByFileId, getLocalFileDataUrl, persistDataUrlToLocal, uploadLocalUrlAsNodeOutput, isLocalAssetUrlToUpload } from '../utils/workflowFileManager';
import { TOOLS } from '../../constants';
import { checkWorkflowOwnership, getCurrentUserId } from '../utils/workflowUtils';
import { workflowSaveQueue } from '../utils/workflowSaveQueue';
import { workflowOfflineQueue } from '../utils/workflowOfflineQueue';
import { isStandalone } from '../config/runtimeMode';
import { createHistoryEntryFromValue, normalizeHistoryMap } from '../utils/historyEntry';

export const useWorkflow = () => {
  const [myWorkflows, setMyWorkflows] = useState<WorkflowState[]>([]);
  const [workflow, setWorkflow] = useState<WorkflowState | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isSaving, setIsSaving] = useState(false);

  // 使用 ref 存储最新的工作流状态，用于保存时获取最新数据
  const workflowRef = useRef<WorkflowState | null>(null);
  useEffect(() => {
    workflowRef.current = workflow;
  }, [workflow]);

  // Load workflows from API (with localStorage fallback)
  const loadWorkflows = useCallback(async () => {
    setIsLoading(true);
    if (isStandalone()) {
      const saved = localStorage.getItem('omniflow_user_data');
      if (saved) {
        try {
          const workflows = JSON.parse(saved);
          setMyWorkflows(Array.isArray(workflows) ? workflows : []);
          console.log('[Workflow] [Standalone] Loaded workflows from localStorage:', (workflows || []).length);
        } catch (e) {
          console.error('[Workflow] Failed to parse omniflow_user_data:', e);
        }
      }
      setIsLoading(false);
      return;
    }
    try {
      // Try to load from API first
      const response = await apiRequest('/api/v1/workflow/list?page=1&page_size=100');
      if (response.ok) {
        const data = await response.json();
        const workflows: WorkflowState[] = data.workflows.map((wf: any) => ({
          id: wf.workflow_id,
          name: wf.name,
          nodes: wf.nodes || [],
          connections: wf.connections || [],
          isDirty: false,
          isRunning: false,
          globalInputs: wf.global_inputs || {},
          updatedAt: wf.update_t * 1000, // Convert to milliseconds
          visibility: wf.visibility || 'private',
          thumsupCount: wf.thumsup_count ?? 0,
          thumsupLiked: wf.thumsup_liked ?? false
        }));
        setMyWorkflows(workflows);
        console.log('[Workflow] Loaded workflows from API:', workflows.length);
        setIsLoading(false);
        return;
      }
    } catch (error) {
      console.warn('[Workflow] Failed to load workflows from API, falling back to localStorage:', error);
    }

    // Fallback to localStorage
    const saved = localStorage.getItem('omniflow_user_data');
    if (saved) {
      try {
        const workflows = JSON.parse(saved);
        setMyWorkflows(workflows);
        console.log('[Workflow] Loaded workflows from localStorage:', workflows.length);
      } catch (e) {
        console.error('Failed to load workflows from localStorage:', e);
      }
    }
    setIsLoading(false);
  }, []);

  // Load workflows on mount
  useEffect(() => {
    loadWorkflows();
  }, [loadWorkflows]);

  // Save workflow to localStorage (for backup and local development)
  // Define this BEFORE saveWorkflowToDatabase to avoid "Cannot access before initialization" error
  const saveWorkflowToLocal = useCallback(async (current: WorkflowState) => {
    let nodesToSave = current.nodes;
    // 仅前端：先把 data: 存入 IndexedDB 得到 local://，再保存列表，这样 MY 列表缩略图能解析显示
    if (isStandalone()) {
      nodesToSave = await Promise.all(current.nodes.map(async (node) => {
        const tool = TOOLS.find(t => t.id === node.toolId);
        if (!tool || tool.category !== 'Input' || !node.data?.value) return node;
        const nodeValue = node.data.value;
        const prefix = `wf_${current.id}_${node.id}`;
        if (Array.isArray(nodeValue)) {
          const replaced = await Promise.all(nodeValue.map((v: string, i: number) =>
            typeof v === 'string' && v.startsWith('data:') ? persistDataUrlToLocal(v, `${prefix}_${i}`) : v
          ));
          return { ...node, data: { ...node.data, value: replaced } };
        }
        if (typeof nodeValue === 'string' && nodeValue.startsWith('data:')) {
          const localRef = await persistDataUrlToLocal(nodeValue, `${prefix}_0`);
          return { ...node, data: { ...node.data, value: localRef } };
        }
        return node;
      }));
    }
    // Clean input node values: remove base64 data URLs, keep only file paths (standalone 下已是 local://)
    const cleanedNodes = nodesToSave.map(node => {
      const tool = TOOLS.find(t => t.id === node.toolId);
      if (!tool || tool.category !== 'Input') {
        // Not an input node, return as-is
        return node;
      }

      const nodeValue = node.data.value;
      if (!nodeValue) {
        return node;
      }

      // Check if it's a base64 data URL
      const isDataURL = (val: any): boolean => {
        if (typeof val === 'string') {
          return val.startsWith('data:');
        }
        if (Array.isArray(val)) {
          return val.some(item => typeof item === 'string' && item.startsWith('data:'));
        }
        return false;
      };

      // Check if it's already a saved path (including standalone local://)
      const isSavedPath = (val: any): boolean => {
        if (typeof val === 'string') {
          return val.startsWith('local://') ||
                 val.startsWith('./assets/workflow/input') ||
                 val.startsWith('./assets/task/') ||
                 val.startsWith('/assets/workflow/input') ||
                 val.startsWith('/assets/task/') ||
                 val.startsWith('/api/v1/workflow/') ||
                 (val.startsWith('http://') || val.startsWith('https://'));
        }
        if (Array.isArray(val)) {
          return val.every(item => typeof item === 'string' && (
            item.startsWith('local://') ||
            item.startsWith('./assets/workflow/input') ||
            item.startsWith('./assets/task/') ||
            item.startsWith('/assets/workflow/input') ||
            item.startsWith('/assets/task/') ||
            item.startsWith('/api/v1/workflow/') ||
            item.startsWith('http://') ||
            item.startsWith('https://')
          ));
        }
        return false;
      };

      // If it's base64 and not saved, remove it (don't save base64 to localStorage)
      if (isDataURL(nodeValue) && !isSavedPath(nodeValue)) {
        // Remove base64 data - user will need to re-upload or it will be saved during execution
        return {
          ...node,
          data: {
            ...node.data,
            value: Array.isArray(nodeValue) ? [] : undefined
          }
        };
      }

      return node;
    });

    const updated = {
      ...current,
      nodes: cleanedNodes, // Use cleaned nodes (no base64 data)
      updatedAt: Date.now(),
      isDirty: false
    };

    setMyWorkflows(prev => {
      const next = prev.find(w => w.id === updated.id)
        ? prev.map(w => w.id === updated.id ? updated : w)
        : [updated, ...prev];

      try {
        localStorage.setItem('omniflow_user_data', JSON.stringify(next));
      } catch (e: any) {
        console.error('Failed to save workflows:', e);
      }
      return next;
    });
  }, []);

  /** 仅保存到本地（localStorage + 列表），不请求后端。用于仅部署前端时 autosave 后端失败后的回退。 */
  const saveWorkflowToLocalOnly = useCallback(async (current: WorkflowState, options?: { name?: string }): Promise<string> => {
    const workflowId = current.id;
    const toSave = options?.name ? { ...current, name: options.name } : current;
    await saveWorkflowToLocal(toSave);
    return workflowId;
  }, [saveWorkflowToLocal]);

  // Save workflow to database (and optionally localStorage)
  const saveWorkflowToDatabase = useCallback(async (current: WorkflowState, options?: { name?: string; description?: string }): Promise<string> => {
    const workflowId = current.id;

    // 使用保存队列，避免并发保存冲突
    return workflowSaveQueue.enqueue(workflowId, async () => {
      setIsSaving(true);
      const latestWorkflow = (workflowRef.current && workflowRef.current.id === workflowId)
        ? workflowRef.current
        : current;

      // preset 工作流（仅前端或有后端）保存时都先分配新 UUID，再按环境保存
      let effectiveId = workflowId;
      let toSave: WorkflowState = options?.name ? { ...latestWorkflow, name: options.name } : latestWorkflow;
      if (workflowId.startsWith('preset-')) {
        effectiveId = typeof crypto !== 'undefined' && crypto.randomUUID
          ? crypto.randomUUID()
          : `workflow-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
        toSave = { ...toSave, id: effectiveId };
        setWorkflow(prev => (prev && prev.id === workflowId ? { ...prev, id: effectiveId } : prev));
        if (typeof window !== 'undefined' && window.history?.replaceState) {
          window.history.replaceState(null, '', `#workflow/${effectiveId}`);
        }
        console.log('[Workflow] Preset workflow assigned new ID on save:', effectiveId);
      }

      if (isStandalone()) {
        await saveWorkflowToLocal(toSave);
        setIsSaving(false);
        return effectiveId;
      }

      let workflowData: any = null;

      try {
        // 检查工作流是否仍然存在（通过检查是否在 myWorkflows 中）
        // 如果工作流已被删除，跳过保存
        // 注意：对于新创建的工作流（UUID格式），可能不在 myWorkflows 中，这是正常的，不需要查询
        const workflows = myWorkflows;
        const workflowExists = workflows.some(w => w.id === effectiveId);

        // 只有在工作流已存在且不在列表中时才检查（可能是被删除了）
        // 对于新创建的工作流（UUID格式），即使不在列表中也是正常的，不需要查询
        if (!workflowExists && workflowId.match(/^workflow-/)) {
          // 只有 workflow- 前缀的工作流才需要检查（这些是已存在的工作流）
          // 尝试查询工作流是否存在
          try {
            const checkResponse = await apiRequest(`/api/v1/workflow/${workflowId}`);
            if (!checkResponse.ok) {
              throw new Error(`Workflow ${workflowId} was deleted, save task cancelled`);
            }
          } catch (error) {
            if (error instanceof Error && error.message.includes('deleted')) {
              throw error;
            }
            // 其他错误继续处理
          }
        }

        // Extract execution state from nodes (status, error, executionTime)
        // Nodes already contain execution state, so we just save them as-is

        // 输入节点与文本输入一致：执行时才存库，保存工作流时保留 data URL / ref
        const cleanedNodes = [...latestWorkflow.nodes];

        const stripBase64FromOutput = (value: any): any => {
          if (typeof value === 'string') {
            return value.startsWith('data:') ? '' : value;
          }
          if (Array.isArray(value)) {
            return value
              .map(stripBase64FromOutput)
              .filter(v => v !== '' && v !== null && v !== undefined);
          }
          if (value && typeof value === 'object') {
            if (value.type === 'file' || value.file_id) {
              return value;
            }
            // Preserve LightX2VResultRef (task_id + output_name) so saved workflow can resolve result_url
            if (value.__type === 'lightx2v_result' && typeof value.task_id === 'string' && typeof value.output_name === 'string') {
              return value;
            }
            const cleaned: Record<string, any> = {};
            Object.entries(value).forEach(([key, val]) => {
              const nextVal = stripBase64FromOutput(val);
              if (nextVal !== '' && nextVal !== null && nextVal !== undefined) {
                cleaned[key] = nextVal;
              }
            });
            return cleaned;
          }
          return value;
        };

        // Helper function to clean base64 data from node data.value
        const cleanNodeDataValue = (nodeData: any): any => {
          if (!nodeData || !nodeData.value) return nodeData;

          const nodeValue = nodeData.value;
          const isDataURL = (val: any): boolean => {
            if (typeof val === 'string') return val.startsWith('data:');
            if (Array.isArray(val)) return val.some(item => typeof item === 'string' && item.startsWith('data:'));
            return false;
          };

          const isSavedPath = (val: any): boolean => {
            if (val && typeof val === 'object' && (val.type === 'file' || val.file_id)) {
              return true;
            }
            if (typeof val === 'string') {
              return val.startsWith('./assets/workflow/input') ||
                     val.startsWith('./assets/task/') ||
                     val.startsWith('/assets/workflow/input') ||
                     val.startsWith('/assets/task/') ||
                     val.startsWith('/api/v1/workflow/') ||
                     (val.startsWith('http://') || val.startsWith('https://'));
            }
            if (Array.isArray(val)) {
              return val.every(item => {
                if (item && typeof item === 'object' && (item.type === 'file' || item.file_id)) return true;
                if (typeof item !== 'string') return false;
                return item.startsWith('./assets/workflow/input') ||
                       item.startsWith('./assets/task/') ||
                       item.startsWith('/assets/workflow/input') ||
                       item.startsWith('/assets/task/') ||
                       item.startsWith('/api/v1/workflow/') ||
                       item.startsWith('http://') ||
                       item.startsWith('https://');
              });
            }
            return false;
          };

          // If it's base64 and not saved, remove it
          if (isDataURL(nodeValue) && !isSavedPath(nodeValue)) {
            return {
              ...nodeData,
              value: Array.isArray(nodeValue) ? [] : undefined
            };
          }

          return nodeData;
        };

        // Clean node outputValue to avoid persisting base64 in workflow nodes
        // 输入节点与文本输入一致：执行时才存库，保存工作流时保留 data URL 不 strip
        for (let i = 0; i < cleanedNodes.length; i++) {
          const node = cleanedNodes[i];
          const nodeTool = TOOLS.find(t => t.id === node.toolId);
          if (nodeTool?.category === 'Input') continue;
          if (node.outputValue !== undefined) {
            const cleanedOutputValue = stripBase64FromOutput(node.outputValue);
            if (Array.isArray(cleanedOutputValue)) {
              cleanedNodes[i] = {
                ...cleanedNodes[i],
                outputValue: cleanedOutputValue.length > 0 ? cleanedOutputValue : undefined
              };
            } else if (cleanedOutputValue && typeof cleanedOutputValue === 'object') {
              cleanedNodes[i] = {
                ...cleanedNodes[i],
                outputValue: Object.keys(cleanedOutputValue).length > 0 ? cleanedOutputValue : undefined
              };
            } else {
              cleanedNodes[i] = {
                ...cleanedNodes[i],
                outputValue: cleanedOutputValue !== '' && cleanedOutputValue !== null && cleanedOutputValue !== undefined
                  ? cleanedOutputValue
                  : undefined
              };
            }
          }
        }

        // 使用后端时，将输入节点中的本地 URL（如 /assets/girl.png）通过 save/upload 存到后端，数据库以 file 类型存储
        const INPUT_PORT_ID: Record<string, string> = {
          'image-input': 'out-image',
          'video-input': 'out-video',
          'audio-input': 'out-audio'
        };
        const resolveInputNodeLocalUrls = async (wfId: string, nodes: WorkflowNode[]): Promise<WorkflowNode[]> => {
          const next = [...nodes];
          for (let i = 0; i < next.length; i++) {
            const node = next[i];
            const tool = TOOLS.find(t => t.id === node.toolId);
            if (!tool || tool.category !== 'Input') continue;
            const portId = INPUT_PORT_ID[node.toolId];
            if (!portId) continue;
            const nodeValue = node.data?.value;
            if (!nodeValue) continue;
            const isLocalUrl = (s: string) => typeof s === 'string' && isLocalAssetUrlToUpload(s);
            if (Array.isArray(nodeValue)) {
              const resolved: any[] = [];
              for (let idx = 0; idx < nodeValue.length; idx++) {
                const item = nodeValue[idx];
                if (item && typeof item === 'object' && (item.type === 'file' || item.file_id)) {
                  resolved.push(item);
                  continue;
                }
                if (!isLocalUrl(item)) {
                  resolved.push(item);
                  continue;
                }
                const ref = await uploadLocalUrlAsNodeOutput(wfId, node.id, portId, item, idx);
                resolved.push(ref || item);
              }
              next[i] = { ...next[i], data: { ...next[i].data, value: resolved } };
            } else if (typeof nodeValue === 'string' && isLocalUrl(nodeValue)) {
              const ref = await uploadLocalUrlAsNodeOutput(wfId, node.id, portId, nodeValue, 0);
              next[i] = { ...next[i], data: { ...next[i].data, value: ref || nodeValue } };
            }
          }
          return next;
        };

        let nodesToSave = cleanedNodes;
        const isUuid = (id: string) => /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(id);
        if (!isStandalone() && effectiveId && isUuid(effectiveId)) {
          nodesToSave = await resolveInputNodeLocalUrls(effectiveId, cleanedNodes);
        }

        // 使用最新的工作流状态构建保存数据（不再写入 history_metadata，执行历史按节点存 nodeOutputHistory / data_store）
        const MAX_NODE_HISTORY = 20;
        const trimNodeOutputHistory = (hist: Record<string, any[]>) => {
          if (!hist || typeof hist !== 'object') return {};
          const out: Record<string, any[]> = {};
          for (const [nodeId, entries] of Object.entries(hist)) {
            if (Array.isArray(entries)) {
              out[nodeId] = entries.slice(0, MAX_NODE_HISTORY);
            }
          }
          return out;
        };

        workflowData = {
          name: options?.name || latestWorkflow.name,
          description: options?.description ?? latestWorkflow.description ?? '',
          nodes: nodesToSave,
          connections: latestWorkflow.connections,
          visibility: latestWorkflow.visibility || 'private',
          tags: latestWorkflow.tags ?? [],
          node_output_history: trimNodeOutputHistory(latestWorkflow.nodeOutputHistory ?? {})
        };

        // Check if workflow exists in database and user owns it (else create new UUID first)
        const isSavedWorkflow = workflowId && workflowId.match(/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i);
        const isTemporaryId = workflowId && (workflowId.startsWith('temp-') || workflowId.startsWith('flow-'));

        const currentUserId = getCurrentUserId();
        const { owned } = await checkWorkflowOwnership(effectiveId, currentUserId);
        if (!owned) {
          // 不拥有（预设、他人工作流或 404）：先分配新 UUID，走创建逻辑
          effectiveId = typeof crypto !== 'undefined' && crypto.randomUUID
            ? crypto.randomUUID()
            : `workflow-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
          toSave = { ...toSave, id: effectiveId };
          setWorkflow(prev => (prev && prev.id === workflowId ? { ...prev, id: effectiveId } : prev));
          if (typeof window !== 'undefined' && window.history?.replaceState) {
            window.history.replaceState(null, '', `#workflow/${effectiveId}`);
          }
          console.log('[Workflow] Workflow not owned (preset/404), creating new with ID:', effectiveId);
        }

        // For saved workflows (UUID) that user owns, try to update first
        if (isSavedWorkflow && owned) {
          const updateResponse = await apiRequest(`/api/v1/workflow/${effectiveId}`, {
            method: 'PUT',
            body: JSON.stringify(workflowData)
          });

          if (updateResponse.ok) {
            console.log('[Workflow] Workflow updated in database:', effectiveId);
            await saveWorkflowToLocal(latestWorkflow);
            return effectiveId;
          } else {
            const errorText = await updateResponse.text();
            if (updateResponse.status === 404) {
              console.log('[Workflow] Workflow not found on update, creating new one');
            } else {
              throw new Error(`Failed to update workflow: ${errorText}`);
            }
          }
        }

        // Create new workflow (preset save uses effectiveId; temp/update-failed let backend generate)
        if (effectiveId !== workflowId && effectiveId.match(/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i)) {
          workflowData.workflow_id = effectiveId;
        }
        if (!isTemporaryId && !isSavedWorkflow) {
          console.warn('[Workflow] Unknown workflow ID format, letting backend generate ID');
        }

        console.log('[Workflow] Creating new workflow (backend will generate workflow_id)');
        const response = await apiRequest('/api/v1/workflow/create', {
          method: 'POST',
          body: JSON.stringify(workflowData)
        });

        if (response.ok) {
          const data = await response.json();
          const newWorkflowId = data.workflow_id;
          console.log('[Workflow] Workflow created in database with ID:', newWorkflowId);

          // For new workflows, now save any remaining input files that weren't saved before
          // (because workflow_id didn't exist yet)
          const remainingInputSavePromises: Promise<void>[] = [];
          const finalCleanedNodes = [...cleanedNodes];

          for (let i = 0; i < finalCleanedNodes.length; i++) {
            const node = finalCleanedNodes[i];
            const tool = TOOLS.find(t => t.id === node.toolId);
            if (!tool || tool.category !== 'Input') continue;

            const nodeValue = node.data.value;
            if (!nodeValue) continue;

            const isDataURL = (val: any): boolean => {
              if (typeof val === 'string') return val.startsWith('data:');
              if (Array.isArray(val)) return val.some(item => typeof item === 'string' && item.startsWith('data:'));
              return false;
            };

            const isSavedPath = (val: any): boolean => {
              if (typeof val === 'string') {
                return val.startsWith('./assets/workflow/input') ||
                       val.startsWith('./assets/task/') ||
                       val.startsWith('/assets/workflow/input') ||
                       val.startsWith('/assets/task/') ||
                       (val.startsWith('http://') || val.startsWith('https://'));
              }
              if (Array.isArray(val)) {
                return val.every(item => typeof item === 'string' && (
                  item.startsWith('./assets/workflow/input') ||
                  item.startsWith('./assets/task/') ||
                  item.startsWith('/assets/workflow/input') ||
                  item.startsWith('/assets/task/') ||
                  item.startsWith('http://') ||
                  item.startsWith('https://')
                ));
              }
              return false;
            };

            if (isDataURL(nodeValue) && !isSavedPath(nodeValue)) {
              // Save input file now that we have workflow_id
              const inputNameMap: Record<string, string> = {
                'image-input': 'input_image',
                'video-input': 'input_video',
                'audio-input': 'input_audio'
              };
              const inputName = inputNameMap[node.toolId];
              if (!inputName) continue;

              const params: Record<string, { type: string; data: any }> = {};

              if (node.toolId === 'image-input' && Array.isArray(nodeValue)) {
                for (let idx = 0; idx < nodeValue.length; idx++) {
                  const dataUrl = nodeValue[idx];
                  if (!dataUrl || !dataUrl.startsWith('data:')) continue;
                  const inpKey = idx > 0 ? `${inputName}/${idx}` : inputName;
                  params[inpKey] = { type: 'base64', data: dataUrl };
                }
              } else {
                const dataUrl = Array.isArray(nodeValue) ? nodeValue[0] : nodeValue;
                if (dataUrl && dataUrl.startsWith('data:')) {
                  params[inputName] = { type: 'base64', data: dataUrl };
                }
              }

              if (Object.keys(params).length === 0) continue;

              remainingInputSavePromises.push(
                (async () => {
                  try {
                    const response = await apiRequest('/api/v1/task/submit', {
                      method: 'POST',
                      body: JSON.stringify({
                        task: 't2i',
                        model_cls: 'z-image-turbo',
                        stage: 'single',
                        prompt: 'dummy',
                        ...params
                      })
                    });

                    if (response.ok) {
                      const taskData = await response.json();
                      const taskId = taskData.task_id;

                      try {
                        await apiRequest(`/api/v1/task/cancel?task_id=${taskId}`, { method: 'GET' });
                      } catch (e) {
                        // Ignore cancel errors
                      }

                      const savedPaths: string[] = [];
                      Object.keys(params).forEach(inp => {
                        let baseName = inp;
                        let ext = 'png';
                        if (inp.includes('video')) ext = 'mp4';
                        else if (inp.includes('audio')) ext = 'mp3';
                        const filename = `${taskId}-${baseName}.${ext}`;
                        const workflowPath = `./assets/workflow/input?workflow_id=${newWorkflowId}&filename=${filename}`;
                        savedPaths.push(workflowPath);
                      });

                      if (node.toolId === 'image-input' && Array.isArray(nodeValue)) {
                        finalCleanedNodes[i] = {
                          ...finalCleanedNodes[i],
                          data: {
                            ...finalCleanedNodes[i].data,
                            value: savedPaths.length > 0 ? savedPaths : nodeValue
                          }
                        };
                      } else {
                        finalCleanedNodes[i] = {
                          ...finalCleanedNodes[i],
                          data: {
                            ...finalCleanedNodes[i].data,
                            value: savedPaths[0] || nodeValue
                          }
                        };
                      }

                      // Update workflow in database with cleaned nodes
                      await apiRequest(`/api/v1/workflow/${newWorkflowId}`, {
                        method: 'PUT',
                        body: JSON.stringify({
                          nodes: finalCleanedNodes
                        })
                      });
                    }
                  } catch (err) {
                    console.error(`[Workflow] Error saving input file for node ${node.id} after workflow creation:`, err);
                  }
                })()
              );
            }
          }

          // Wait for remaining input files to be saved
          if (remainingInputSavePromises.length > 0) {
            await Promise.allSettled(remainingInputSavePromises);
          }

          // 新建后：将输入节点中的本地 URL（如 /assets/girl.png）通过 upload 存到后端，再 PUT 节点（数据库以 file 类型存储）
          const nodesAfterLocalUrls = await resolveInputNodeLocalUrls(newWorkflowId, finalCleanedNodes);
          await apiRequest(`/api/v1/workflow/${newWorkflowId}`, {
            method: 'PUT',
            body: JSON.stringify({ nodes: nodesAfterLocalUrls })
          });

          // Always update workflow with backend-generated ID and cleaned nodes
          const updatedWorkflow = { ...latestWorkflow, id: newWorkflowId, nodes: nodesAfterLocalUrls };
          setWorkflow(updatedWorkflow);
          workflowRef.current = updatedWorkflow;
          await saveWorkflowToLocal(updatedWorkflow);
          // Update URL hash to reflect the new workflow ID
          if (typeof window !== 'undefined') {
            window.history.replaceState(null, '', `#workflow/${newWorkflowId}`);
          }
          return newWorkflowId;
        } else {
          const errorText = await response.text();
          throw new Error(`Failed to create workflow: ${errorText}`);
        }
      } catch (error) {
        console.error('[Workflow] Failed to save workflow to database:', error);

        // Check if it's a network error
        const isNetworkError = error instanceof TypeError ||
                               (error instanceof Error && error.message.includes('fetch')) ||
                               !navigator.onLine;

        if (isNetworkError && workflowData) {
          // Add to offline queue
          console.log('[Workflow] Network error detected, adding to offline queue');
          workflowOfflineQueue.addTask(workflowId, workflowData);
        }

        // Fallback to localStorage
        await saveWorkflowToLocal(latestWorkflow);
        throw error;
      } finally {
        setIsSaving(false);
      }
    });
  }, [saveWorkflowToLocal, setWorkflow, myWorkflows]);

  // Load a single workflow from API (or from localStorage in standalone)
  const loadWorkflow = useCallback(async (workflowId: string): Promise<{ workflow: WorkflowState } | null> => {
    try {
      if (isStandalone()) {
        const saved = localStorage.getItem('omniflow_user_data');
        if (!saved) {
          setWorkflow(null);
          return null;
        }
        let workflows: WorkflowState[];
        try {
          workflows = JSON.parse(saved);
          if (!Array.isArray(workflows)) workflows = [];
        } catch {
          setWorkflow(null);
          return null;
        }
        const found = workflows.find((w: WorkflowState) => w.id === workflowId);
        if (!found) {
          setWorkflow(null);
          return null;
        }
        const loadedOutputs: Record<string, any> = {};
        for (const node of found.nodes) {
          const tool = TOOLS.find(t => t.id === node.toolId);
          if (!tool || tool.category !== 'Input') continue;
          const nodeValue = node.data?.value;
          if (!nodeValue) continue;
          const resolveOne = async (path: string): Promise<string | null> => {
            if (typeof path !== 'string') return null;
            if (path.startsWith('local://')) return getLocalFileDataUrl(path);
            return path;
          };
          if (Array.isArray(nodeValue)) {
            const urls = await Promise.all(nodeValue.map((p: string) => resolveOne(p)));
            loadedOutputs[node.id] = urls.filter((u): u is string => u != null);
          } else {
            const url = await resolveOne(nodeValue);
            if (url) loadedOutputs[node.id] = url;
          }
        }
        // Merge loadedOutputs into nodes (outputValue is the single source of truth)
        const nodesWithOutputs = found.nodes.map((n: WorkflowNode) => ({
          ...n,
          outputValue: loadedOutputs[n.id] ?? n.outputValue
        }));
        const workflowWithOutputs = { ...found, nodes: nodesWithOutputs };
        setWorkflow(workflowWithOutputs);
        return { workflow: workflowWithOutputs };
      }

      const response = await apiRequest(`/api/v1/workflow/${workflowId}`);
      if (response.ok) {
        const wf = await response.json();

        const nodes = wf.nodes || [];
        const loadPromises: Promise<unknown>[] = [];
        const loadedOutputs: Record<string, any> = {};

        for (const node of nodes) {
          const tool = TOOLS.find(t => t.id === node.toolId);
          if (!tool) continue;

          if (tool.category === 'Input') {
            const nodeValue = node.data?.value;
            if (!nodeValue) continue;

            const isFilePath = (val: any): boolean => {
              if (typeof val === 'string') {
                return val.startsWith('/api/v1/workflow/') ||
                       val.startsWith('./assets/') ||
                       val.startsWith('http://') ||
                       val.startsWith('https://');
              }
              if (Array.isArray(val)) {
                return val.every(item => typeof item === 'string' && (
                  item.startsWith('/api/v1/workflow/') ||
                  item.startsWith('./assets/') ||
                  item.startsWith('http://') ||
                  item.startsWith('https://')
                ));
              }
              return false;
            };

            if (isFilePath(nodeValue)) {
              if (Array.isArray(nodeValue)) {
                const fileLoadPromises = nodeValue.map((path: string) => {
                  if (path.startsWith('/api/v1/workflow/')) {
                    const match = path.match(/\/api\/v1\/workflow\/([^/]+)\/file\/(.+)$/);
                    if (match) {
                      const [, workflowId, fileId] = match;
                      return getWorkflowFileByFileId(workflowId, fileId);
                    }
                  }
                  return Promise.resolve(path);
                });
                loadPromises.push(
                  Promise.all(fileLoadPromises).then(dataUrls => {
                    loadedOutputs[node.id] = dataUrls.filter((url: any) => url !== null);
                    return dataUrls;
                  }).catch(err => {
                    console.error(`[Workflow] Error loading input files for node ${node.id}:`, err);
                    loadedOutputs[node.id] = nodeValue;
                    return nodeValue;
                  })
                );
              } else {
                if (typeof nodeValue === 'string' && nodeValue.startsWith('/api/v1/workflow/')) {
                  const match = nodeValue.match(/\/api\/v1\/workflow\/([^/]+)\/file\/(.+)$/);
                  if (match) {
                    const [, workflowId, fileId] = match;
                    loadPromises.push(
                      getWorkflowFileByFileId(workflowId, fileId).then(dataUrl => {
                        if (dataUrl) {
                          loadedOutputs[node.id] = dataUrl;
                        } else {
                          loadedOutputs[node.id] = nodeValue;
                        }
                      }).catch(err => {
                        console.error(`[Workflow] Error loading input file for node ${node.id}:`, err);
                        loadedOutputs[node.id] = nodeValue;
                      })
                    );
                  } else {
                    loadedOutputs[node.id] = nodeValue;
                  }
                } else {
                  loadedOutputs[node.id] = nodeValue;
                }
              }
            } else {
              loadedOutputs[node.id] = nodeValue;
            }
            continue;
          }

          if (node.outputValue != null) {
            loadedOutputs[node.id] = node.outputValue;
          }
        }

        if (loadPromises.length > 0) {
          await Promise.race([
            Promise.all(loadPromises),
            new Promise(resolve => setTimeout(resolve, 5000))
          ]);
        }

        const extraInfo = wf.extra_info || {};
        if (extraInfo.active_outputs && typeof extraInfo.active_outputs === 'object') {
          Object.entries(extraInfo.active_outputs).forEach(([nodeId, value]) => {
            if (loadedOutputs[nodeId] == null) {
              loadedOutputs[nodeId] = value;
            }
          });
        }

        let normalizedHistory = normalizeHistoryMap(wf.node_output_history || {});
        if ((!wf.node_output_history || Object.keys(normalizedHistory).length === 0) && wf.data_store?.outputs) {
          const dataStoreOutputs = wf.data_store.outputs;
          const legacyHistory: Record<string, NodeHistoryEntry[]> = {};
          for (const nodeId of Object.keys(dataStoreOutputs)) {
            const portData = dataStoreOutputs[nodeId];
            if (!portData || typeof portData !== 'object') continue;
            const portIds = Object.keys(portData);
            if (portIds.length === 0) continue;
            const histories = portIds.map((pid: string) => {
              const h = portData[pid]?.history;
              return Array.isArray(h) ? h : [];
            });
            const minLen = Math.min(...histories.map(h => h.length));
            if (minLen === 0) continue;
            const entries: NodeHistoryEntry[] = [];
            for (let i = 0; i < minLen; i++) {
              const firstRef = histories[0][i];
              const ts = firstRef?.metadata?.created_at ?? Math.floor(Date.now() / 1000) * 1000;
              const timestamp = typeof ts === 'number' ? (ts < 1e12 ? ts * 1000 : ts) : Date.now();
              const id = `node-${nodeId}-${timestamp}`;
              let output: any;
              if (portIds.length === 1) {
                output = histories[0][i];
              } else {
                output = {} as Record<string, any>;
                portIds.forEach((pid, idx) => {
                  output[pid] = histories[idx][i];
                });
              }
              const entry = createHistoryEntryFromValue({ id, timestamp, value: output });
              if (entry) entries.push(entry);
            }
            if (entries.length > 0) {
              legacyHistory[nodeId] = entries;
            }
          }
          normalizedHistory = normalizeHistoryMap(legacyHistory);
        }

        const createT = wf.create_t ?? wf.create_at;
        const createAtMs = createT != null ? (createT < 1e12 ? createT * 1000 : createT) : undefined;

        const nodesWithOutputs = nodes.map((n: WorkflowNode) => ({
          ...n,
          outputValue: loadedOutputs[n.id] ?? n.outputValue
        }));
        const workflow: WorkflowState = {
          id: wf.workflow_id,
          name: wf.name,
          description: wf.description ?? '',
          nodes: nodesWithOutputs,
          connections: wf.connections || [],
          isDirty: false,
          isRunning: false,
          globalInputs: wf.global_inputs || {},
          nodeOutputHistory: normalizedHistory,
          chatHistory: wf.chat_history || [],
          updatedAt: wf.update_t * 1000,
          createAt: createAtMs,
          visibility: wf.visibility || 'private',
          thumsupCount: wf.thumsup_count ?? 0,
          thumsupLiked: wf.thumsup_liked ?? false,
          userId: wf.user_id,
          tags: wf.tags ?? []
        };
        setWorkflow(workflow);
        return { workflow };
      } else {
        console.warn('[Workflow] Failed to load workflow from API, trying localStorage');
        // Fallback to localStorage：用节点的 outputValue 恢复执行结果预览（不依赖 nodeOutputHistory）
        const saved = localStorage.getItem('omniflow_user_data');
        if (saved) {
          const workflows = JSON.parse(saved);
          const found = workflows.find((w: WorkflowState) => w.id === workflowId);
          if (found) {
            const normalizedFound: WorkflowState = {
              ...found,
              nodeOutputHistory: normalizeHistoryMap(found.nodeOutputHistory)
            };
            setWorkflow(normalizedFound);
            return { workflow: normalizedFound };
          }
        }
        return null;
      }
    } catch (error) {
      console.error('[Workflow] Error loading workflow:', error);
      return null;
    }
  }, [setWorkflow]);

  // Delete workflow
  const deleteWorkflow = useCallback(async (id: string) => {
    workflowSaveQueue.removeTask(id);
    workflowOfflineQueue.removeTask(id);

    if (!isStandalone()) {
      try {
        if (id.startsWith('workflow-') || id.match(/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i)) {
          const response = await apiRequest(`/api/v1/workflow/${id}`, {
            method: 'DELETE'
          });
          if (response.ok) {
            console.log('[Workflow] Workflow deleted from database:', id);
          }
        }
      } catch (error) {
        console.warn('[Workflow] Failed to delete workflow from database, deleting from localStorage:', error);
      }
    }

    setMyWorkflows(prev => {
      const next = prev.filter(w => w.id !== id);
      try {
        localStorage.setItem('omniflow_user_data', JSON.stringify(next));
      } catch (e) {
        console.error('Failed to delete workflow from localStorage:', e);
      }
      return next;
    });
  }, [setMyWorkflows]);

  const updateWorkflowVisibility = useCallback(async (id: string, visibility: 'private' | 'public') => {
    if (!id) return;
    if (visibility !== 'private' && visibility !== 'public') return;

    if (isStandalone()) {
      setMyWorkflows(prev => {
        const next = prev.map(w => w.id === id ? { ...w, visibility } : w);
        try {
          localStorage.setItem('omniflow_user_data', JSON.stringify(next));
        } catch (e) {
          console.error('Failed to save workflows to localStorage:', e);
        }
        return next;
      });
      setWorkflow(prev => prev ? (prev.id === id ? { ...prev, visibility, isDirty: true } : prev) : prev);
      return;
    }

    try {
      const response = await apiRequest(`/api/v1/workflow/${id}/visibility`, {
        method: 'PUT',
        body: JSON.stringify({ visibility })
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Failed to update visibility: ${errorText}`);
      }

      setMyWorkflows(prev => {
        const next = prev.map(w => w.id === id ? { ...w, visibility } : w);
        try {
          localStorage.setItem('omniflow_user_data', JSON.stringify(next));
        } catch (e) {
          console.error('Failed to save workflows to localStorage:', e);
        }
        return next;
      });

      setWorkflow(prev => prev ? (prev.id === id ? { ...prev, visibility, isDirty: true } : prev) : prev);
    } catch (error) {
      console.error('[Workflow] Failed to update workflow visibility:', error);
      throw error;
    }
  }, [setMyWorkflows, setWorkflow]);

  /**
   * 确保当前工作流属于当前用户；若不拥有（预设/404），则先创建新 UUID 并更新状态。
   * 用于 update/保存/自动保存前统一逻辑：不拥有则先建新再操作。
   * @returns 实际应使用的 workflow id（原 id 或新建的 UUID）
   */
  const ensureWorkflowOwned = useCallback(async (current: WorkflowState): Promise<string> => {
    if (isStandalone()) return current.id;
    const { owned } = await checkWorkflowOwnership(current.id, getCurrentUserId());
    if (owned) return current.id;
    const newId = typeof crypto !== 'undefined' && crypto.randomUUID
      ? crypto.randomUUID()
      : `workflow-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
    const MAX_NODE = 20;
    const trimNodeHist = (hist: Record<string, any[]>) => {
      if (!hist || typeof hist !== 'object') return {};
      const out: Record<string, any[]> = {};
      for (const [nodeId, entries] of Object.entries(hist)) {
        if (Array.isArray(entries)) out[nodeId] = entries.slice(0, MAX_NODE);
      }
      return out;
    };
    const createData = {
      name: current.name,
      description: current.description ?? '',
      nodes: current.nodes,
      connections: current.connections,
      workflow_id: newId,
      tags: current.tags ?? [],
      node_output_history: trimNodeHist(current.nodeOutputHistory ?? {})
    };
    const res = await apiRequest('/api/v1/workflow/create', { method: 'POST', body: JSON.stringify(createData) });
    if (!res.ok) {
      const text = await res.text();
      throw new Error(`Failed to create workflow: ${text}`);
    }
    setWorkflow(prev => (prev && prev.id === current.id ? { ...prev, id: newId } : prev));
    if (typeof window !== 'undefined' && window.history?.replaceState) {
      window.history.replaceState(null, '', `#workflow/${newId}`);
    }
    console.log('[Workflow] ensureWorkflowOwned: created new workflow', newId);
    return newId;
  }, [setWorkflow]);

  /** 获取当前最新工作流（用于 AI 等需要“实时”状态的场景，避免闭包拿到旧状态） */
  const getWorkflow = useCallback(() => workflowRef.current, []);

  return {
    myWorkflows,
    workflow,
    setWorkflow,
    getWorkflow,
    setMyWorkflows,
    saveWorkflowToLocal,
    saveWorkflowToLocalOnly,
    saveWorkflowToDatabase,
    loadWorkflow,
    loadWorkflows,
    deleteWorkflow,
    updateWorkflowVisibility,
    ensureWorkflowOwned,
    isSaving,
    isLoading
  };
};
