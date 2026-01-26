import React, { useState, useCallback, useEffect, useRef } from 'react';
import { WorkflowState, NodeStatus } from '../../types';
import { apiRequest } from '../utils/apiClient';
import { getNodeOutputData, getWorkflowFileByFileId } from '../utils/workflowFileManager';
import { TOOLS } from '../../constants';
import { checkWorkflowOwnership, getCurrentUserId } from '../utils/workflowUtils';
import { workflowSaveQueue } from '../utils/workflowSaveQueue';
import { workflowOfflineQueue } from '../utils/workflowOfflineQueue';

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
          env: {
            lightx2v_url: '',
            lightx2v_token: ''
          },
          history: (wf.history_metadata || []).map((meta: any) => ({
            id: meta.run_id || `run-${meta.timestamp}`,
            timestamp: meta.timestamp,
            totalTime: meta.totalTime,
            nodesSnapshot: [],
            outputs: {}
          })),
          updatedAt: wf.update_t * 1000, // Convert to milliseconds
          showIntermediateResults: false
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
  const saveWorkflowToLocal = useCallback((current: WorkflowState) => {
    // Clean input node values: remove base64 data URLs, keep only file paths
    // This prevents storing large base64 data in localStorage
    const cleanedNodes = current.nodes.map(node => {
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

      // Check if it's already a saved path
      const isSavedPath = (val: any): boolean => {
        if (typeof val === 'string') {
          return val.startsWith('./assets/workflow/input') ||
                 val.startsWith('./assets/task/') ||
                 val.startsWith('/assets/workflow/input') ||
                 val.startsWith('/assets/task/') ||
                 val.startsWith('/api/v1/workflow/') ||
                 (val.startsWith('http://') || val.startsWith('https://'));
        }
        if (Array.isArray(val)) {
          return val.every(item => typeof item === 'string' && (
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

    // Clean history to remove base64 data but keep URLs before saving to avoid localStorage quota issues
    const cleanedHistory = current.history.map(run => {
      const cleanedOutputs: Record<string, any> = {};
      // Keep URLs, but remove base64 data (data:image/..., data:video/..., data:audio/...)
      Object.entries(run.outputs || {}).forEach(([nodeId, output]) => {
        if (Array.isArray(output)) {
          cleanedOutputs[nodeId] = output.map((item: any) => {
            if (typeof item === 'string' && item.startsWith('data:')) {
              // Remove base64 data URLs
              return '';
            }
            return item; // Keep URLs (http/https) and other non-base64 data
          }).filter((item: any) => item !== '');
        } else if (typeof output === 'string') {
          if (output.startsWith('data:')) {
            // Remove base64 data URLs
            cleanedOutputs[nodeId] = '';
          } else {
            // Keep regular URLs (http/https)
            cleanedOutputs[nodeId] = output;
          }
        } else {
          cleanedOutputs[nodeId] = output;
        }
      });
      // Only keep outputs that have non-empty values
      Object.keys(cleanedOutputs).forEach(key => {
        if (cleanedOutputs[key] === '' || (Array.isArray(cleanedOutputs[key]) && cleanedOutputs[key].length === 0)) {
          delete cleanedOutputs[key];
        }
      });

      return {
        id: run.id,
        timestamp: run.timestamp,
        totalTime: run.totalTime,
        nodesSnapshot: run.nodesSnapshot,
        outputs: cleanedOutputs // Keep URLs, remove base64 data
      };
    });

    const updated = {
      ...current,
      nodes: cleanedNodes, // Use cleaned nodes (no base64 data)
      updatedAt: Date.now(),
      isDirty: false,
      history: cleanedHistory
    };

    setMyWorkflows(prev => {
      const next = prev.find(w => w.id === updated.id)
        ? prev.map(w => w.id === updated.id ? updated : w)
        : [updated, ...prev];

      try {
        localStorage.setItem('omniflow_user_data', JSON.stringify(next));
      } catch (e: any) {
        if (e.name === 'QuotaExceededError' || e.code === 22) {
          // If still too large, try to clean all workflows' history
          const fullyCleaned = next.map(w => ({
            ...w,
            history: w.history.map(run => {
              const cleanedOutputs: Record<string, any> = {};
              Object.entries(run.outputs || {}).forEach(([nodeId, output]) => {
                if (Array.isArray(output)) {
                  cleanedOutputs[nodeId] = output.map((item: any) => {
                    if (typeof item === 'string' && item.startsWith('data:')) {
                      return '';
                    }
                    return item;
                  }).filter((item: any) => item !== '');
                } else if (typeof output === 'string') {
                  if (!output.startsWith('data:')) {
                    cleanedOutputs[nodeId] = output;
                  }
                } else {
                  cleanedOutputs[nodeId] = output;
                }
              });
              Object.keys(cleanedOutputs).forEach(key => {
                if (cleanedOutputs[key] === '' || (Array.isArray(cleanedOutputs[key]) && cleanedOutputs[key].length === 0)) {
                  delete cleanedOutputs[key];
                }
              });
              return {
                id: run.id,
                timestamp: run.timestamp,
                totalTime: run.totalTime,
                nodesSnapshot: run.nodesSnapshot,
                outputs: cleanedOutputs
              };
            })
          }));
          try {
            localStorage.setItem('omniflow_user_data', JSON.stringify(fullyCleaned));
            return fullyCleaned;
          } catch (e2) {
            console.error('Failed to save workflows even after cleaning:', e2);
            return next;
          }
        } else {
          console.error('Failed to save workflows:', e);
          return next;
        }
      }
      return next;
    });
  }, []);

  // Save workflow to database (and optionally localStorage)
  const saveWorkflowToDatabase = useCallback(async (current: WorkflowState, options?: { name?: string; description?: string; activeOutputs?: Record<string, any> }): Promise<string> => {
    const workflowId = current.id;
    // Allow saving without workflow_id (for new workflows, backend will generate it)

    // 使用保存队列，避免并发保存冲突
    // 注意：保存函数会在执行时重新获取最新的工作流状态，避免状态过期
    return workflowSaveQueue.enqueue(workflowId, async () => {
      setIsSaving(true);
      let workflowData: any = null;

      // 重新获取最新的工作流状态（从 ref 获取，确保是最新的）
      // 优先使用 ref 中的最新状态，如果 ref 中没有或 ID 不匹配，使用传入的 current
      const latestWorkflow = (workflowRef.current && workflowRef.current.id === workflowId)
        ? workflowRef.current
        : current;

      try {
        // 检查工作流是否仍然存在（通过检查是否在 myWorkflows 中）
        // 如果工作流已被删除，跳过保存
        // 注意：对于新创建的工作流（UUID格式），可能不在 myWorkflows 中，这是正常的，不需要查询
        const workflows = myWorkflows;
        const workflowExists = workflows.some(w => w.id === workflowId);

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

        // Clean input node values: save base64 data URLs as files and replace with file paths
        // This prevents storing large base64 data in workflow JSON
        const cleanedNodes = [...latestWorkflow.nodes];

        for (let i = 0; i < cleanedNodes.length; i++) {
          const node = cleanedNodes[i];
          const tool = TOOLS.find(t => t.id === node.toolId);
          if (!tool || tool.category !== 'Input') {
            // Not an input node, skip
            continue;
          }

          const nodeValue = node.data.value;
          if (!nodeValue) {
            continue;
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

          // Check if it's already a saved path
          const isSavedPath = (val: any): boolean => {
            if (typeof val === 'string') {
              return val.startsWith('./assets/workflow/input') ||
                     val.startsWith('./assets/task/') ||
                     val.startsWith('/assets/workflow/input') ||
                     val.startsWith('/assets/task/') ||
                     val.startsWith('/api/v1/workflow/') ||
                     (val.startsWith('http://') || val.startsWith('https://'));
            }
            if (Array.isArray(val)) {
              return val.every(item => typeof item === 'string' && (
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

          // Remove base64 data - it should not be saved to database
          // Input files are kept as base64 in memory and saved only during execution
          if (isDataURL(nodeValue) && !isSavedPath(nodeValue)) {
            // Remove base64 data from node.data.value
            cleanedNodes[i] = {
              ...cleanedNodes[i],
              data: {
                ...cleanedNodes[i].data,
                value: Array.isArray(nodeValue) ? [] : undefined
              }
            };
          }
        }

        // Store active_outputs in extra_info for resuming execution
        const extraInfo: Record<string, any> = {};
        if (options?.activeOutputs) {
          // Convert data URLs to file references for storage efficiency
          const cleanedActiveOutputs: Record<string, any> = {};
          for (const [nodeId, output] of Object.entries(options.activeOutputs)) {
            if (typeof output === 'string' && output.startsWith('data:')) {
              // Keep data URL for now, will be replaced with file reference when file is saved
              cleanedActiveOutputs[nodeId] = output;
            } else {
              cleanedActiveOutputs[nodeId] = output;
            }
          }
          extraInfo.active_outputs = cleanedActiveOutputs;
        }

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
            if (typeof val === 'string') {
              return val.startsWith('./assets/workflow/input') ||
                     val.startsWith('./assets/task/') ||
                     val.startsWith('/assets/workflow/input') ||
                     val.startsWith('/assets/task/') ||
                     val.startsWith('/api/v1/workflow/') ||
                     (val.startsWith('http://') || val.startsWith('https://'));
            }
            if (Array.isArray(val)) {
              return val.every(item => typeof item === 'string' && (
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

          // If it's base64 and not saved, remove it
          if (isDataURL(nodeValue) && !isSavedPath(nodeValue)) {
            return {
              ...nodeData,
              value: Array.isArray(nodeValue) ? [] : undefined
            };
          }

          return nodeData;
        };

        // Clean history_metadata outputs: remove base64 data URLs, keep only file paths/URLs
        // This prevents storing large base64 data in workflow JSON
        const cleanedHistoryMetadata = latestWorkflow.history.map(run => {
          const cleanedOutputs: Record<string, any> = {};
          // Keep URLs and file paths, but remove base64 data (data:image/..., data:video/..., data:audio/...)
          Object.entries(run.outputs || {}).forEach(([nodeId, output]) => {
            if (Array.isArray(output)) {
              cleanedOutputs[nodeId] = output.map((item: any) => {
                if (typeof item === 'string' && item.startsWith('data:')) {
                  // Remove base64 data URLs
                  return '';
                }
                return item; // Keep URLs (http/https) and file paths
              }).filter((item: any) => item !== '');
            } else if (typeof output === 'string') {
              if (output.startsWith('data:')) {
                // Remove base64 data URLs
                cleanedOutputs[nodeId] = '';
              } else {
                // Keep regular URLs (http/https) and file paths
                cleanedOutputs[nodeId] = output;
              }
            } else {
              cleanedOutputs[nodeId] = output;
            }
          });
          // Only keep outputs that have non-empty values
          Object.keys(cleanedOutputs).forEach(key => {
            if (cleanedOutputs[key] === '' || (Array.isArray(cleanedOutputs[key]) && cleanedOutputs[key].length === 0)) {
              delete cleanedOutputs[key];
            }
          });

          // Clean nodes_snapshot: remove base64 data from node.data.value
          const cleanedNodesSnapshot = (run.nodesSnapshot || []).map((node: any) => {
            if (node.data && node.data.value) {
              return {
                ...node,
                data: cleanNodeDataValue(node.data)
              };
            }
            return node;
          });

          return {
            run_id: run.id,
            timestamp: run.timestamp,
            totalTime: run.totalTime,
            node_ids: Object.keys(cleanedOutputs),
            nodes_snapshot: cleanedNodesSnapshot, // Save cleaned nodes snapshot (no base64 data in node.data.value)
            outputs: cleanedOutputs // Save cleaned outputs (no base64 data)
          };
        });

        // 使用最新的工作流状态构建保存数据
        workflowData = {
          name: options?.name || latestWorkflow.name,
          description: options?.description || '',
          nodes: cleanedNodes, // Use cleaned nodes (input nodes with base64 will be kept for now)
          connections: latestWorkflow.connections, // Save connections
          history_metadata: cleanedHistoryMetadata, // Use cleaned history metadata (no base64 data)
          chat_history: latestWorkflow.chatHistory || [],
          extra_info: extraInfo
        };

        // Check if workflow exists in database
        // UUID format (8-4-4-4-12) means it's a saved workflow, try to update first
        const isSavedWorkflow = workflowId && workflowId.match(/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i);
        const isTemporaryId = workflowId && (workflowId.startsWith('temp-') || workflowId.startsWith('flow-'));

        // For saved workflows (UUID), try to update first
        if (isSavedWorkflow) {
          const updateResponse = await apiRequest(`/api/v1/workflow/${workflowId}`, {
            method: 'PUT',
            body: JSON.stringify(workflowData)
          });

          if (updateResponse.ok) {
            console.log('[Workflow] Workflow updated in database:', workflowId);
            saveWorkflowToLocal(latestWorkflow);
            return workflowId;
          } else {
            const errorText = await updateResponse.text();
            // If update fails with 404, workflow doesn't exist - create new one (let backend generate ID)
            if (updateResponse.status === 404) {
              console.log('[Workflow] Workflow not found, creating new one (backend will generate ID)');
              // Don't set workflow_id, let backend generate it
            } else if (updateResponse.status === 400 || updateResponse.status === 403) {
              // Ownership issue - check if it's a preset workflow
              const currentUserId = getCurrentUserId();
              const { isPreset } = await checkWorkflowOwnership(workflowId, currentUserId);

              if (isPreset) {
                console.log('[Workflow] Preset workflow detected, creating new one (backend will generate ID)');
                // Don't set workflow_id, let backend generate it
              } else {
                throw new Error(`Failed to update workflow: ${errorText}`);
              }
            } else {
              throw new Error(`Failed to update workflow: ${errorText}`);
            }
          }
        }

        // Create new workflow (backend will generate workflow_id)
        // Don't set workflow_id for temporary IDs or when update failed
        if (!isTemporaryId && !isSavedWorkflow) {
          // This shouldn't happen, but if it does, let backend generate ID
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

          // Always update workflow with backend-generated ID and cleaned nodes
          const updatedWorkflow = { ...latestWorkflow, id: newWorkflowId, nodes: finalCleanedNodes };
          setWorkflow(updatedWorkflow);
          workflowRef.current = updatedWorkflow;
          saveWorkflowToLocal(updatedWorkflow);
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

        if (isNetworkError) {
          // Add to offline queue
          console.log('[Workflow] Network error detected, adding to offline queue');
          workflowOfflineQueue.addTask(workflowId, workflowData);
        }

        // Fallback to localStorage
        saveWorkflowToLocal(latestWorkflow);
        throw error;
      } finally {
        setIsSaving(false);
      }
    });
  }, [saveWorkflowToLocal, setWorkflow, myWorkflows]);

  // Load a single workflow from API
  const loadWorkflow = useCallback(async (workflowId: string): Promise<{ workflow: WorkflowState; activeOutputs: Record<string, any> } | null> => {
    try {
      const response = await apiRequest(`/api/v1/workflow/${workflowId}`);
      if (response.ok) {
        const wf = await response.json();

        // Restore execution state from nodes (status, error, executionTime are already in nodes)
        // Restore active_outputs from extra_info
        const extraInfo = wf.extra_info || {};
        let activeOutputs = extraInfo.active_outputs || {};

        // Load node output data from data_store.outputs (new unified structure)
        const nodes = wf.nodes || [];
        const loadPromises: Promise<any>[] = [];
        const loadedOutputs: Record<string, any> = { ...activeOutputs };

        for (const node of nodes) {
          // Skip Input nodes - their output is node.data.value, not stored in data_store
          const tool = TOOLS.find(t => t.id === node.toolId);
          if (tool && tool.category === 'Input') {
            // For Input nodes, check node.data.value or data_store.outputs
            // If it's a file path/URL, load it through the backend API
            if (node.data.value) {
              const nodeValue = node.data.value;
              // Check if it's a file path/URL (not base64)
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
                // Load file(s) through backend API (await to ensure files are loaded before returning)
                if (Array.isArray(nodeValue)) {
                  // Multiple files
                  const fileLoadPromises = nodeValue.map((path: string) => {
                    if (path.startsWith('/api/v1/workflow/')) {
                      // Extract file_id from path: /api/v1/workflow/{workflow_id}/file/{file_id}
                      const match = path.match(/\/api\/v1\/workflow\/([^/]+)\/file\/(.+)$/);
                      if (match) {
                        const [, workflowId, fileId] = match;
                        return getWorkflowFileByFileId(workflowId, fileId);
                      }
                    }
                    // For other paths (./assets/, http://, https://), return as-is
                    return Promise.resolve(path);
                  });
                  loadPromises.push(
                    Promise.all(fileLoadPromises).then(dataUrls => {
                      loadedOutputs[node.id] = dataUrls.filter((url: any) => url !== null);
                      return dataUrls;
                    }).catch(err => {
                      console.error(`[Workflow] Error loading input files for node ${node.id}:`, err);
                      // Fallback to original paths if loading fails
                      loadedOutputs[node.id] = nodeValue;
                      return nodeValue;
                    })
                  );
                } else {
                  // Single file
                  if (nodeValue.startsWith('/api/v1/workflow/')) {
                    // Extract file_id from path
                    const match = nodeValue.match(/\/api\/v1\/workflow\/([^/]+)\/file\/(.+)$/);
                    if (match) {
                      const [, workflowId, fileId] = match;
                      loadPromises.push(
                        getWorkflowFileByFileId(workflowId, fileId).then(dataUrl => {
                          if (dataUrl) {
                            loadedOutputs[node.id] = dataUrl;
                          } else {
                            // Fallback to original path if loading fails
                            loadedOutputs[node.id] = nodeValue;
                          }
                        }).catch(err => {
                          console.error(`[Workflow] Error loading input file for node ${node.id}:`, err);
                          // Fallback to original path if loading fails
                          loadedOutputs[node.id] = nodeValue;
                        })
                      );
                    } else {
                      // Invalid path format, use directly
                      loadedOutputs[node.id] = nodeValue;
                    }
                  } else {
                    // For other paths, use directly
                    loadedOutputs[node.id] = nodeValue;
                  }
                }
              } else {
                // Not a file path, use directly (shouldn't happen for saved workflows, but handle it)
                loadedOutputs[node.id] = nodeValue;
              }
            }
            continue;
          }

          // Only load output data for nodes that have been executed
          if (node.status === NodeStatus.SUCCESS || node.status === NodeStatus.ERROR) {
            if (tool && tool.outputs) {
              // Load output data for each output port from data_store.outputs
              for (const port of tool.outputs) {
                // Try to load from data_store.outputs using new interface
                loadPromises.push(
                  getNodeOutputData(wf.workflow_id, node.id, port.id)
                    .then(dataUrl => {
                      if (dataUrl) {
                        // Update activeOutputs with loaded file
                        if (!loadedOutputs[node.id]) {
                          loadedOutputs[node.id] = {};
                        }
                        if (tool.outputs.length === 1) {
                          // Single output node
                          loadedOutputs[node.id] = dataUrl;
                        } else {
                          // Multi-output node
                          if (typeof loadedOutputs[node.id] !== 'object') {
                            loadedOutputs[node.id] = {};
                          }
                          loadedOutputs[node.id][port.id] = dataUrl;
                        }
                      }
                    })
                    .catch(err => {
                      // 404 errors are expected if the node output doesn't exist yet, don't log as error
                      const errorMsg = err instanceof Error ? err.message : String(err);
                      if (!errorMsg.includes('404') && !errorMsg.includes('not found')) {
                        console.warn(`[Workflow] Failed to load node output data for ${node.id}/${port.id}:`, errorMsg);
                      }
                    })
                );
              }
            }
          }
        }

        // Wait for all output data to load (or timeout after 5 seconds)
        await Promise.race([
          Promise.all(loadPromises),
          new Promise(resolve => setTimeout(resolve, 5000))
        ]);

        // Use loaded outputs, fallback to saved activeOutputs if loading failed
        activeOutputs = Object.keys(loadedOutputs).length > 0 ? loadedOutputs : activeOutputs;

        const workflow: WorkflowState = {
          id: wf.workflow_id,
          name: wf.name,
          nodes: nodes, // Contains execution state (status, error, executionTime)
          connections: wf.connections || [],
          isDirty: false,
          isRunning: false,
          globalInputs: wf.global_inputs || {},
          env: {
            lightx2v_url: '',
            lightx2v_token: ''
          },
          history: (wf.history_metadata || []).map((meta: any) => ({
            id: meta.run_id || `run-${meta.timestamp}`,
            timestamp: meta.timestamp,
            totalTime: meta.totalTime,
            nodesSnapshot: [],
            outputs: {}
          })),
          chatHistory: wf.chat_history || [],
          updatedAt: wf.update_t * 1000,
          showIntermediateResults: false
        };
        setWorkflow(workflow);
        return { workflow, activeOutputs };
      } else {
        console.warn('[Workflow] Failed to load workflow from API, trying localStorage');
        // Fallback to localStorage
        const saved = localStorage.getItem('omniflow_user_data');
        if (saved) {
          const workflows = JSON.parse(saved);
          const found = workflows.find((w: WorkflowState) => w.id === workflowId);
          if (found) {
            setWorkflow(found);
            return { workflow: found, activeOutputs: {} };
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
    // 从保存队列中移除相关任务
    workflowSaveQueue.removeTask(id);

    // 从离线队列中移除相关任务
    workflowOfflineQueue.removeTask(id);

    try {
      // Try to delete from database first (check if it's a database ID)
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

    // Also delete from localStorage
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

  return {
    myWorkflows,
    workflow,
    setWorkflow,
    setMyWorkflows,
    saveWorkflowToLocal,
    saveWorkflowToDatabase,
    loadWorkflow,
    loadWorkflows,
    deleteWorkflow,
    isSaving,
    isLoading
  };
};
