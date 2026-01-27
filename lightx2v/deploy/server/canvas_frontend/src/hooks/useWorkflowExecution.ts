import React, { useCallback, useRef } from 'react';
import { WorkflowState, Connection, NodeStatus, GenerationRun } from '../../types';
import { TOOLS } from '../../constants';
import {
  geminiText, geminiImage, geminiSpeech, geminiVideo,
  lightX2VTask, lightX2VTTS, lightX2VVoiceCloneTTS,
  deepseekText, doubaoText, ppchatGeminiText,
  getLightX2VConfigForModel
} from '../../services/geminiService';
import { removeGeminiWatermark } from '../../services/watermarkRemover';
import { useTranslation, Language } from '../i18n/useTranslation';
import { saveNodeOutputData, getNodeOutputData, getWorkflowFileByFileId } from '../utils/workflowFileManager';
import { apiRequest } from '../utils/apiClient';
import { workflowSaveQueue } from '../utils/workflowSaveQueue';
import { workflowOfflineQueue } from '../utils/workflowOfflineQueue';
import { checkWorkflowOwnership, getCurrentUserId } from '../utils/workflowUtils';
import { getAssetPath } from '../utils/assetPath';

interface UseWorkflowExecutionProps {
  workflow: WorkflowState | null;
  setWorkflow: React.Dispatch<React.SetStateAction<WorkflowState | null>>;
  activeOutputs: Record<string, any>;
  setActiveOutputs: React.Dispatch<React.SetStateAction<Record<string, any>>>;
  isPausedRef: React.MutableRefObject<boolean>;
  setIsPaused: (paused: boolean) => void;
  runningTaskIdsRef: React.MutableRefObject<Map<string, string>>;
  abortControllerRef: React.MutableRefObject<AbortController | null>;
  getLightX2VConfig: (workflow: WorkflowState | null) => { url: string; token: string };
  setValidationErrors: (errors: { message: string; type: 'ENV' | 'INPUT' }[]) => void;
  setSelectedRunId: (runId: string | null) => void;
  setGlobalError: (error: { message: string; details?: string } | null) => void;
  updateNodeData: (nodeId: string, key: string, value: any) => void;
  voiceList: {
    lightX2VVoiceList: any;
  };
  lang: Language;
}

export const useWorkflowExecution = ({
  workflow,
  setWorkflow,
  activeOutputs,
  setActiveOutputs,
  isPausedRef,
  setIsPaused,
  runningTaskIdsRef,
  abortControllerRef,
  getLightX2VConfig,
  setValidationErrors,
  setSelectedRunId,
  setGlobalError,
  updateNodeData,
  voiceList,
  lang
}: UseWorkflowExecutionProps) => {
  const { t } = useTranslation(lang);

  const getDescendants = useCallback((nodeId: string, connections: Connection[]): Set<string> => {
    const descendants = new Set<string>();
    const stack = [nodeId];
    while (stack.length > 0) {
      const current = stack.pop()!;
      connections.filter(c => c.sourceNodeId === current).forEach(c => {
        if (!descendants.has(c.targetNodeId)) {
          descendants.add(c.targetNodeId);
          stack.push(c.targetNodeId);
        }
      });
    }
    return descendants;
  }, []);

  const validateWorkflow = useCallback((nodesToRunIds: Set<string>): { message: string; type: 'ENV' | 'INPUT' }[] => {
    if (!workflow) return [];
    const errors: { message: string; type: 'ENV' | 'INPUT' }[] = [];

    const usesLightX2V = Array.from(nodesToRunIds).some(id => {
      const node = workflow.nodes.find(n => n.id === id);
      return node && (node.toolId.includes('lightx2v') || node.toolId.includes('video') || node.toolId === 'avatar-gen' || ((node.toolId === 'text-to-image' || node.toolId === 'image-to-image') && node.data.model?.startsWith('Qwen')));
    });

    if (usesLightX2V) {
      const config = getLightX2VConfig(workflow);
      // 检查是否有 apiClient（在主应用环境中，url 可以为空字符串，使用相对路径）
      const apiClient = (window as any).__API_CLIENT__;
      // 如果有 apiClient，只需要检查 token；否则需要检查 url 和 token
      if (apiClient) {
        // 在主应用环境中，只需要有 token 即可（url 为空字符串表示使用相对路径）
        if (!config.token?.trim()) {
          errors.push({ message: t('missing_env_msg'), type: 'ENV' });
        }
      } else {
        // 独立运行模式，需要 url 和 token
      if (!config.url?.trim() || !config.token?.trim()) {
        errors.push({ message: t('missing_env_msg'), type: 'ENV' });
        }
      }
    }

    workflow.nodes.forEach(node => {
      if (!nodesToRunIds.has(node.id)) return;
      const tool = TOOLS.find(t => t.id === node.toolId);
      if (!tool) return;

      if (tool.category === 'Input') {
        const val = node.data.value;
        const isEmpty = (Array.isArray(val) && val.length === 0) || !val;
        if (isEmpty) {
          errors.push({
            message: `${lang === 'zh' ? tool.name_zh : tool.name} (${t('executing')})`,
            type: 'INPUT'
          });
        }
        return;
      }

      tool.inputs.forEach(port => {
        const isOptional = port.label.toLowerCase().includes('optional') || port.label.toLowerCase().includes('(opt)');
        if (isOptional) return;

        const isConnected = workflow.connections.some(c => c.targetNodeId === node.id && c.targetPortId === port.id);
        const hasGlobalVal = !!workflow.globalInputs[`${node.id}-${port.id}`]?.toString().trim();

        if (!isConnected && !hasGlobalVal) {
          errors.push({
            message: `${lang === 'zh' ? tool.name_zh : tool.name} -> ${port.label}`,
            type: 'INPUT'
          });
        }
      });

      // Special validation for voice clone nodes
      if (node.toolId === 'lightx2v-voice-clone') {
        if (!node.data.speakerId) {
          errors.push({
            message: `${lang === 'zh' ? tool.name_zh : tool.name}: ${t('select_cloned_voice')}`,
            type: 'INPUT'
          });
        }
      }
    });

    return errors;
  }, [workflow, getLightX2VConfig, t, lang]);

  const runWorkflow = useCallback(async (startNodeId?: string, onlyOne?: boolean) => {
    if (!workflow || workflow.isRunning) return;

    // Reset pause state when starting a new workflow
    setIsPaused(false);
    isPausedRef.current = false;

    setSelectedRunId(null);
    const runStartTime = performance.now();
    let nodesToRunIds: Set<string>;
    if (startNodeId) {
      if (onlyOne) nodesToRunIds = new Set([startNodeId]);
      else {
        nodesToRunIds = getDescendants(startNodeId, workflow.connections);
        nodesToRunIds.add(startNodeId);
      }
    } else {
      nodesToRunIds = new Set(workflow.nodes.map(n => n.id));
    }

    // Clear previous errors immediately on rerun
    setWorkflow(prev => prev ? ({
      ...prev,
      nodes: prev.nodes.map(n => nodesToRunIds.has(n.id)
        ? { ...n, status: NodeStatus.IDLE, error: undefined }
        : n)
    }) : null);

    const errors = validateWorkflow(nodesToRunIds);
    if (errors.length > 0) {
      setValidationErrors(errors);
      return;
    }
    setValidationErrors([]);

    const requiresUserApiKey = workflow.nodes
      .filter(n => nodesToRunIds.has(n.id))
      .some(n =>
        n.toolId.includes('video') ||
        n.toolId === 'avatar-gen' ||
        n.data.model === 'gemini-3-pro-image-preview' ||
        n.data.model === 'gemini-2.5-flash-image'
      );

    if (requiresUserApiKey) {
      try {
        if (!(await (window as any).aistudio.hasSelectedApiKey())) {
          await (window as any).aistudio.openSelectKey();
        }
      } catch (err) {}
    }

    setWorkflow(prev => prev ? ({
      ...prev,
      isRunning: true,
      nodes: prev.nodes.map(n => nodesToRunIds.has(n.id) ? { ...n, status: NodeStatus.IDLE, error: undefined, executionTime: undefined, startTime: undefined } : n)
    }) : null);

    const executedInSession = new Set<string>();
    const sessionOutputs: Record<string, any> = {};

    // If running from a specific node, preserve outputs from nodes that won't be re-run
    // Otherwise, clear all outputs for a fresh start
    if (startNodeId) {
      // Preserve outputs from nodes that won't be re-run
      Object.entries(activeOutputs).forEach(([nodeId, val]) => {
        if (!nodesToRunIds.has(nodeId)) sessionOutputs[nodeId] = val;
      });

      // Load intermediate files for nodes that are not being re-run but are needed as inputs
      // Find all nodes that are not in nodesToRunIds but are connected to nodes that will run
      const nodesNeededAsInputs = new Set<string>();
      workflow.connections.forEach(conn => {
        if (nodesToRunIds.has(conn.targetNodeId) && !nodesToRunIds.has(conn.sourceNodeId)) {
          nodesNeededAsInputs.add(conn.sourceNodeId);
        }
      });

      // Load intermediate files for nodes needed as inputs
      if (workflow.id && (workflow.id.startsWith('workflow-') || workflow.id.match(/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i))) {
        const loadPromises: Promise<void>[] = [];

        for (const nodeId of nodesNeededAsInputs) {
          const node = workflow.nodes.find(n => n.id === nodeId);
          if (!node) continue;

          // If we already have the output in activeOutputs, skip loading
          if (sessionOutputs[nodeId] !== undefined) continue;

          const tool = TOOLS.find(t => t.id === node.toolId);
          if (!tool) continue;

          // Skip Input nodes - their output is node.data.value, not stored in data_store
          if (tool.category === 'Input') {
            // For Input nodes, use node.data.value directly
            if (node.data.value) {
              sessionOutputs[nodeId] = node.data.value;
            }
            continue;
          }

          if (!tool.outputs) continue;

          // Load node output data for each output port (using new unified interface)
          for (const port of tool.outputs) {
            loadPromises.push(
              getNodeOutputData(workflow.id, nodeId, port.id)
                .then(dataUrl => {
                  if (dataUrl) {
                    if (tool.outputs.length === 1) {
                      // Single output node
                      sessionOutputs[nodeId] = dataUrl;
                    } else {
                      // Multi-output node
                      if (!sessionOutputs[nodeId] || typeof sessionOutputs[nodeId] !== 'object') {
                        sessionOutputs[nodeId] = {};
                      }
                      sessionOutputs[nodeId][port.id] = dataUrl;
                    }
                  }
                })
                .catch(err => {
                  console.warn(`[WorkflowExecution] Failed to load node output data for ${nodeId}/${port.id}:`, err);
                })
            );
          }
        }

        // Wait for intermediate files to load (with timeout)
        if (loadPromises.length > 0) {
          await Promise.race([
            Promise.all(loadPromises),
            new Promise(resolve => setTimeout(resolve, 3000))
          ]);
        }
      }

      setActiveOutputs(prev => {
        const next = { ...prev, ...sessionOutputs };
        nodesToRunIds.forEach(id => delete next[id]);
        return next;
      });
    } else {
      // Full workflow run: clear all outputs to prevent memory accumulation
      setActiveOutputs({});
    }

    // Input nodes should already have files uploaded (as URLs), not base64 data
    // Files are uploaded immediately when user selects them, so we just use the existing URLs
    // Skip saving input node outputs during execution - they should already be saved during upload
    const runId = `run-${Date.now()}`;

    // Initialize sessionOutputs for input nodes (use their existing values)
    if (workflow.id && (workflow.id.startsWith('workflow-') || workflow.id.match(/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i))) {
      for (const node of workflow.nodes) {
        if (!nodesToRunIds.has(node.id)) continue;

        const tool = TOOLS.find(t => t.id === node.toolId);
        if (!tool || tool.category !== 'Input') continue;

        const nodeValue = node.data.value;
        if (nodeValue) {
          // Use existing value (should already be a file path/URL, not base64)
          sessionOutputs[node.id] = nodeValue;
        }
      }
    }

    // Get LightX2V config once at the start of workflow execution
    const lightX2VConfig = getLightX2VConfig(workflow);

    try {
      // Execute nodes in parallel by layer, with max 3 concurrent executions
      const MAX_CONCURRENT = 3;

      while (executedInSession.size < workflow.nodes.filter(n => nodesToRunIds.has(n.id)).length) {
        // Check if workflow is paused, wait until resumed
        while (isPausedRef.current) {
          await new Promise(resolve => setTimeout(resolve, 100));
          // Check if workflow is still running (might have been stopped)
          const currentWorkflow = workflow;
          if (!currentWorkflow?.isRunning) {
            return;
          }
        }

        // Find all nodes ready to execute (all inputs are ready)
        const readyNodes: typeof workflow.nodes = [];
        for (const node of workflow.nodes) {
          if (!nodesToRunIds.has(node.id) || executedInSession.has(node.id)) continue;
          const tool = TOOLS.find(t => t.id === node.toolId)!;
          const incomingConns = workflow.connections.filter(c => c.targetNodeId === node.id);
          const inputsReady = incomingConns.every(c => !nodesToRunIds.has(c.sourceNodeId) || executedInSession.has(c.sourceNodeId));

          if (inputsReady) {
            readyNodes.push(node);
          }
        }

        // If no nodes are ready, break to avoid infinite loop
        if (readyNodes.length === 0) break;

        // Execute ready nodes in batches of MAX_CONCURRENT
        for (let i = 0; i < readyNodes.length; i += MAX_CONCURRENT) {
          // Check pause state before starting each batch
          while (isPausedRef.current) {
            await new Promise(resolve => setTimeout(resolve, 100));
            const currentWorkflow = workflow;
            if (!currentWorkflow?.isRunning) {
              return;
            }
          }

          const batch = readyNodes.slice(i, i + MAX_CONCURRENT);

          // Execute batch in parallel
          const executionPromises = batch.map(async (node) => {
            const tool = TOOLS.find(t => t.id === node.toolId)!;
            const incomingConns = workflow.connections.filter(c => c.targetNodeId === node.id);
            const nodeStart = performance.now();

            // Update node status to RUNNING
            setWorkflow(prev => prev ? ({ ...prev, nodes: prev.nodes.map(n => n.id === node.id ? { ...n, status: NodeStatus.RUNNING, startTime: nodeStart } : n) }) : null);

            try {
              const nodeInputs: Record<string, any> = {};
              await Promise.all(tool.inputs.map(async (port) => {
                // Check if there's an override value for this port
                if (node.data.inputOverrides && node.data.inputOverrides[port.id] !== undefined) {
                  nodeInputs[port.id] = node.data.inputOverrides[port.id];
                  return;
                }

                const conns = incomingConns.filter(c => c.targetPortId === port.id);
                if (conns.length > 0) {
                  const values = (await Promise.all(conns.map(async (c) => {
                  // First check if source node has executed and has output in sessionOutputs
                  if (sessionOutputs[c.sourceNodeId] !== undefined) {
                    const sourceRes = sessionOutputs[c.sourceNodeId];
                    return (typeof sourceRes === 'object' && sourceRes !== null && c.sourcePortId in sourceRes) ? sourceRes[c.sourcePortId] : sourceRes;
                  }
                  // If not executed yet, check if it's an input node and read from node.data.value
                  // This handles the case where input nodes haven't been executed but their values are needed
                  const sourceNode = workflow.nodes.find(n => n.id === c.sourceNodeId);
                  if (sourceNode) {
                    const sourceTool = TOOLS.find(t => t.id === sourceNode.toolId);
                    if (sourceTool?.category === 'Input') {
                      // For input nodes, read directly from node.data.value
                      let inputValue = sourceNode.data.value;

                      // Convert file paths to base64 data URLs for image and audio inputs
                      if (sourceNode.toolId === 'image-input' && Array.isArray(inputValue) && inputValue.length > 0) {
                        inputValue = await Promise.all(inputValue.map(async (img: string) => {
                          // If it's already a data URL or base64, return as is
                          if (img.startsWith('data:') || (!img.startsWith('http') && img.includes(','))) {
                            return img;
                          }
                          // If it's a file path (starts with /), load and convert to base64
                          if (img.startsWith('/')) {
                            try {
                              // 修复资源路径：如果在 qiankun 环境，确保路径包含 /canvas/
                              let imagePath = img;
                              const basePath = (window as any).__ASSET_BASE_PATH__ || '/canvas';
                              if (img.startsWith('/assets/') && !img.startsWith('/canvas/')) {
                                imagePath = `${basePath}${img}`;
                              }
                              const response = await fetch(imagePath);
                              const blob = await response.blob();
                              return await new Promise<string>((resolve, reject) => {
                                const reader = new FileReader();
                                reader.onloadend = () => resolve(reader.result as string);
                                reader.onerror = reject;
                                reader.readAsDataURL(blob);
                              });
                            } catch (e) {
                              console.error(`Failed to load image ${img}:`, e);
                              return img; // Return original path if loading fails
                            }
                          }
                          return img;
                        }));
                      } else if (sourceNode.toolId === 'audio-input' && inputValue && typeof inputValue === 'string' && inputValue.startsWith('/')) {
                        // Convert audio file path to base64 data URL
                        // 修复资源路径：如果在 qiankun 环境，确保路径包含 /canvas/
                        let audioPath = inputValue;
                        const basePath = (window as any).__ASSET_BASE_PATH__ || '/canvas';
                        if (inputValue.startsWith('/assets/') && !inputValue.startsWith('/canvas/')) {
                          audioPath = `${basePath}${inputValue}`;
                        }
                        try {
                          const response = await fetch(audioPath);
                          const blob = await response.blob();
                          inputValue = await new Promise<string>((resolve, reject) => {
                            const reader = new FileReader();
                            reader.onloadend = () => resolve(reader.result as string);
                            reader.onerror = reject;
                            reader.readAsDataURL(blob);
                          });
                        } catch (e) {
                          console.error(`Failed to load audio ${inputValue}:`, e);
                          // Keep original path if loading fails
                        }
                      }

                      // Check if this is a multi-output node (like text-generation with customOutputs)
                      if (sourceNode.toolId === 'text-generation' && sourceNode.data.customOutputs && typeof inputValue === 'object' && inputValue !== null) {
                        return c.sourcePortId in inputValue ? inputValue[c.sourcePortId] : inputValue;
                      }
                      return inputValue;
                    }
                    // For other nodes that haven't executed, try to read from previously executed outputs
                    // This handles nodes that were executed in previous runs
                    const prevOutput = activeOutputs[c.sourceNodeId];
                    if (prevOutput !== undefined) {
                      return (typeof prevOutput === 'object' && prevOutput !== null && c.sourcePortId in prevOutput) ? prevOutput[c.sourcePortId] : prevOutput;
                    }
              }
              return undefined;
            }))).filter(v => v !== undefined).flat();
              nodeInputs[port.id] = values.length === 1 ? values[0] : values.length > 0 ? values : undefined;
                } else nodeInputs[port.id] = workflow.globalInputs[`${node.id}-${port.id}`];
              }));

              let result: any;
              const model = node.data.model;
              switch (node.toolId) {
                case 'text-input': result = node.data.value || ""; break;
                case 'image-input':
                  const imageValue = node.data.value || [];
                  // For workflow input paths or URLs, use directly (no conversion needed)
                  // Only convert local file paths (starting with /) that are not workflow/task paths
                  if (Array.isArray(imageValue) && imageValue.length > 0) {
                    const convertedImages = await Promise.all(imageValue.map(async (img: string) => {
                      // If it's already a data URL or base64, return as is
                      if (img.startsWith('data:') || (!img.startsWith('http') && img.includes(','))) {
                        return img;
                      }
                      // If it's a workflow input path or task result path, use directly
                      if (img.includes('/assets/workflow/input') || img.includes('/assets/task/') ||
                          img.startsWith('http://') || img.startsWith('https://')) {
                        return img;
                      }
                      // If it's a local file path (starts with /) that's not a workflow/task path, load and convert to base64
                      // This is for backward compatibility with old file paths
                      if (img.startsWith('/')) {
                        try {
                          const response = await fetch(getAssetPath(img));
                          const blob = await response.blob();
                          return await new Promise<string>((resolve, reject) => {
                            const reader = new FileReader();
                            reader.onloadend = () => resolve(reader.result as string);
                            reader.onerror = reject;
                            reader.readAsDataURL(blob);
                          });
                        } catch (e) {
                          console.error(`Failed to load image ${img}:`, e);
                          return img; // Return original path if loading fails
                        }
                      }
                      // For other cases, return as is
                      return img;
                    }));
                    result = convertedImages;
                  } else {
                    result = imageValue;
                  }
                  break;
                case 'audio-input':
                  const audioValue = node.data.value;
                  // For workflow input paths or URLs, use directly (no conversion needed)
                  // Only convert local file paths (starting with /) that are not workflow/task paths
                  if (audioValue && typeof audioValue === 'string') {
                    // If it's already a data URL or base64, return as is
                    if (audioValue.startsWith('data:') || (!audioValue.startsWith('http') && audioValue.includes(','))) {
                      result = audioValue;
                    } else if (audioValue.includes('/assets/workflow/input') || audioValue.includes('/assets/task/') ||
                               audioValue.startsWith('http://') || audioValue.startsWith('https://')) {
                      // If it's a workflow input path or task result path or URL, use directly
                      result = audioValue;
                    } else if (audioValue.startsWith('/')) {
                      // If it's a local file path (starts with /) that's not a workflow/task path, load and convert to base64
                      // This is for backward compatibility with old file paths
                      try {
                        const response = await fetch(getAssetPath(audioValue));
                        const blob = await response.blob();
                        result = await new Promise<string>((resolve, reject) => {
                          const reader = new FileReader();
                          reader.onloadend = () => resolve(reader.result as string);
                          reader.onerror = reject;
                          reader.readAsDataURL(blob);
                        });
                      } catch (e) {
                        console.error(`Failed to load audio ${audioValue}:`, e);
                        result = audioValue; // Return original path if loading fails
                      }
                    } else {
                      // For other cases, return as is
                      result = audioValue;
                    }
                  } else {
                    result = audioValue;
                  }
                  break;
                case 'video-input': result = node.data.value; break;
                case 'web-search': result = await geminiText(nodeInputs['in-text'] || "Search query", true, 'basic', undefined, model); break;
                case 'text-generation':
                  const outputFields = (node.data.customOutputs || []).map((o: any) => ({ id: o.id, description: o.description || o.label }));
                  const useSearch = node.data.useSearch || false;
                  // Use DeepSeek for deepseek models, Doubao for doubao models, PP Chat for ppchat models, otherwise use Gemini
                  if (model && model.startsWith('deepseek-')) {
                    result = await deepseekText(nodeInputs['in-text'] || "...", node.data.mode, node.data.customInstruction, model, outputFields, useSearch);
                  } else if (model && model.startsWith('doubao-')) {
                    const imageInput = nodeInputs['in-image'];
                    result = await doubaoText(nodeInputs['in-text'] || "...", node.data.mode, node.data.customInstruction, model, outputFields, imageInput, useSearch);
                  } else if (model && model.startsWith('ppchat-')) {
                    const imageInput = nodeInputs['in-image'];
                    result = await ppchatGeminiText(nodeInputs['in-text'] || "...", node.data.mode, node.data.customInstruction, model.replace('ppchat-', ''), outputFields, imageInput);
                  } else {
                  result = await geminiText(nodeInputs['in-text'] || "...", false, node.data.mode, node.data.customInstruction, model, outputFields);
                  }
                  break;
                case 'text-to-image':
                  if (model === 'gemini-2.5-flash-image') {
                    result = await geminiImage(nodeInputs['in-text'] || "Artistic portrait", undefined, node.data.aspectRatio, model);
                  } else {
                    // Get config for this specific model (handles -cloud suffix)
                    const t2iModelConfig = getLightX2VConfigForModel(model || 'Qwen-Image-2512');
                    console.log("[LightX2V] Calling lightX2VTask for text-to-image");
                    result = await lightX2VTask(
                      t2iModelConfig.url,
                      t2iModelConfig.token,
                      't2i',
                      model || 'Qwen-Image-2512',
                      nodeInputs['in-text'] || "",
                      undefined, undefined, undefined,
                      'output_image',
                      node.data.aspectRatio,
                      undefined,
                      (taskId) => runningTaskIdsRef.current.set(node.id, taskId),
                      abortControllerRef.current?.signal
                    );
                  }
                  break;
                case 'image-to-image':
                  if (model === 'gemini-2.5-flash-image') {
                    // For Gemini, if multiple images are provided, combine them intelligently
                    const geminiImgs = Array.isArray(nodeInputs['in-image']) ? nodeInputs['in-image'] : (nodeInputs['in-image'] ? [nodeInputs['in-image']] : []);
                    result = await geminiImage(nodeInputs['in-text'] || "Transform", geminiImgs.length > 0 ? geminiImgs : undefined, node.data.aspectRatio || "1:1", model);
                  } else {
                    // For LightX2V i2i, handle multiple images:
                    // - Server supports multiple images via array input
                    // - If multiple images provided, pass all of them to the server
                    // - Server will handle them as input_image_1, input_image_2, etc.
                    const i2iImgs = Array.isArray(nodeInputs['in-image']) ? nodeInputs['in-image'] : (nodeInputs['in-image'] ? [nodeInputs['in-image']] : []);
                    // Pass all images to the server (single image or array of images)
                    const imageInput = i2iImgs.length === 0 ? undefined : (i2iImgs.length === 1 ? i2iImgs[0] : i2iImgs);
                    // Get config for this specific model (handles -cloud suffix)
                    const i2iModelConfig = getLightX2VConfigForModel(model || 'Qwen-Image-Edit-2511');
                    result = await lightX2VTask(
                      i2iModelConfig.url,
                      i2iModelConfig.token,
                      'i2i',
                      model || 'Qwen-Image-Edit-2511',
                      nodeInputs['in-text'] || "",
                      imageInput,
                      undefined,
                      undefined,
                      'output_image',
                      node.data.aspectRatio,
                      undefined,
                      (taskId) => runningTaskIdsRef.current.set(node.id, taskId),
                      abortControllerRef.current?.signal
                    );
                  }
                  break;
                case 'gemini-watermark-remover':
                  const watermarkImg = Array.isArray(nodeInputs['in-image']) ? nodeInputs['in-image'][0] : nodeInputs['in-image'];
                  if (!watermarkImg) throw new Error("Image input is required for watermark removal");
                  result = await removeGeminiWatermark(watermarkImg);
                  break;
                case 'tts':
                  // Determine which service to use based on model
                  const isLightX2V = model === 'lightx2v' || model?.startsWith('lightx2v');

                  if (isLightX2V) {
                    // Use LightX2V TTS
                    const voiceTypeToUse = node.data.voiceType || 'zh_female_vv_uranus_bigtts';
                    let resourceIdToUse = node.data.resourceId;

                    // Always try to match resource_id from voice list to ensure correctness
                    if (voiceList.lightX2VVoiceList?.voices && voiceList.lightX2VVoiceList.voices.length > 0) {
                      const matchingVoice = voiceList.lightX2VVoiceList.voices.find((v: any) => v.voice_type === voiceTypeToUse);
                      if (matchingVoice?.resource_id) {
                        resourceIdToUse = matchingVoice.resource_id;
                        // Update node data with correct resource_id for future use
                        if (!node.data.resourceId || node.data.resourceId !== resourceIdToUse) {
                          updateNodeData(node.id, 'resourceId', resourceIdToUse);
                          console.log(`[LightX2V] Matched resource_id: ${resourceIdToUse} for voice: ${voiceTypeToUse}`);
                        }
                      } else {
                        console.warn(`[LightX2V] No matching voice found for voice_type: ${voiceTypeToUse}`);
                      }
                    } else {
                      console.warn(`[LightX2V] Voice list not loaded, using stored resourceId: ${resourceIdToUse || 'none'}`);
                    }

                    // Fallback to default if still not found
                    if (!resourceIdToUse) {
                      resourceIdToUse = "seed-tts-1.0";
                      console.warn(`[LightX2V] Using fallback resourceId: ${resourceIdToUse}`);
                    }

                    const contextTone = nodeInputs['in-context-tone'] || "";
                  result = await lightX2VTTS(
                      lightX2VConfig.url,
                      lightX2VConfig.token,
                    nodeInputs['in-text'] || "",
                      voiceTypeToUse,
                      contextTone,
                    node.data.emotion || "",
                    node.data.emotionScale || 3,
                    node.data.speechRate || 0,
                    node.data.pitch || 0,
                    node.data.loudnessRate || 0,
                      resourceIdToUse
                    );
                  } else {
                    // Use Gemini TTS
                    const contextTone = nodeInputs['in-context-tone'] || "";
                    result = await geminiSpeech(
                      nodeInputs['in-text'] || "Script",
                      node.data.voice || "Kore",
                      model || 'gemini-2.5-flash-preview-tts',
                      contextTone
                    );
                  }
                  break;
                case 'lightx2v-voice-clone':
                  // Use selected speaker_id from node data
                  const speakerId = node.data.speakerId;

                  if (!speakerId) {
                    throw new Error("Please select a cloned voice. Use the node settings to choose or create a new cloned voice.");
                  }

                  // Generate TTS with cloned voice
                  const ttsText = nodeInputs['in-tts-text'] || nodeInputs['in-text'] || "";
                  if (!ttsText) throw new Error("TTS text is required");
                  result = await lightX2VVoiceCloneTTS(
                    lightX2VConfig.url,
                    lightX2VConfig.token,
                    ttsText,
                    speakerId,
                    node.data.style || "正常",
                    node.data.speed || 1.0,
                    node.data.volume || 0,
                    node.data.pitch || 0,
                    node.data.language || "ZH_CN"
                  );
                  break;
                case 'video-gen-text':
                  // Get config for this specific model (handles -cloud suffix)
                  const t2vModelConfig = getLightX2VConfigForModel(model || 'Wan2.2_T2V_A14B_distilled');
                  result = await lightX2VTask(
                      t2vModelConfig.url,
                      t2vModelConfig.token,
                    't2v',
                    model || 'Wan2.2_T2V_A14B_distilled',
                    nodeInputs['in-text'] || "",
                    undefined, undefined, undefined,
                    'output_video',
                    node.data.aspectRatio,
                    undefined,
                    (taskId) => runningTaskIdsRef.current.set(node.id, taskId),
                    abortControllerRef.current?.signal
                  );
                  break;
                case 'video-gen-image':
                  const startImg = Array.isArray(nodeInputs['in-image']) ? nodeInputs['in-image'][0] : nodeInputs['in-image'];
                  // Get config for this specific model (handles -cloud suffix)
                  const i2vModelConfig = getLightX2VConfigForModel(model || 'Wan2.2_I2V_A14B_distilled');
                  result = await lightX2VTask(
                    i2vModelConfig.url,
                    i2vModelConfig.token,
                    'i2v',
                    model || 'Wan2.2_I2V_A14B_distilled',
                    nodeInputs['in-text'] || "",
                    startImg,
                    undefined, undefined,
                    'output_video',
                    node.data.aspectRatio,
                    undefined,
                    (taskId) => runningTaskIdsRef.current.set(node.id, taskId),
                    abortControllerRef.current?.signal
                  );
                  break;
                case 'video-gen-dual-frame':
                    const dualStart = Array.isArray(nodeInputs['in-image-start']) ? nodeInputs['in-image-start'][0] : nodeInputs['in-image-start'];
                    const dualEnd = Array.isArray(nodeInputs['in-image-end']) ? nodeInputs['in-image-end'][0] : nodeInputs['in-image-end'];
                    // Get config for this specific model (handles -cloud suffix)
                    const flf2vModelConfig = getLightX2VConfigForModel(model || 'Wan2.2_I2V_A14B_distilled');
                    result = await lightX2VTask(
                        flf2vModelConfig.url,
                        flf2vModelConfig.token,
                        'flf2v',
                        model || 'Wan2.2_I2V_A14B_distilled',
                        nodeInputs['in-text'] || "",
                        dualStart,
                        undefined,
                        dualEnd,
                        'output_video',
                        node.data.aspectRatio,
                        undefined,
                        (taskId) => runningTaskIdsRef.current.set(node.id, taskId),
                        abortControllerRef.current?.signal
                    );
                    break;
                case 'character-swap':
                  const swapImg = Array.isArray(nodeInputs['in-image']) ? nodeInputs['in-image'][0] : nodeInputs['in-image'];
                  const swapVid = Array.isArray(nodeInputs['in-video']) ? nodeInputs['in-video'][0] : nodeInputs['in-video'];

                  // Use LightX2V animate task for wan2.2_animate model, otherwise use Gemini
                  if (model === 'wan2.2_animate' || model?.endsWith('-cloud')) {
                    // Get config for this specific model (handles -cloud suffix)
                    const animateModelConfig = getLightX2VConfigForModel(model);
                    result = await lightX2VTask(
                      animateModelConfig.url,
                      animateModelConfig.token,
                      'animate',
                      model,
                      nodeInputs['in-text'] || "",
                      swapImg,
                      undefined, undefined,
                      'output_video',
                      node.data.aspectRatio,
                      swapVid,
                      (taskId) => runningTaskIdsRef.current.set(node.id, taskId),
                      abortControllerRef.current?.signal
                    );
                  } else {
                  result = await geminiVideo(nodeInputs['in-text'] || "Swap character", swapImg, "16:9", "720p", swapVid, model);
                  }
                  break;
                case 'avatar-gen':
                  const avatarImg = Array.isArray(nodeInputs['in-image']) ? nodeInputs['in-image'][0] : nodeInputs['in-image'];
                  const avatarAudio = Array.isArray(nodeInputs['in-audio']) ? nodeInputs['in-audio'][0] : nodeInputs['in-audio'];
                  // Get config for this specific model (handles -cloud suffix)
                  const s2vModelConfig = getLightX2VConfigForModel(model || "SekoTalk");
                  result = await lightX2VTask(
                    s2vModelConfig.url,
                    s2vModelConfig.token,
                    's2v',
                    model || "SekoTalk",
                    nodeInputs['in-text'] || "A person talking naturally.",
                    avatarImg || "",
                    avatarAudio || "",
                    undefined,
                    'output_video',
                    undefined,
                    undefined,
                    (taskId) => runningTaskIdsRef.current.set(node.id, taskId),
                    abortControllerRef.current?.signal
                  );
                  break;
                default: result = "Processed";
              }
              const nodeDuration = performance.now() - nodeStart;

            // Store result in sessionOutputs (need to handle race condition)
            // Use a function to safely update sessionOutputs
            return { nodeId: node.id, result, duration: nodeDuration };
            } catch (err: any) {
              const nodeDuration = performance.now() - nodeStart;
              if (err.message?.includes("Requested entity was not found")) {
                await (window as any).aistudio.openSelectKey();
              }
              setWorkflow(prev => prev ? ({
                ...prev,
                nodes: prev.nodes.map(n => n.id === node.id ? {
                  ...n,
                  status: NodeStatus.ERROR,
                  error: err.message || 'Unknown execution error',
                  executionTime: nodeDuration,
                  completedAt: Date.now()
                } : n)
              }) : null);
            throw { nodeId: node.id, error: err, duration: nodeDuration };
          }
          });

          // Wait for all nodes in this batch to complete
          const results = await Promise.allSettled(executionPromises);

          // Process results and update state - batch updates to reduce re-renders
          const successfulResults: Array<{ nodeId: string; result: any; duration: number }> = [];
          const failedNodes: Array<{ nodeId: string; error: any; duration: number }> = [];

          results.forEach((settledResult, index) => {
            const node = batch[index];
            if (settledResult.status === 'fulfilled') {
              const { nodeId, result, duration } = settledResult.value;
              sessionOutputs[nodeId] = result;
              setActiveOutputs(prev => ({ ...prev, [nodeId]: result }));
              // Update node with result and execution state - IMPORTANT: set outputValue so it can be saved
              setWorkflow(prev => prev ? ({
                ...prev,
                nodes: prev.nodes.map(n => n.id === nodeId ? {
                  ...n,
                  status: NodeStatus.SUCCESS,
                  executionTime: duration,
                  outputValue: result, // Set outputValue so it can be saved
                  completedAt: Date.now()
                } : n)
              }) : null);
              executedInSession.add(nodeId);

              // Note: Immediate save is skipped here - we'll save all outputs together after execution completes
              // This ensures we have the correct outputValue set and can save with runId
            } else {
              const errorInfo = settledResult.reason;
              if (errorInfo && errorInfo.error) {
                // Error was already handled in the catch block, just mark as executed
                executedInSession.add(errorInfo.nodeId);
                // Ensure error is a string
                const errorMessage = errorInfo.error instanceof Error
                  ? errorInfo.error.message
                  : (typeof errorInfo.error === 'string'
                      ? errorInfo.error
                      : String(errorInfo.error || 'Unknown execution error'));
                failedNodes.push({ nodeId: errorInfo.nodeId, error: errorMessage, duration: errorInfo.duration || 0 });
              } else {
                // Unhandled error
                const nodeDuration = performance.now() - (node.startTime || performance.now());
                const errorMessage = settledResult.reason instanceof Error
                  ? settledResult.reason.message
                  : (typeof settledResult.reason === 'string'
                      ? settledResult.reason
                      : 'Unknown execution error');
                failedNodes.push({ nodeId: node.id, error: errorMessage, duration: nodeDuration });
                executedInSession.add(node.id);
              }
            }
          });

          // Batch update state for successful results
          if (successfulResults.length > 0) {
            setActiveOutputs(prev => {
              const next = { ...prev };
              successfulResults.forEach(({ nodeId, result }) => {
                next[nodeId] = result;
              });
              return next;
            });

            setWorkflow(prev => {
              if (!prev) return null;
              const updatedNodes = [...prev.nodes];
              successfulResults.forEach(({ nodeId, duration }) => {
                const index = updatedNodes.findIndex(n => n.id === nodeId);
                if (index >= 0) {
                  updatedNodes[index] = { ...updatedNodes[index], status: NodeStatus.SUCCESS, executionTime: duration };
                }
              });
              return { ...prev, nodes: updatedNodes };
            });
          }

          // Batch update state for failed nodes
          if (failedNodes.length > 0) {
            setWorkflow(prev => {
              if (!prev) return null;
              const updatedNodes = [...prev.nodes];
              failedNodes.forEach(({ nodeId, error, duration }) => {
                const index = updatedNodes.findIndex(n => n.id === nodeId);
                if (index >= 0) {
                  updatedNodes[index] = {
                    ...updatedNodes[index],
                    status: NodeStatus.ERROR,
                    error,
                    executionTime: duration,
                    completedAt: updatedNodes[index].completedAt ?? Date.now()
                  };
                }
              });
              return { ...prev, nodes: updatedNodes };
            });
          }
        }
      }
      const runTotalTime = performance.now() - runStartTime;

      // Generate runId first (will be used for saving outputs and creating history)
      const runId = `run-${Date.now()}`;

      // Optimize history storage: only keep essential data, limit history size
      // Create a lightweight snapshot without deep copying all node data
      const lightweightNodesSnapshot = workflow.nodes.map(n => ({
        id: n.id,
        toolId: n.toolId,
        x: n.x,
        y: n.y,
        status: n.status,
        data: { ...n.data }, // Shallow copy of data
        error: n.error,
        executionTime: n.executionTime,
        completedAt: n.completedAt
      }));

      // Optimize history outputs: don't save full base64 data, save references instead
      // For data URLs, we'll save references (file_id or data_id) if available
      // For text, we can save directly (small size)
      const optimizedOutputs: Record<string, any> = {};

      for (const [nodeId, output] of Object.entries(sessionOutputs)) {
        if (typeof output === 'string' && output.startsWith('data:')) {
          // Data URL - save as reference (will be replaced with data_id after save)
          // For now, mark it as a data URL that needs to be saved
          optimizedOutputs[nodeId] = {
            type: 'data_url',
            data: output.substring(0, 100) + '...', // Truncate for memory efficiency
            _full_data: output // Keep full data temporarily for saving
          };
        } else if (typeof output === 'string') {
          // Plain text - save directly (small size)
          optimizedOutputs[nodeId] = {
            type: 'text',
            data: output
          };
        } else if (typeof output === 'object' && output !== null) {
          // JSON object - save directly (usually small)
          optimizedOutputs[nodeId] = {
            type: 'json',
            data: output
          };
        } else {
          // Other types
          optimizedOutputs[nodeId] = output;
        }
      }

      const newRun: GenerationRun = {
        id: runId,
        timestamp: Date.now(),
        outputs: optimizedOutputs, // Use optimized outputs (references instead of full data)
        nodesSnapshot: lightweightNodesSnapshot,
        totalTime: runTotalTime
      };

      // Save output files to database (if workflow has a database ID)
      if (workflow.id && (workflow.id.startsWith('workflow-') || workflow.id.match(/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i))) {
        try {
          console.log(`[WorkflowExecution] Starting to save outputs for workflow ${workflow.id}, sessionOutputs keys:`, Object.keys(sessionOutputs));
          // Save output files for each node (async, don't block execution)
          // Pass runId to associate saves with workflow history
          const savePromises: Promise<{ file_id?: string; data_id: string; file_url?: string; url?: string } | null>[] = [];
          const saveMeta: Array<{ nodeId: string; portId: string; value: any; kind: 'single' | 'multi' | 'array'; index?: number }> = [];

          for (const [nodeId, output] of Object.entries(sessionOutputs)) {
            const node = workflow.nodes.find(n => n.id === nodeId);
            if (!node) {
              console.warn(`[WorkflowExecution] Node ${nodeId} not found in workflow.nodes`);
              continue;
            }

            const tool = TOOLS.find(t => t.id === node.toolId);
            if (!tool || !tool.outputs) {
              console.warn(`[WorkflowExecution] Tool ${node.toolId} not found or has no outputs`);
              continue;
            }

            // Use node.outputValue if available (more reliable), otherwise fall back to sessionOutputs
            const outputToSave = node.outputValue !== undefined ? node.outputValue : output;
            console.log(`[WorkflowExecution] Node ${nodeId} (${node.toolId}) outputToSave type:`, typeof outputToSave, outputToSave ? (typeof outputToSave === 'string' ? (outputToSave.length > 100 ? outputToSave.substring(0, 100) + '...' : outputToSave) : 'object/array') : 'empty');

            // Check if outputToSave is empty or invalid
            if (!outputToSave || (typeof outputToSave === 'string' && outputToSave.length === 0)) {
              console.warn(`[WorkflowExecution] Node ${nodeId} (${node.toolId}) has no output to save, skipping`);
              continue;
            }

            // Handle different output formats - save ALL types of outputs (data URLs, text, JSON, task result URLs)
            if (outputToSave && typeof outputToSave === 'object' && !Array.isArray(outputToSave)) {
              // Multi-output node (outputValue is a Record<portId, value>)
              for (const [portId, value] of Object.entries(outputToSave)) {
                // Save all types: data URLs, text, JSON, task result URLs
                if ((typeof value === 'string' && value.length > 0) || (typeof value === 'object' && value !== null && !Array.isArray(value))) {
                  console.log(`[WorkflowExecution] Saving multi-output node ${nodeId}/${portId}:`, typeof value === 'string' ? (value.length > 100 ? value.substring(0, 100) + '...' : value) : 'object');
                  saveMeta.push({ nodeId, portId, value, kind: 'multi' });
                  savePromises.push(
                    saveNodeOutputData(workflow.id, nodeId, portId, value, runId)
                      .then(result => {
                        if (result) {
                          console.log(`[WorkflowExecution] ✓ Saved node output data for ${nodeId}/${portId}:`, result);
                        } else {
                          console.warn(`[WorkflowExecution] ✗ Save returned null for ${nodeId}/${portId}`);
                        }
                        return result;
                      })
                      .catch(err => {
                        console.error(`[WorkflowExecution] ✗ Error saving node output data for ${nodeId}/${portId}:`, err);
                        // Don't return null, throw to be caught by Promise.allSettled
                        throw err;
                      })
                  );
                }
              }
            } else if (typeof outputToSave === 'string' && outputToSave.length > 0) {
              // Single output node - use first output port (save data URLs, plain text, and task result URLs, including CDN URLs)
              const firstOutputPort = tool.outputs[0];
              if (firstOutputPort) {
                const isUrl = outputToSave.startsWith('http://') || outputToSave.startsWith('https://') || outputToSave.startsWith('./assets/');
                console.log(`[WorkflowExecution] Saving single-output node ${nodeId}/${firstOutputPort.id} (${isUrl ? 'URL' : 'text'}):`, outputToSave.length > 100 ? outputToSave.substring(0, 100) + '...' : outputToSave);
                saveMeta.push({ nodeId, portId: firstOutputPort.id, value: outputToSave, kind: 'single' });
                savePromises.push(
                  saveNodeOutputData(workflow.id, nodeId, firstOutputPort.id, outputToSave, runId)
                    .then(result => {
                      if (result) {
                        console.log(`[WorkflowExecution] ✓ Saved node output data for ${nodeId}/${firstOutputPort.id}:`, result);
                      } else {
                        console.warn(`[WorkflowExecution] ✗ Save returned null for ${nodeId}/${firstOutputPort.id}`);
                      }
                      return result;
                    })
                    .catch(err => {
                      console.error(`[WorkflowExecution] ✗ Error saving node output data for ${nodeId}/${firstOutputPort.id}:`, err);
                      // Don't return null, throw to be caught by Promise.allSettled
                      throw err;
                    })
                );
              } else {
                console.warn(`[WorkflowExecution] Node ${nodeId} (${node.toolId}) has no output ports defined`);
              }
            } else if (typeof outputToSave === 'object' && outputToSave !== null && !Array.isArray(outputToSave)) {
              // JSON object output
              const firstOutputPort = tool.outputs[0];
              if (firstOutputPort) {
                saveMeta.push({ nodeId, portId: firstOutputPort.id, value: outputToSave, kind: 'single' });
                savePromises.push(
                  saveNodeOutputData(workflow.id, nodeId, firstOutputPort.id, outputToSave, runId)
                    .then(result => {
                      if (result) {
                        console.log(`[WorkflowExecution] Saved node output data for ${nodeId}/${firstOutputPort.id}:`, result);
                      }
                      return result;
                    })
                    .catch(err => {
                      console.error(`[WorkflowExecution] Error saving node output data for ${nodeId}/${firstOutputPort.id}:`, err);
                      return null;
                    })
                );
              }
            } else if (Array.isArray(outputToSave)) {
              // Array output - save each item (data URLs, text, or JSON)
              const firstOutputPort = tool.outputs[0];
              if (firstOutputPort) {
                for (let i = 0; i < outputToSave.length; i++) {
                  const item = outputToSave[i];
                  const portId = tool.outputs.length > 1 ? tool.outputs[i]?.id || firstOutputPort.id : firstOutputPort.id;
                  // Save all types in array
                  if ((typeof item === 'string' && item.length > 0) || (typeof item === 'object' && item !== null)) {
                    saveMeta.push({ nodeId, portId, value: item, kind: 'array', index: i });
                    savePromises.push(
                      saveNodeOutputData(workflow.id, nodeId, portId, item, runId)
                        .then(result => {
                          if (result) {
                            console.log(`[WorkflowExecution] Saved node output data for ${nodeId}/${portId}:`, result);
                          }
                          return result;
                        })
                        .catch(err => {
                          console.error(`[WorkflowExecution] Error saving node output data for ${nodeId}/${portId}:`, err);
                          return null;
                        })
                    );
                  }
                }
              }
            }
          }

          // Wait for all saves to complete and update history outputs with data_id references
          Promise.allSettled(savePromises).then(results => {
            // Check for any failed saves
            const failedSaves: string[] = [];
            results.forEach((result, index) => {
              if (result.status === 'rejected') {
                const errorMsg = result.reason instanceof Error ? result.reason.message : String(result.reason);
                console.error(`[WorkflowExecution] Save promise ${index} failed:`, errorMsg);
                failedSaves.push(`Save ${index + 1}: ${errorMsg}`);
              }
            });

            if (failedSaves.length > 0) {
              setGlobalError({
                message: `Failed to save ${failedSaves.length} node output(s)`,
                details: failedSaves.join('\n')
              });
            }

            // Update history outputs with data_id references (replace data URLs with references)
            const updatedOutputs: Record<string, any> = { ...optimizedOutputs };
            const outputReplacements: Array<{ nodeId: string; portId: string; value: any; kind: 'single' | 'multi' | 'array'; index?: number }> = [];

            const isDataUrl = (value: any) => typeof value === 'string' && value.startsWith('data:');

            results.forEach((result, idx) => {
              if (result.status !== 'fulfilled' || !result.value) return;
              const meta = saveMeta[idx];
              if (!meta) return;
              const saveResult = result.value;
              const fileUrl = saveResult.file_url || saveResult.url;
              const shouldReplaceWithUrl = fileUrl && isDataUrl(meta.value);

              if (meta.kind === 'multi') {
                if (!updatedOutputs[meta.nodeId] || typeof updatedOutputs[meta.nodeId] !== 'object') {
                  updatedOutputs[meta.nodeId] = {};
                }
                updatedOutputs[meta.nodeId][meta.portId] = shouldReplaceWithUrl
                  ? { type: 'url', data_id: saveResult.data_id, data: fileUrl }
                  : { type: 'reference', data_id: saveResult.data_id, file_id: saveResult.file_id };
              } else if (meta.kind === 'single') {
                if (typeof meta.value === 'string' && meta.value.startsWith('data:')) {
                  updatedOutputs[meta.nodeId] = shouldReplaceWithUrl
                    ? { type: 'url', data_id: saveResult.data_id, data: fileUrl }
                    : { type: 'reference', data_id: saveResult.data_id, file_id: saveResult.file_id };
                } else if (typeof meta.value === 'string') {
                  const isTaskResultUrl = meta.value.startsWith('./assets/task/result') ||
                                         meta.value.startsWith('http://') ||
                                         meta.value.startsWith('https://');
                  updatedOutputs[meta.nodeId] = isTaskResultUrl
                    ? { type: 'url', data_id: saveResult.data_id, data: meta.value }
                    : { type: 'text', data_id: saveResult.data_id, data: meta.value };
                } else {
                  updatedOutputs[meta.nodeId] = { type: 'json', data_id: saveResult.data_id, data: meta.value };
                }
              } else if (meta.kind === 'array') {
                if (!Array.isArray(updatedOutputs[meta.nodeId])) {
                  updatedOutputs[meta.nodeId] = [];
                }
                const entry = shouldReplaceWithUrl
                  ? { type: 'url', data_id: saveResult.data_id, data: fileUrl }
                  : { type: 'reference', data_id: saveResult.data_id, file_id: saveResult.file_id };
                (updatedOutputs[meta.nodeId] as any[])[meta.index ?? 0] = entry;
              }

              if (shouldReplaceWithUrl) {
                outputReplacements.push({
                  nodeId: meta.nodeId,
                  portId: meta.portId,
                  value: fileUrl,
                  kind: meta.kind,
                  index: meta.index
                });
              }
            });

            // Update the run with optimized outputs
            setWorkflow(prev => {
              if (!prev) return null;
              const updatedHistory = prev.history.map(run =>
                run.id === runId ? { ...run, outputs: updatedOutputs } : run
              );
              const updatedNodes = prev.nodes.map(node => {
                const updates = outputReplacements.filter(r => r.nodeId === node.id);
                if (updates.length === 0) return node;
                if (node.outputValue && typeof node.outputValue === 'object' && !Array.isArray(node.outputValue)) {
                  const nextOutputValue = { ...node.outputValue };
                  updates.forEach(update => {
                    if (update.kind === 'multi') {
                      nextOutputValue[update.portId] = update.value;
                    }
                  });
                  return { ...node, outputValue: nextOutputValue };
                }
                if (Array.isArray(node.outputValue)) {
                  const nextArray = [...node.outputValue];
                  updates.forEach(update => {
                    if (update.kind === 'array') {
                      nextArray[update.index ?? 0] = update.value;
                    }
                  });
                  return { ...node, outputValue: nextArray };
                }
                const singleUpdate = updates.find(update => update.kind === 'single');
                return singleUpdate ? { ...node, outputValue: singleUpdate.value } : node;
              });
              return { ...prev, history: updatedHistory, nodes: updatedNodes };
            });

            if (outputReplacements.length > 0) {
              setActiveOutputs(prev => {
                const next = { ...prev };
                outputReplacements.forEach(update => {
                  if (update.kind === 'multi') {
                    const existing = next[update.nodeId];
                    next[update.nodeId] = {
                      ...(existing && typeof existing === 'object' && !Array.isArray(existing) ? existing : {}),
                      [update.portId]: update.value
                    };
                  } else if (update.kind === 'array') {
                    const existing = Array.isArray(next[update.nodeId]) ? [...next[update.nodeId]] : [];
                    existing[update.index ?? 0] = update.value;
                    next[update.nodeId] = existing;
                  } else {
                    next[update.nodeId] = update.value;
                  }
                });
                return next;
              });
            }
          }).catch(err => {
            const errorMsg = err instanceof Error ? err.message : String(err);
            console.error('[WorkflowExecution] Error processing save results:', errorMsg);
            setGlobalError({
              message: 'Failed to process save results',
              details: errorMsg
            });
            // Still update history even if some saves failed
            setWorkflow(prev => {
              if (!prev) return null;
              return { ...prev, history: [newRun, ...prev.history].slice(0, 5) };
            });
          });
        } catch (error) {
          console.error('[WorkflowExecution] Error initiating output file saves:', error);
          setGlobalError({
            message: 'Failed to initiate output file saves',
            details: error instanceof Error ? error.message : String(error)
          });
          // Don't fail the workflow execution if file saving fails
        }
      } else {
        console.warn(`[WorkflowExecution] Workflow ID ${workflow?.id} is not a valid database ID, skipping save`);
      }

      // Note: History will be updated after saves complete (see Promise.all above)
      // For now, add the run with optimized outputs (will be updated with data_id references later)
      setWorkflow(prev => prev ? ({ ...prev, history: [newRun, ...prev.history].slice(0, 5) }) : null);
      setSelectedRunId(newRun.id);

      // Save execution state (nodes with status, activeOutputs) to database
      // Capture values before async operations to avoid closure issues
      const workflowId = workflow.id;
      const shouldSave = workflowId && (workflowId.startsWith('workflow-') || workflowId.match(/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i));

      if (shouldSave) {
        // Capture current state before async operation to avoid closure issues
        // IMPORTANT: Use a function to get the latest workflow state after all setWorkflow calls
        // This ensures we capture the updated node values (including saved file paths for input nodes)
        const getLatestWorkflowState = () => {
          // This will be called after setTimeout, so workflow state should be updated
          // But we need to use a ref or state getter to get the latest value
          // For now, we'll use workflow directly, but note that input node values should be updated
          return workflow;
        };

        const currentActiveOutputs = { ...sessionOutputs };
        // Capture newRun to include it in history_metadata when saving
        const capturedNewRun = { ...newRun };
        // Capture current history (before adding newRun) and manually add newRun for saving
        const currentHistory = workflow.history || [];
        const historyToSave = [capturedNewRun, ...currentHistory].slice(0, 5);

        // Build nodes state from session execution results
        // IMPORTANT: Use the latest workflow state to ensure input node values are updated
        // Input nodes may have been updated with saved file paths during execution
        // We need to capture the latest state after all setWorkflow calls
        const nodesToSave = workflow.nodes.map(node => {
          const wasExecuted = sessionOutputs[node.id] !== undefined;
          if (wasExecuted) {
            // Node was executed in this session, use the updated state
            // Note: Input nodes may have been updated with saved file paths
            const updatedNode = workflow.nodes.find(n => n.id === node.id);
            return {
              id: node.id,
              toolId: node.toolId,
              x: node.x,
              y: node.y,
              status: updatedNode?.status || node.status,
              data: updatedNode?.data || node.data, // Use updated data (may include saved file paths for input nodes)
              error: updatedNode?.error,
              executionTime: updatedNode?.executionTime,
              outputValue: updatedNode?.outputValue || node.outputValue
            };
          } else {
            // Node was not executed, keep existing state
            return {
              id: node.id,
              toolId: node.toolId,
              x: node.x,
              y: node.y,
              status: node.status,
              data: node.data,
              error: node.error,
              executionTime: node.executionTime,
              outputValue: node.outputValue
            };
          }
        });

        // Save execution state via API (using save queue to avoid conflicts)
        // Use setTimeout to defer and ensure state updates are complete
        setTimeout(async () => {
          try {
            const currentUserId = getCurrentUserId();
            const { isPreset } = await checkWorkflowOwnership(workflowId, currentUserId);

            // Use save queue to avoid conflicts with manual saves
            await workflowSaveQueue.enqueue(workflowId, async () => {
              if (isPreset) {
                // If preset workflow, create new workflow first
                try {
                  // Create new workflow with current structure
                  // Create new workflow with the existing workflow_id (if it's a UUID)
                // This ensures the frontend and backend use the same workflow_id
                const createData: any = {
                  name: workflow.name,
                  description: '',
                  nodes: nodesToSave,
                  connections: workflow.connections, // Save connections
                  history_metadata: historyToSave.map(run => ({
                    run_id: run.id,
                    timestamp: run.timestamp,
                    totalTime: run.totalTime,
                    node_ids: Object.keys(run.outputs || {}),
                    nodes_snapshot: run.nodesSnapshot || [], // Save nodes snapshot
                    outputs: run.outputs || {} // Save optimized outputs (references instead of full data)
                  })),
                  chat_history: workflow.chatHistory || [],
                  extra_info: {
                    active_outputs: currentActiveOutputs
                  }
                };

                // If workflow has a UUID, pass it to backend to use the same ID
                if (workflowId && workflowId.match(/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i)) {
                  createData.workflow_id = workflowId;
                }

                const createResponse = await apiRequest('/api/v1/workflow/create', {
                    method: 'POST',
                    body: JSON.stringify(createData)
                  });

                  if (createResponse.ok) {
                    const data = await createResponse.json();
                    const finalWorkflowId = data.workflow_id;
                    console.log('[WorkflowExecution] Created new workflow for preset:', finalWorkflowId);

                    // Only update workflow state and URL if the ID changed
                    // If we passed workflow_id and backend used it, finalWorkflowId should match workflowId
                    if (finalWorkflowId !== workflowId) {
                      // Update workflow state with new ID
                      setWorkflow(prev => prev ? { ...prev, id: finalWorkflowId } : null);

                      // Update URL
                      if (window.history && window.history.replaceState) {
                        window.history.replaceState(null, '', `#workflow/${finalWorkflowId}`);
                      }
                    }
                  } else {
                    const errorText = await createResponse.text();
                    // If 409 Conflict (workflow already exists), try to update instead
                    if (createResponse.status === 409) {
                      console.log('[WorkflowExecution] Workflow already exists (409), trying to update instead');
                      // Try to update the existing workflow
                      const updateResponse = await apiRequest(`/api/v1/workflow/${workflowId}`, {
                        method: 'PUT',
                        body: JSON.stringify({
                          nodes: nodesToSave,
                          connections: workflow.connections, // Save connections
                          history_metadata: historyToSave.map(run => ({
                            run_id: run.id,
                            timestamp: run.timestamp,
                            totalTime: run.totalTime,
                            node_ids: Object.keys(run.outputs || {}),
                            nodes_snapshot: run.nodesSnapshot || [],
                            outputs: run.outputs || {}
                          })),
                          extra_info: {
                            active_outputs: currentActiveOutputs
                          }
                        })
                      });

                      if (updateResponse.ok) {
                        console.log('[WorkflowExecution] Updated existing workflow instead of creating');
                        return; // Success, no need to update ID
                      } else {
                        const updateErrorText = await updateResponse.text();
                        throw new Error(`Failed to update existing workflow: ${updateErrorText}`);
                      }
                    } else {
                      throw new Error(`Failed to create new workflow: ${errorText}`);
                    }
                  }
                } catch (error) {
                  console.error('[WorkflowExecution] Failed to create new workflow for preset:', error);
                  // Show error to user
                  setGlobalError({
                    message: 'Failed to save execution state: preset workflow detected but creation failed',
                    details: error instanceof Error ? error.message : String(error)
                  });
                }
              } else {
                // Update existing workflow
                try {
                  const updateResponse = await apiRequest(`/api/v1/workflow/${workflowId}`, {
                    method: 'PUT',
                    body: JSON.stringify({
                      nodes: nodesToSave,
                      connections: workflow.connections, // Save connections
                      history_metadata: historyToSave.map(run => ({
                        run_id: run.id,
                        timestamp: run.timestamp,
                        totalTime: run.totalTime,
                        node_ids: Object.keys(run.outputs || {}),
                        nodes_snapshot: run.nodesSnapshot || [], // Save nodes snapshot
                        outputs: run.outputs || {} // Save optimized outputs (references instead of full data)
                      })),
                      extra_info: {
                        active_outputs: currentActiveOutputs
                      }
                    })
                  });

                  if (!updateResponse.ok) {
                    const errorText = await updateResponse.text();
                    throw new Error(`Failed to update workflow: ${errorText}`);
                  }
                  console.log('[WorkflowExecution] Execution state saved successfully');
                } catch (error) {
                  console.error('[WorkflowExecution] Failed to update workflow:', error);

                  // 检查是否是网络错误，如果是则添加到离线队列
                  const isNetworkError = error instanceof TypeError ||
                                       (error instanceof Error && error.message.includes('fetch')) ||
                                       !navigator.onLine;

                  if (isNetworkError) {
                    // 添加到离线队列以便后续恢复
                    workflowOfflineQueue.addTask(workflowId, {
                      nodes: nodesToSave,
                      extra_info: {
                        active_outputs: currentActiveOutputs
                      }
                    });
                    console.log('[WorkflowExecution] Execution state save failed (network error), added to offline queue');
                  }

                  // Show error to user
                  setGlobalError({
                    message: 'Failed to save execution state',
                    details: error instanceof Error ? error.message : String(error)
                  });

                  // 提示用户可以通过手动保存来重试
                  console.warn('[WorkflowExecution] Execution state save failed. User can retry by manually saving the workflow.');
                }
              }
            });
          } catch (err) {
            console.error('[WorkflowExecution] Error saving execution state:', err);

            // 检查是否是网络错误，如果是则添加到离线队列
            const isNetworkError = err instanceof TypeError ||
                                 (err instanceof Error && err.message.includes('fetch')) ||
                                 !navigator.onLine;

            if (isNetworkError) {
              // 添加到离线队列以便后续恢复
              workflowOfflineQueue.addTask(workflowId, {
                nodes: nodesToSave,
                extra_info: {
                  active_outputs: currentActiveOutputs
                }
              });
              console.log('[WorkflowExecution] Execution state save failed (network error), added to offline queue');
            }

            // Show error to user
            setGlobalError({
              message: 'Failed to save execution state',
              details: err instanceof Error ? err.message : String(err)
            });

            // 提示用户可以通过手动保存来重试
            console.warn('[WorkflowExecution] Execution state save failed. User can retry by manually saving the workflow.');
          }
        }, 100); // Reduced delay from 300ms to 100ms for faster save
      }
    } catch (e: any) {
      console.error('[Workflow] Execution error:', e);
      const errorMessage = e?.message || e?.toString() || '工作流执行失败';
      setGlobalError({
        message: errorMessage,
        details: e?.stack || (typeof e === 'string' ? e : JSON.stringify(e, null, 2))
      });
      setSelectedRunId(null);
    } finally {
      setWorkflow(prev => prev ? ({ ...prev, isRunning: false }) : null);
    }
  }, [
    workflow,
    setWorkflow,
    activeOutputs,
    setActiveOutputs,
    isPausedRef,
    setIsPaused,
    runningTaskIdsRef,
    abortControllerRef,
    getLightX2VConfig,
    getDescendants,
    validateWorkflow,
    setValidationErrors,
    setSelectedRunId,
    setGlobalError,
    updateNodeData,
    voiceList,
    t
  ]);

  // Stop workflow execution by cancelling all running tasks
  const stopWorkflow = useCallback(async () => {
    if (!workflow) {
      console.warn('[WorkflowExecution] Cannot stop: workflow is null');
      return;
    }

    console.log('[WorkflowExecution] Stopping workflow...');

    // Cancel all running tasks via API
    const taskIds = Array.from(runningTaskIdsRef.current.values());
    console.log(`[WorkflowExecution] Found ${taskIds.length} running tasks to cancel:`, taskIds);

    if (taskIds.length === 0) {
      console.warn('[WorkflowExecution] No running tasks found in runningTaskIdsRef');
      // Still try to abort ongoing requests and stop workflow
    }

    const cancelPromises = taskIds.map(async (taskId) => {
      try {
        console.log(`[WorkflowExecution] Sending cancel request for task ${taskId}...`);
        const response = await apiRequest(`/api/v1/task/cancel?task_id=${taskId}`, {
          method: 'GET'
        });
        if (response.ok) {
          const data = await response.json();
          console.log(`[WorkflowExecution] Task ${taskId} cancelled successfully:`, data);
        } else {
          const errorText = await response.text();
          console.warn(`[WorkflowExecution] Failed to cancel task ${taskId}:`, response.status, errorText);
        }
      } catch (error) {
        console.error(`[WorkflowExecution] Error cancelling task ${taskId}:`, error);
      }
    });

    // Wait for all cancellation requests to complete
    await Promise.all(cancelPromises);

    // Clear running task IDs
    runningTaskIdsRef.current.clear();

    // Abort any ongoing fetch requests
    if (abortControllerRef.current) {
      console.log('[WorkflowExecution] Aborting ongoing fetch requests...');
      abortControllerRef.current.abort();
      abortControllerRef.current = new AbortController();
    }

    // Stop workflow execution
    setWorkflow(prev => prev ? ({ ...prev, isRunning: false }) : null);

    // Update all running nodes to IDLE status
    setWorkflow(prev => {
      if (!prev) return null;
      return {
        ...prev,
        nodes: prev.nodes.map(n =>
          n.status === NodeStatus.RUNNING ? { ...n, status: NodeStatus.IDLE, error: undefined } : n
        ),
        isRunning: false
      };
    });

    console.log('[WorkflowExecution] Workflow stopped');
  }, [workflow, runningTaskIdsRef, abortControllerRef, setWorkflow]);

  return {
    runWorkflow,
    stopWorkflow,
    validateWorkflow,
    getDescendants
  };
};
