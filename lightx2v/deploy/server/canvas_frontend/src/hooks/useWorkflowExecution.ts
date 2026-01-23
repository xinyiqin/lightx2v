import { useCallback, useRef } from 'react';
import { WorkflowState, Connection, NodeStatus, GenerationRun } from '../../types';
import { TOOLS } from '../../constants';
import { 
  geminiText, geminiImage, geminiSpeech, geminiVideo, 
  lightX2VTask, lightX2VTTS, lightX2VVoiceCloneTTS,
  deepseekText, doubaoText, ppchatGeminiText
} from '../../services/geminiService';
import { removeGeminiWatermark } from '../../services/watermarkRemover';
import { useTranslation, Language } from '../i18n/useTranslation';

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
      Object.entries(activeOutputs).forEach(([nodeId, val]) => {
        if (!nodesToRunIds.has(nodeId)) sessionOutputs[nodeId] = val;
      });
      setActiveOutputs(prev => {
        const next = { ...prev };
        nodesToRunIds.forEach(id => delete next[id]);
        return next;
      });
    } else {
      // Full workflow run: clear all outputs to prevent memory accumulation
      setActiveOutputs({});
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
                case 'text-prompt': result = node.data.value || ""; break;
                case 'image-input': 
                  const imageValue = node.data.value || [];
                  // Convert file paths to base64 data URLs if needed
                  if (Array.isArray(imageValue) && imageValue.length > 0) {
                    const convertedImages = await Promise.all(imageValue.map(async (img: string) => {
                      // If it's already a data URL or base64, return as is
                      if (img.startsWith('data:') || (!img.startsWith('http') && img.includes(','))) {
                        return img;
                      }
                      // If it's a file path (starts with /), load and convert to base64
                      if (img.startsWith('/')) {
                        try {
                          const response = await fetch(img);
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
                      // If it's a URL, return as is (will be handled by the service)
                      return img;
                    }));
                    result = convertedImages;
                  } else {
                    result = imageValue;
                  }
                  break;
                case 'audio-input': 
                  const audioValue = node.data.value;
                  // Convert file path to base64 data URL if needed
                  if (audioValue && typeof audioValue === 'string') {
                    // If it's already a data URL or base64, return as is
                    if (audioValue.startsWith('data:') || (!audioValue.startsWith('http') && audioValue.includes(','))) {
                      result = audioValue;
                    } else if (audioValue.startsWith('/')) {
                      // If it's a file path (starts with /), load and convert to base64
                      try {
                        const response = await fetch(audioValue);
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
                      // If it's a URL or other format, return as is
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
                    console.log("[LightX2V] Calling lightX2VTask for text-to-image");
                    result = await lightX2VTask(
                      lightX2VConfig.url, 
                      lightX2VConfig.token, 
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
                    result = await lightX2VTask(
                      lightX2VConfig.url, 
                      lightX2VConfig.token, 
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
                  result = await lightX2VTask(
                    lightX2VConfig.url, 
                    lightX2VConfig.token, 
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
                  result = await lightX2VTask(
                    lightX2VConfig.url, 
                    lightX2VConfig.token, 
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
                    result = await lightX2VTask(
                        lightX2VConfig.url, 
                        lightX2VConfig.token, 
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
                  if (model === 'wan2.2_animate') {
                    result = await lightX2VTask(
                      lightX2VConfig.url, 
                      lightX2VConfig.token, 
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
                  result = await lightX2VTask(
                    lightX2VConfig.url, 
                    lightX2VConfig.token, 
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
              setWorkflow(prev => prev ? ({ ...prev, nodes: prev.nodes.map(n => n.id === node.id ? { ...n, status: NodeStatus.ERROR, error: err.message || 'Unknown execution error', executionTime: nodeDuration } : n) }) : null);
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
              setWorkflow(prev => prev ? ({ ...prev, nodes: prev.nodes.map(n => n.id === nodeId ? { ...n, status: NodeStatus.SUCCESS, executionTime: duration } : n) }) : null);
              executedInSession.add(nodeId);
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
                  updatedNodes[index] = { ...updatedNodes[index], status: NodeStatus.ERROR, error, executionTime: duration };
                }
              });
              return { ...prev, nodes: updatedNodes };
            });
          }
        }
      }
      const runTotalTime = performance.now() - runStartTime;
      
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
        executionTime: n.executionTime
      }));
      
      const newRun: GenerationRun = { 
        id: `run-${Date.now()}`, 
        timestamp: Date.now(), 
        outputs: { ...sessionOutputs }, 
        nodesSnapshot: lightweightNodesSnapshot,
        totalTime: runTotalTime 
      };
      
      // Limit history to 5 runs to reduce memory usage (was 10)
      setWorkflow(prev => prev ? ({ ...prev, history: [newRun, ...prev.history].slice(0, 5) }) : null);
      setSelectedRunId(newRun.id);
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

  return {
    runWorkflow,
    validateWorkflow,
    getDescendants
  };
};

