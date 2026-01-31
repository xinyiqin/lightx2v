import { useMemo, useCallback } from 'react';
import { WorkflowState, WorkflowNode, NodeStatus, DataType } from '../../types';
import { TOOLS } from '../../constants';
import { Language } from '../i18n/useTranslation';

/** 单条结果：某次运行中某个节点的输出，按执行开始时间展示，用于结果面板按时间线平铺 */
export interface ResultEntry {
  runId: string;
  runTimestamp: number;
  nodeId: string;
  node: WorkflowNode;
  output: any; // 已归一化，可直接展示（data_url/url/text/json 已展开）
}

function normalizeOutputForDisplay(v: any): any {
  if (v == null) return v;
  if (typeof v === 'object' && !Array.isArray(v)) {
    if (v.type === 'data_url' && typeof v._full_data === 'string') return v._full_data;
    if (v.type === 'url' && typeof v.data === 'string') return v.data;
    if (v.type === 'reference' && v.data_id) return { _type: 'reference', data_id: v.data_id, file_id: v.file_id, _note: 'Data will be loaded on demand' };
    if (v.type === 'text' && v.data !== undefined) return v.data;
    if (v.type === 'json' && v.data !== undefined) return v.data;
  }
  if (Array.isArray(v)) return v.map(normalizeOutputForDisplay);
  return v;
}

interface UseResultManagementProps {
  workflow: WorkflowState | null;
  selectedRunId: string | null;
  activeOutputs: Record<string, any>;
  expandedOutput: { nodeId: string; fieldId?: string; runId?: string } | null;
  tempEditValue: string;
  setActiveOutputs: React.Dispatch<React.SetStateAction<Record<string, any>>>;
  setWorkflow: React.Dispatch<React.SetStateAction<WorkflowState | null>>;
  setExpandedOutput: (output: { nodeId: string; fieldId?: string; runId?: string } | null) => void;
  setIsEditingResult: (editing: boolean) => void;
  lang: Language;
}

export const useResultManagement = ({
  workflow,
  selectedRunId,
  activeOutputs,
  expandedOutput,
  tempEditValue,
  setActiveOutputs,
  setWorkflow,
  setExpandedOutput,
  setIsEditingResult,
  lang
}: UseResultManagementProps) => {
  const expandedResultData = useMemo(() => {
    if (!expandedOutput || !workflow) return null;
    const runId = expandedOutput.runId ?? selectedRunId;
    const run = runId && runId !== 'current' ? workflow.history.find(r => r.id === runId) : null;
    const outputs = run ? run.outputs : activeOutputs;
    const nodes = run ? run.nodesSnapshot : workflow.nodes;
    const node = nodes.find(n => n.id === expandedOutput.nodeId);
    if (!node) return null;
    const tool = TOOLS.find(t => t.id === node.toolId);
    let content = outputs[node.id];

    // Handle optimized history output shapes (纯前端未保存时 run.outputs 为 type 'data_url' / 'url' / 'text'，需展开为实际值)
    if (content && typeof content === 'object' && !Array.isArray(content)) {
      if (content.type === 'data_url' && typeof content._full_data === 'string') {
        content = content._full_data;
      } else if (content.type === 'url' && typeof content.data === 'string') {
        content = content.data;
      } else if (content.type === 'reference' && content.data_id) {
        content = {
          _type: 'reference',
          data_id: content.data_id,
          file_id: content.file_id,
          _note: 'Data will be loaded on demand'
        };
      } else if (content.type === 'text' && content.data !== undefined) {
        content = content.data;
      } else if (content.type === 'json' && content.data !== undefined) {
        content = content.data;
      }
    }

    if (!content && tool?.category === 'Input') {
      content = node.data.value;
    }

    let label = (lang === 'zh' ? tool?.name_zh : tool?.name) || "Output";
    let type = tool?.outputs[0]?.type || DataType.TEXT;
    if (expandedOutput.fieldId && content && typeof content === 'object') {
      content = content[expandedOutput.fieldId];
      label = expandedOutput.fieldId;
    }
    return { content, label, type, nodeId: node.id, originalOutput: outputs[node.id] };
  }, [expandedOutput, selectedRunId, workflow, activeOutputs, lang]);

  const resultEntries = useMemo((): ResultEntry[] => {
    if (!workflow) return [];
    const entries: ResultEntry[] = [];
    const isTerminal = (n: WorkflowNode) => !workflow.connections.some(c => c.sourceNodeId === n.id);

    const historyRuns = [...workflow.history].sort((a, b) => b.timestamp - a.timestamp);
    for (const run of historyRuns) {
      const nodes = run.nodesSnapshot || workflow.nodes;
      for (const node of nodes) {
        if (node.status === NodeStatus.ERROR) {
          entries.push({ runId: run.id, runTimestamp: run.timestamp, nodeId: node.id, node, output: null });
          continue;
        }
        const raw = run.outputs[node.id];
        if (!raw) {
          const tool = TOOLS.find(t => t.id === node.toolId);
          if (tool?.category === 'Input' && node.data?.value) entries.push({ runId: run.id, runTimestamp: run.timestamp, nodeId: node.id, node, output: node.data.value });
          continue;
        }
        if (!workflow.showIntermediateResults && !isTerminal(node) && node.status !== NodeStatus.ERROR) continue;
        entries.push({ runId: run.id, runTimestamp: run.timestamp, nodeId: node.id, node, output: normalizeOutputForDisplay(raw) });
      }
    }

    if (workflow.isRunning) {
      const nodes = workflow.nodes;
      const now = Date.now();
      for (const node of nodes) {
        if (node.status !== NodeStatus.RUNNING && node.status !== NodeStatus.SUCCESS && node.status !== NodeStatus.ERROR) continue;
        if (!workflow.showIntermediateResults && !isTerminal(node) && node.status !== NodeStatus.ERROR) continue;
        const raw = activeOutputs[node.id] ?? node.outputValue;
        const tool = TOOLS.find(t => t.id === node.toolId);
        const output = raw != null ? normalizeOutputForDisplay(raw) : (tool?.category === 'Input' ? node.data?.value : undefined);
        entries.unshift({ runId: 'current', runTimestamp: now, nodeId: node.id, node, output: output ?? null });
      }
    }

    return entries.sort((a, b) => b.runTimestamp - a.runTimestamp);
  }, [workflow, activeOutputs]);

  const activeResultsList = useMemo(() => {
    if (!workflow) return [];
    const sourceRun = selectedRunId ? workflow.history.find(r => r.id === selectedRunId) : null;
    const data = sourceRun ? sourceRun.outputs : activeOutputs;
    const nodes = sourceRun ? sourceRun.nodesSnapshot : workflow.nodes;

    const filtered = nodes.filter(n => {
      if (n.status === NodeStatus.ERROR) return true;
      if (!data[n.id]) {
        const tool = TOOLS.find(t => t.id === n.toolId);
        return tool?.category === 'Input' && n.data.value;
      }
      return true;
    }).filter(n => {
      if (!workflow.showIntermediateResults) {
        const isTerminal = !workflow.connections.some(c => c.sourceNodeId === n.id);
        return isTerminal || n.status === NodeStatus.ERROR;
      }
      return true;
    });
    const getCompletedTime = (node: WorkflowNode) => {
      if (typeof node.completedAt === 'number') return node.completedAt;
      return 0;
    };
    return [...filtered].sort((a, b) => getCompletedTime(b) - getCompletedTime(a));
  }, [workflow, selectedRunId, activeOutputs]);

  const handleManualResultEdit = useCallback(() => {
    if (!expandedResultData || !expandedOutput) return;

    let finalValue: any = tempEditValue;

    // Try to parse JSON if editing the entire object and it looks like JSON
    if (!expandedOutput.fieldId && (tempEditValue.trim().startsWith('{') || tempEditValue.trim().startsWith('['))) {
      try {
        finalValue = JSON.parse(tempEditValue);
      } catch (e) {}
    }

    setActiveOutputs(prev => {
      const nodeId = expandedOutput.nodeId;
      const fieldId = expandedOutput.fieldId;
      const existingNodeOutput = prev[nodeId];

      let newNodeOutput;
      if (fieldId && typeof existingNodeOutput === 'object' && existingNodeOutput !== null) {
        // Merge edited field into the existing structured object
        newNodeOutput = { ...existingNodeOutput, [fieldId]: finalValue };
      } else {
        // Overwrite entire node output
        newNodeOutput = finalValue;
      }

      return {
        ...prev,
        [nodeId]: newNodeOutput
      };
    });

    setWorkflow(prev => prev ? ({
      ...prev,
      isDirty: true,
      nodes: prev.nodes.map(n => n.id === expandedOutput.nodeId ? { ...n, status: NodeStatus.SUCCESS } : n)
    }) : null);

    setIsEditingResult(false);
  }, [expandedResultData, expandedOutput, tempEditValue, setActiveOutputs, setWorkflow, setIsEditingResult]);

  return {
    expandedResultData,
    activeResultsList,
    resultEntries,
    handleManualResultEdit
  };
};
