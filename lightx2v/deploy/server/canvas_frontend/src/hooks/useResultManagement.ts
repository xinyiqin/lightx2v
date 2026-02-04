import { useMemo, useCallback } from 'react';
import { WorkflowState, WorkflowNode, NodeStatus, DataType } from '../../types';
import { TOOLS } from '../../constants';
import { Language } from '../i18n/useTranslation';
import { historyEntryToDisplayValue } from '../utils/historyEntry';

/** 单条结果：某节点某次执行的输出；entryId 为 NodeHistoryEntry.id，'current' 表示当前输出 */
export interface ResultEntry {
  runId: string;       // historyEntryId (e.g. node-xxx-ts) or 'current'
  runTimestamp: number;
  nodeId: string;
  node: WorkflowNode;
  output: any;         // 已归一化，可直接展示
}

function normalizeOutputForDisplay(v: any): any {
  if (v == null) return v;
  if (typeof v === 'object' && !Array.isArray(v)) {
    if (v.type === 'data_url' && typeof v._full_data === 'string') return v._full_data;
    if (v.type === 'url' && typeof v.data === 'string') return v.data;
    if (v.type === 'reference' && v.file_id) return { _type: 'reference', file_id: v.file_id, _note: 'Data will be loaded on demand' };
    if (v.type === 'text' && v.data !== undefined) return v.data;
    if (v.type === 'json' && v.data !== undefined) return v.data;
  }
  if (Array.isArray(v)) return v.map(normalizeOutputForDisplay);
  return v;
}

interface UseResultManagementProps {
  workflow: WorkflowState | null;
  showIntermediateResults: boolean;
  expandedOutput: { nodeId: string; fieldId?: string; historyEntryId?: string } | null;
  tempEditValue: string;
  setWorkflow: React.Dispatch<React.SetStateAction<WorkflowState | null>>;
  setExpandedOutput: (output: { nodeId: string; fieldId?: string; historyEntryId?: string } | null) => void;
  setIsEditingResult: (editing: boolean) => void;
  lang: Language;
}

export const useResultManagement = ({
  workflow,
  showIntermediateResults,
  expandedOutput,
  tempEditValue,
  setWorkflow,
  setExpandedOutput,
  setIsEditingResult,
  lang
}: UseResultManagementProps) => {
  const expandedResultData = useMemo(() => {
    if (!expandedOutput || !workflow) return null;
    const node = workflow.nodes.find(n => n.id === expandedOutput.nodeId);
    if (!node) return null;
    const tool = TOOLS.find(t => t.id === node.toolId);
    let content: any;
    if (expandedOutput.historyEntryId) {
      const list = workflow.nodeOutputHistory?.[expandedOutput.nodeId] || [];
      const entry = list.find(e => e.id === expandedOutput.historyEntryId);
      content = entry ? historyEntryToDisplayValue(entry) : undefined;
    } else {
      content = node.outputValue;
    }
    if (content && typeof content === 'object' && !Array.isArray(content)) {
      if (content.type === 'data_url' && typeof content._full_data === 'string') content = content._full_data;
      else if (content.type === 'url' && typeof content.data === 'string') content = content.data;
      else if (content.type === 'reference' && content.file_id) content = { _type: 'reference', file_id: content.file_id, _note: 'Data will be loaded on demand' };
      else if (content.type === 'text' && content.data !== undefined) content = content.data;
      else if (content.type === 'json' && content.data !== undefined) content = content.data;
    }
    if (!content && tool?.category === 'Input') content = node.data.value;
    let label = (lang === 'zh' ? tool?.name_zh : tool?.name) || 'Output';
    const type = tool?.outputs?.[0]?.type || DataType.TEXT;
    if (expandedOutput.fieldId && content && typeof content === 'object') {
      content = content[expandedOutput.fieldId];
      label = expandedOutput.fieldId;
    }
    return { content, label, type, nodeId: node.id, originalOutput: content };
  }, [expandedOutput, workflow, lang]);

  const resultEntries = useMemo((): ResultEntry[] => {
    if (!workflow) return [];
    const entries: ResultEntry[] = [];
    const isTerminal = (n: WorkflowNode) => !workflow.connections.some(c => c.sourceNodeId === n.id);

    // From nodeOutputHistory: one entry per (nodeId, historyEntry)
    const nodeHistory = workflow.nodeOutputHistory || {};
    for (const nodeId of Object.keys(nodeHistory)) {
      const node = workflow.nodes.find(n => n.id === nodeId);
      if (!node) continue;
      const list = nodeHistory[nodeId] || [];
      for (const entry of list) {
        if (!showIntermediateResults && !isTerminal(node)) continue;
        const historyValue = historyEntryToDisplayValue(entry);
        const output = normalizeOutputForDisplay(historyValue);
        entries.push({
          runId: entry.id,
          runTimestamp: entry.timestamp,
          nodeId,
          node,
          output
        });
      }
    }

    // Current run (running or just finished)
    if (workflow.isRunning) {
      const now = Date.now();
      for (const node of workflow.nodes) {
        if (node.status !== NodeStatus.RUNNING && node.status !== NodeStatus.PENDING && node.status !== NodeStatus.SUCCESS && node.status !== NodeStatus.ERROR) continue;
        if (!showIntermediateResults && !isTerminal(node) && node.status !== NodeStatus.ERROR) continue;
        const raw = node.outputValue;
        const tool = TOOLS.find(t => t.id === node.toolId);
        const output = raw != null ? normalizeOutputForDisplay(raw) : (tool?.category === 'Input' ? node.data?.value : undefined);
        entries.push({ runId: 'current', runTimestamp: now, nodeId: node.id, node, output: output ?? null });
      }
    }

    return entries.sort((a, b) => b.runTimestamp - a.runTimestamp);
  }, [workflow, showIntermediateResults]);

  const activeResultsList = useMemo(() => {
    if (!workflow) return [];
    const data = workflow ? Object.fromEntries(workflow.nodes.filter(n => n.outputValue != null).map(n => [n.id, n.outputValue])) : {};
    const nodes = workflow.nodes;
    const filtered = nodes.filter(n => {
      if (n.status === NodeStatus.ERROR) return true;
      const hasOutput = data[n.id] != null || (workflow.nodeOutputHistory?.[n.id]?.length ?? 0) > 0;
      const tool = TOOLS.find(t => t.id === n.toolId);
      const inputHasValue = tool?.category === 'Input' && n.data?.value;
      if (!hasOutput && !inputHasValue) return false;
      return true;
    }).filter(n => {
      if (!showIntermediateResults) {
        const isTerminal = !workflow.connections.some(c => c.sourceNodeId === n.id);
        return isTerminal || n.status === NodeStatus.ERROR;
      }
      return true;
    });
    const getCompletedTime = (node: WorkflowNode) => {
      if (typeof node.completedAt === 'number') return node.completedAt;
      const list = workflow.nodeOutputHistory?.[node.id];
      if (list?.length) return list[0].timestamp;
      return 0;
    };
    return [...filtered].sort((a, b) => getCompletedTime(b) - getCompletedTime(a));
  }, [workflow, showIntermediateResults]);

  const handleManualResultEdit = useCallback(() => {
    if (!expandedResultData || !expandedOutput) return;

    let finalValue: any = tempEditValue;

    // Try to parse JSON if editing the entire object and it looks like JSON
    if (!expandedOutput.fieldId && (tempEditValue.trim().startsWith('{') || tempEditValue.trim().startsWith('['))) {
      try {
        finalValue = JSON.parse(tempEditValue);
      } catch (e) {}
    }

    const nodeId = expandedOutput.nodeId;
    const fieldId = expandedOutput.fieldId;
    setWorkflow(prev => {
      if (!prev) return prev;
      const node = prev.nodes.find(n => n.id === nodeId);
      if (!node) return prev;
      const existingNodeOutput = node.outputValue;
      let newNodeOutput: any;
      if (fieldId && typeof existingNodeOutput === 'object' && existingNodeOutput !== null && !Array.isArray(existingNodeOutput)) {
        newNodeOutput = { ...existingNodeOutput, [fieldId]: finalValue };
      } else {
        newNodeOutput = finalValue;
      }
      return {
        ...prev,
        isDirty: true,
        nodes: prev.nodes.map(n => n.id === nodeId ? { ...n, outputValue: newNodeOutput, status: NodeStatus.SUCCESS } : n)
      };
    });

    setIsEditingResult(false);
  }, [expandedResultData, expandedOutput, tempEditValue, setWorkflow, setIsEditingResult]);

  return {
    expandedResultData,
    activeResultsList,
    resultEntries,
    handleManualResultEdit
  };
};
