import { useMemo, useCallback } from 'react';
import { WorkflowState, WorkflowNode, NodeStatus, DataType } from '../../types';
import { TOOLS } from '../../constants';
import { Language } from '../i18n/useTranslation';

interface UseResultManagementProps {
  workflow: WorkflowState | null;
  selectedRunId: string | null;
  activeOutputs: Record<string, any>;
  expandedOutput: { nodeId: string; fieldId?: string } | null;
  tempEditValue: string;
  setActiveOutputs: React.Dispatch<React.SetStateAction<Record<string, any>>>;
  setWorkflow: React.Dispatch<React.SetStateAction<WorkflowState | null>>;
  setExpandedOutput: (output: { nodeId: string; fieldId?: string } | null) => void;
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
    const run = selectedRunId ? workflow.history.find(r => r.id === selectedRunId) : null;
    const outputs = run ? run.outputs : activeOutputs;
    const nodes = run ? run.nodesSnapshot : workflow.nodes;
    const node = nodes.find(n => n.id === expandedOutput.nodeId);
    if (!node) return null;
    const tool = TOOLS.find(t => t.id === node.toolId);
    let content = outputs[node.id];

    // Handle reference types (from optimized history outputs)
    if (content && typeof content === 'object' && content.type === 'reference' && content.data_id) {
      // This is a reference, we need to load the actual data
      // For now, return the reference info (actual loading can be done on demand)
      content = {
        _type: 'reference',
        data_id: content.data_id,
        file_id: content.file_id,
        _note: 'Data will be loaded on demand'
      };
    } else if (content && typeof content === 'object' && content.type === 'text' && content.data_id) {
      // Text with data_id - use the text data directly
      content = content.data;
    } else if (content && typeof content === 'object' && content.type === 'json' && content.data_id) {
      // JSON with data_id - use the JSON data directly
      content = content.data;
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

  const activeResultsList = useMemo(() => {
    if (!workflow) return [];
    const sourceRun = selectedRunId ? workflow.history.find(r => r.id === selectedRunId) : null;
    const data = sourceRun ? sourceRun.outputs : activeOutputs;
    const nodes = sourceRun ? sourceRun.nodesSnapshot : workflow.nodes;

    return nodes.filter(n => {
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
    handleManualResultEdit
  };
};
