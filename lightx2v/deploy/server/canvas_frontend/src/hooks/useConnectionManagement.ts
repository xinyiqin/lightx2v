import { useCallback } from 'react';
import { WorkflowState, Connection } from '../../types';

interface UseConnectionManagementProps {
  workflow: WorkflowState | null;
  setWorkflow: React.Dispatch<React.SetStateAction<WorkflowState | null>>;
  selectedConnectionId: string | null;
  setSelectedConnectionId: (id: string | null) => void;
  selectedRunId: string | null;
  setConnecting: (connecting: { nodeId: string; portId: string; isOutput: boolean } | null) => void;
}

export const useConnectionManagement = ({
  workflow,
  setWorkflow,
  selectedConnectionId,
  setSelectedConnectionId,
  selectedRunId,
  setConnecting
}: UseConnectionManagementProps) => {
  const addConnection = useCallback((connection: {
    id: string;
    sourceNodeId: string;
    sourcePortId: string;
    targetNodeId: string;
    targetPortId: string;
  }) => {
    if (selectedRunId) return;
    setWorkflow(prev => prev ? ({ ...prev, connections: [...prev.connections, connection], isDirty: true }) : null);
    setConnecting(null);
  }, [selectedRunId, setWorkflow, setConnecting]);

  const deleteConnection = useCallback((connectionId: string) => {
    if (!connectionId) return;
    if (selectedRunId) return;
    setWorkflow(prev => prev ? ({ ...prev, connections: prev.connections.filter(c => c.id !== connectionId), isDirty: true }) : null);
    if (selectedConnectionId === connectionId) {
      setSelectedConnectionId(null);
    }
  }, [selectedConnectionId, selectedRunId, setWorkflow, setSelectedConnectionId]);

  return {
    addConnection,
    deleteConnection
  };
};
