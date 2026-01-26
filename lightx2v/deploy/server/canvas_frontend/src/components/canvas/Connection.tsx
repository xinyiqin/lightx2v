import React from 'react';
import { Trash2 } from 'lucide-react';
import { Connection as ConnectionType, WorkflowNode, NodeStatus } from '../../../types';
import { ToolDefinition } from '../../../types';
import { TOOLS } from '../../../constants';

interface ConnectionProps {
  connection: ConnectionType;
  sourceNode: WorkflowNode;
  targetNode: WorkflowNode;
  sourceOutputs: any[];
  targetInputs: any[];
  sourceNodeHeight: number;
  isSelected: boolean;
  view: { zoom: number };
  onSelect: () => void;
  onDelete: () => void;
}

export const Connection: React.FC<ConnectionProps> = ({
  connection,
  sourceNode,
  targetNode,
  sourceOutputs,
  targetInputs,
  sourceNodeHeight,
  isSelected,
  view,
  onSelect,
  onDelete
}) => {
  // Calculate port positions
  const outputPortIndex = sourceOutputs.findIndex((p) => p.id === connection.sourcePortId);
  const inputPortIndex = targetInputs.findIndex((p) => p.id === connection.targetPortId);

  const x1 = sourceNode.x + 224; // Output port center X
  const nodeBottomY = sourceNode.y + sourceNodeHeight;
  const y1 = nodeBottomY - ((sourceOutputs.length - 1 - outputPortIndex) * 30) - 24; // Output port center Y
  const x2 = targetNode.x; // Input port center X
  const y2 = targetNode.y + 71 + (inputPortIndex * 30); // Input port center Y

  const path = `M ${x1} ${y1} C ${x1 + 100} ${y1}, ${x2 - 100} ${y2}, ${x2} ${y2}`;
  const isTargetRunning = targetNode.status === NodeStatus.RUNNING;

  return (
    <g>
      <path
        d={path}
        stroke="transparent"
        strokeWidth="15"
        fill="none"
        className="pointer-events-auto cursor-pointer"
        onClick={(e) => {
          e.stopPropagation();
          onSelect();
        }}
      />
      <path
        d={path}
        stroke={isSelected ? '#90dce1' : '#5fb3b9'}
        strokeWidth={isSelected ? 4 : 3}
        fill="none"
        className="connection-path transition-all"
      />
      {isTargetRunning && (
        <circle r="4" fill="#90dce1" className="shadow-lg shadow-[#90dce1]/50">
          <animateMotion path={path} dur="1.5s" repeatCount="indefinite" />
        </circle>
      )}
      {isSelected && (
        <foreignObject
          x={(x1 + x2) / 2 - 15}
          y={(y1 + y2) / 2 - 30}
          width="30"
          height="30"
          className="overflow-visible"
        >
          <button
            onMouseDown={(e) => e.stopPropagation()}
            onClick={(e) => {
              e.stopPropagation();
              onDelete();
            }}
            className="p-1.5 bg-red-500 text-white rounded-full shadow-lg hover:bg-red-600 transition-all active:scale-90 pointer-events-auto"
          >
            <Trash2 size={12} />
          </button>
        </foreignObject>
      )}
    </g>
  );
};
