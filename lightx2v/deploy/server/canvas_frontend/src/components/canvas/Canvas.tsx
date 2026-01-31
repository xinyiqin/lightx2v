import React, { useRef } from 'react';
import { WorkflowState, WorkflowNode, Connection, NodeStatus, DataType } from '../../../types';
import { ViewState } from '../../utils/canvas';
import { Connection as ConnectionComponent } from './Connection';
import { Node } from './Node';
import { TOOLS } from '../../../constants';
import { ToolDefinition, Port } from '../../../types';

interface CanvasProps {
  workflow: WorkflowState;
  view: ViewState;
  selectedNodeId: string | null;
  selectedConnectionId: string | null;
  connecting: {
    nodeId: string;
    portId: string;
    type: DataType;
    direction: 'in' | 'out';
    startX: number;
    startY: number;
  } | null;
  mousePos: { x: number; y: number };
  activeOutputs: Record<string, any>;
  nodeHeights: Map<string, number>;
  sourceNodes: WorkflowNode[];
  sourceOutputs: Record<string, any>;
  isOverNode: boolean;
  isPanning: boolean;
  canvasRef: React.RefObject<HTMLDivElement>;
  onMouseMove: (e: React.MouseEvent) => void;
  onMouseDown: (e: React.MouseEvent) => void;
  onMouseUp: () => void;
  onMouseLeave: () => void;
  onWheel: (e: React.WheelEvent) => void;
  onNodeSelect: (nodeId: string) => void;
  onConnectionSelect: (connectionId: string) => void;
  onDeleteConnection: (connectionId: string) => void;
  onNodeDragStart: (nodeId: string, offsetX: number, offsetY: number) => void;
  onNodeDrag: (nodeId: string, x: number, y: number) => void;
  onNodeDragEnd: () => void;
  getNodeOutputs: (node: WorkflowNode) => any[];
  lang: 'en' | 'zh';
  // Node component props
  showReplaceMenu?: string | null;
  showOutputQuickAdd?: { nodeId: string; portId: string } | null;
  showModelSelect?: string | null;
  showVoiceSelect?: string | null;
  lightX2VVoiceList?: { voices?: any[]; emotions?: string[]; languages?: any[] } | null;
  cloneVoiceList?: any[];
  onUpdateNodeData?: (nodeId: string, key: string, value: any) => void;
  onDeleteNode?: (nodeId: string) => void;
  onReplaceNode?: (nodeId: string, newToolId: string) => void;
  onRunWorkflow?: (nodeId?: string, runThisOnly?: boolean) => void;
  onCancelNodeRun?: (nodeId: string) => void;
  pendingRunNodeIds?: string[];
  onSetReplaceMenu?: (nodeId: string | null) => void;
  onSetOutputQuickAdd?: (value: { nodeId: string; portId: string } | null) => void;
  onSetModelSelect?: (nodeId: string | null) => void;
  onSetVoiceSelect?: (nodeId: string | null) => void;
  onSetExpandedOutput?: (value: { nodeId: string; fieldId?: string } | null) => void;
  onSetShowAudioEditor?: (nodeId: string | null) => void;
  onSetShowVideoEditor?: (nodeId: string | null) => void;
  onSetConnecting?: (value: {
    nodeId: string;
    portId: string;
    type: DataType;
    direction: 'in' | 'out';
    startX: number;
    startY: number;
  } | null) => void;
  onAddConnection?: (connection: {
    id: string;
    sourceNodeId: string;
    sourcePortId: string;
    targetNodeId: string;
    targetPortId: string;
  }) => void;
  onClearSelectedRunId?: () => void;
  getReplaceableTools?: (nodeId: string) => ToolDefinition[];
  getCompatibleToolsForOutput?: (outputType: DataType) => ToolDefinition[];
  quickAddInput?: (node: WorkflowNode, port: Port) => void;
  quickAddOutput?: (node: WorkflowNode, port: Port, toolId: string) => void;
  onNodeHeightChange?: (nodeId: string, height: number) => void;
  onAddNodeToChat?: (nodeId: string, name: string) => void;
  resolveLightX2VResultRef?: (ref: import('../../hooks/useWorkflowExecution').LightX2VResultRef) => Promise<string>;
}

export const Canvas: React.FC<CanvasProps> = ({
  workflow,
  view,
  selectedNodeId,
  selectedConnectionId,
  connecting,
  mousePos,
  activeOutputs,
  nodeHeights,
  sourceNodes,
  sourceOutputs,
  isOverNode,
  isPanning,
  canvasRef,
  onMouseMove,
  onMouseDown,
  onMouseUp,
  onMouseLeave,
  onWheel,
  onNodeSelect,
  onConnectionSelect,
  onDeleteConnection,
  onNodeDragStart,
  onNodeDrag,
  onNodeDragEnd,
  getNodeOutputs,
  lang,
  showReplaceMenu,
  showOutputQuickAdd,
  showModelSelect,
  showVoiceSelect,
  lightX2VVoiceList,
  cloneVoiceList = [],
  onUpdateNodeData = () => {},
  onDeleteNode = () => {},
  onReplaceNode = () => {},
  onRunWorkflow = () => {},
  onCancelNodeRun = () => {},
  pendingRunNodeIds = [],
  onSetReplaceMenu = () => {},
  onSetOutputQuickAdd = () => {},
  onSetModelSelect = () => {},
  onSetVoiceSelect = () => {},
  onSetExpandedOutput = () => {},
  onSetShowAudioEditor = () => {},
  onSetShowVideoEditor = () => {},
  onSetConnecting = () => {},
  onAddConnection = () => {},
  onClearSelectedRunId = () => {},
  getReplaceableTools = () => [],
  getCompatibleToolsForOutput = () => [],
  quickAddInput = () => {},
  quickAddOutput = () => {},
  onNodeHeightChange,
  onAddNodeToChat,
  resolveLightX2VResultRef
}) => {
  return (
    <main
      ref={canvasRef}
      className="flex-1 h-full relative overflow-hidden canvas-grid bg-[#0a0f1e]"
      style={{
        cursor: isOverNode ? 'default' : isPanning ? 'grabbing' : 'grab'
      }}
      onMouseMove={onMouseMove}
      onMouseUp={onMouseUp}
      onMouseDown={onMouseDown}
      onMouseLeave={onMouseLeave}
      onWheel={onWheel}
    >
      <div
        style={{
          transform: `translate(${view.x}px, ${view.y}px) scale(${view.zoom})`,
          transformOrigin: '0 0',
          width: '100%',
          height: '100%'
        }}
      >
        {/* SVG for connections */}
        <svg className="absolute inset-0 w-full h-full pointer-events-none z-0 overflow-visible">
          {workflow.connections.map((c) => {
            const sNode = sourceNodes.find((n) => n.id === c.sourceNodeId);
            const tNode = sourceNodes.find((n) => n.id === c.targetNodeId);
            if (!sNode || !tNode) return null;

            const sOutputs = getNodeOutputs(sNode);
            const tTool = TOOLS.find((t) => t.id === tNode.toolId);
            const tInputs = tTool?.inputs || [];

            // Calculate node height based on output count if not available from nodeHeights
            // Header (48px) + top padding (16px) + outputs area ((n-1)*30 + 30) + bottom padding (16px)
            const sourceNodeHeight = nodeHeights.get(sNode.id) || Math.max(140, 48 + 16 + Math.max(0, (sOutputs.length - 1) * 30) + 30 + 16);

            return (
              <ConnectionComponent
                key={c.id}
                connection={c}
                sourceNode={sNode}
                targetNode={tNode}
                sourceOutputs={sOutputs}
                targetInputs={tInputs}
                sourceNodeHeight={sourceNodeHeight}
                isSelected={selectedConnectionId === c.id}
                view={view}
                onSelect={() => {
                  onConnectionSelect(c.id);
                  onNodeSelect('');
                }}
                onDelete={() => {
                  onDeleteConnection(c.id);
                }}
              />
            );
          })}
          {connecting && (
            <path
              d={`M ${connecting.startX} ${connecting.startY} C ${
                connecting.startX + 100 / view.zoom
              } ${connecting.startY}, ${((mousePos.x - view.x) / view.zoom) - 100 / view.zoom} ${
                (mousePos.y - view.y) / view.zoom
              }, ${(mousePos.x - view.x) / view.zoom} ${(mousePos.y - view.y) / view.zoom}`}
              stroke="#4f46e5"
              strokeWidth={3 / view.zoom}
              strokeDasharray={`${6 / view.zoom},${6 / view.zoom}`}
              fill="none"
              className="animate-marching-ants"
            />
          )}
        </svg>

        {/* Nodes */}
        {sourceNodes.map((node) => (
          <Node
            key={node.id}
            node={node}
            workflow={workflow}
            isSelected={selectedNodeId === node.id}
            activeOutputs={activeOutputs}
            sourceOutputs={sourceOutputs}
            nodeHeight={nodeHeights.get(node.id) || 140}
            onSelect={onNodeSelect}
            onDragStart={onNodeDragStart}
            onDrag={onNodeDrag}
            onDragEnd={onNodeDragEnd}
            getNodeOutputs={getNodeOutputs}
            canvasRef={canvasRef}
            view={view}
            lang={lang}
            showReplaceMenu={showReplaceMenu}
            showOutputQuickAdd={showOutputQuickAdd}
            showModelSelect={showModelSelect}
            showVoiceSelect={showVoiceSelect}
            lightX2VVoiceList={lightX2VVoiceList}
            cloneVoiceList={cloneVoiceList}
            onUpdateNodeData={onUpdateNodeData}
            onDeleteNode={onDeleteNode}
            onReplaceNode={onReplaceNode}
            onRunWorkflow={onRunWorkflow}
            onCancelNodeRun={onCancelNodeRun}
            pendingRunNodeIds={pendingRunNodeIds}
            onSetReplaceMenu={onSetReplaceMenu}
            onSetOutputQuickAdd={onSetOutputQuickAdd}
            onSetModelSelect={onSetModelSelect}
            onSetVoiceSelect={onSetVoiceSelect}
            onSetExpandedOutput={onSetExpandedOutput}
            onSetShowAudioEditor={onSetShowAudioEditor}
            onSetShowVideoEditor={onSetShowVideoEditor}
            onSetConnecting={onSetConnecting}
            onAddConnection={onAddConnection}
            onClearSelectedRunId={onClearSelectedRunId}
            getReplaceableTools={getReplaceableTools}
            getCompatibleToolsForOutput={getCompatibleToolsForOutput}
            quickAddInput={quickAddInput}
            quickAddOutput={quickAddOutput}
            connecting={connecting}
            onNodeHeightChange={onNodeHeightChange}
            onAddNodeToChat={onAddNodeToChat}
            resolveLightX2VResultRef={resolveLightX2VResultRef}
          />
        ))}
      </div>
    </main>
  );
};
