
export interface ChatImage {
  data: string;
  mimeType: string;
}

export interface ChatSource {
  title?: string;
  url: string;
  siteName?: string;
}

export interface ChatOperation {
  type: 'add_node' | 'delete_node' | 'update_node' | 'replace_node' |
        'add_connection' | 'delete_connection' | 'move_node';
  details: any;
}

export interface ChatOperationResult {
  success: boolean;
  operation: ChatOperation;
  result?: any;
  error?: string;
  affectedElements?: { nodeIds?: string[]; connectionIds?: string[] };
}

export interface ChatMessagePersisted {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  image?: ChatImage;
  useSearch?: boolean;
  sources?: ChatSource[];
  timestamp: number;
  error?: string;
}

export interface ChatMessage extends ChatMessagePersisted {
  operations?: ChatOperation[];
  operationResults?: ChatOperationResult[];
  historyCheckpoint?: number;
  thinking?: string;
  isStreaming?: boolean;
  choices?: string[];
}

export interface ChatHistoryList {
  workflowId: string;
  messages: ChatMessagePersisted[];
  updatedAt: number;
}

export type WorkflowChatHistoryMap = Record<string, ChatHistoryList>;

export interface ChatHistoryListResponse {
  workflow_id: string;
  messages: ChatMessagePersisted[];
  updated_at: number;
}

export enum DataType {
  TEXT = 'TEXT',
  IMAGE = 'IMAGE',
  AUDIO = 'AUDIO',
  VIDEO = 'VIDEO'
}

export enum NodeStatus {
  IDLE = 'IDLE',
  PENDING = 'PENDING',
  RUNNING = 'RUNNING',
  SUCCESS = 'SUCCESS',
  ERROR = 'ERROR'
}

export interface Port {
  id: string;
  type: DataType;
  label: string;
  label_zh?: string;
}

export interface ModelDefaultParams {
  [key: string]: any; // 模型特定的默认参数
}

export interface ModelDefinition {
  id: string;
  name: string;
  defaultParams?: ModelDefaultParams;
}

export interface ToolDefinition {
  id: string;
  name: string;
  name_zh: string;
  category: string;
  category_zh: string;
  description: string;
  description_zh: string;
  inputs: Port[];
  outputs: Port[];
  icon: string;
  models?: ModelDefinition[];
  defaultParams?: ModelDefaultParams;
}

export interface WorkflowNode {
  id: string;
  name?: string;
  toolId: string;
  x: number;
  y: number;
  status: NodeStatus;
  data: Record<string, any>; // For internal settings
  outputValue?: any; // For multi-output nodes, this is a Record<portId, value>
  error?: string;
  executionTime?: number;
  startTime?: number;
  completedAt?: number;
}

export interface Connection {
  id: string;
  sourceNodeId: string;
  sourcePortId: string;
  targetNodeId: string;
  targetPortId: string;
}

export type NodeHistoryEntryKind = 'text' | 'json' | 'file' | 'lightx2v_result';

export type NodeHistoryValue =
  | { text: string }
  | { json: any }
  | {
      fileId?: string;
      url?: string;
      dataUrl?: string;
      ext?: string;
    }
  | {
      taskId: string;
      outputName: string;
      isCloud: boolean;
    };

export interface NodeHistoryEntry {
  id: string;
  timestamp: number;
  executionTime?: number;
  metadata?: Record<string, any>;
  kind: NodeHistoryEntryKind;
  value: NodeHistoryValue;
}

export interface WorkflowState {
  id: string;
  name: string;
  description?: string;
  nodes: WorkflowNode[];
  connections: Connection[];
  isDirty: boolean;
  isRunning: boolean;
  globalInputs: Record<string, any>;
  nodeOutputHistory?: Record<string, NodeHistoryEntry[]>;
  updatedAt: number;
  createAt?: number;
  visibility?: 'private' | 'public';
  thumsupCount?: number;
  thumsupLiked?: boolean;
  authorName?: string;
  authorId?: string;
  userId?: string;
  tags?: string[];
  /**
   * AI 对话历史（运行时，从独立 API 加载）
   * 关联：通过 workflow.id 查询 GET /api/v1/workflow/{id}/chat
   */
  chatHistory?: ChatMessage[];
}
