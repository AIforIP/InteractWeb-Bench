export interface AgentAction {
  // [修改点1] 加上 'finish'，并允许 string 以兼容未来扩展
  type: 'shell' | 'file' | 'start' | 'finish' | string;

  status: 'pending' | 'running' | 'completed' | 'failed';

  // [修改点2] 设为可选，因为 start 动作可能没有 detail
  detail?: string;
}

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;

  // [保留] 虽然新版后端不怎么发 thought 了，但保留定义防止老代码报错
  thought?: string;

  isStreaming?: boolean;
  actions?: AgentAction[];
}

export interface FileNode {
  name: string;
  content?: string;
  type: 'file' | 'folder';
  children?: FileNode[];
}