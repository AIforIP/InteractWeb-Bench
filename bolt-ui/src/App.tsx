import { useState } from 'react';
import Header from './components/Header';
import ChatPanel from './components/ChatPanel';
import WorkspacePanel from './components/WorkspacePanel';
import type { Message } from './types';
import { sendMessageToBackend } from './lib/api';

function App() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: 'welcome-1',
      role: 'assistant',
      content: '👋 您好！我是 WebGen Agent。请告诉我，您想做一个什么样的网页？(例如："做一个大转盘抽奖页面" 或 "帮我写个贪吃蛇游戏")'
    }
  ]);

  const [isGenerating, setIsGenerating] = useState(false);
  const [isWaitingForReply, setIsWaitingForReply] = useState(false);
  const [previewUrl, setPreviewUrl] = useState<string>('http://localhost:3000/');

  const handleSendMessage = (content: string) => {
    // 1. 用户发送
    const userMsg: Message = { id: Date.now().toString(), role: 'user', content };
    setMessages(prev => [...prev, userMsg]);

    // 2. 锁定输入
    setIsGenerating(true);
    setIsWaitingForReply(false);

    const agentMsgId = (Date.now() + 1).toString();
    const initialAgentMsg: Message = {
      id: agentMsgId,
      role: 'assistant',
      content: '',
      actions: []
    };
    setMessages(prev => [...prev, initialAgentMsg]);

    sendMessageToBackend(
      content,
      // Callback 1: 收到数据块
      (chunk: any) => {
        setMessages(prev => prev.map(msg =>
          msg.id === agentMsgId
            ? { ...msg, ...chunk, content: (msg.content || '') + (chunk.content || '') }
            : msg
        ));

        // 如果是提问，解锁并等待回复
        if (chunk.is_question) {
          setIsGenerating(false);
          setIsWaitingForReply(true);
        }
      },
      // Callback 2: 收到 Action 指令
      (action) => {
        if (action.type === 'start') {
          setPreviewUrl(`http://localhost:3000/?t=${Date.now()}`);
        }
        // [新增] 如果收到 finish 指令，也应该停止生成状态
        if (action.type === 'finish') {
           setIsGenerating(false);
        }
      },
      // Callback 3 [核心修复]: 无论成功还是失败，只要流结束了，就必须停止 Loading！
      () => {
        setIsGenerating(false);
      }
    );
  };

  return (
    <div className="flex flex-col h-screen bg-bolt-dark text-gray-200">
      <Header />
      <div className="flex flex-1 overflow-hidden">
        <div className="w-[30%] min-w-[350px] max-w-[500px] border-r border-bolt-border flex flex-col">
          <ChatPanel
            messages={messages}
            isGenerating={isGenerating}
            isWaitingForReply={isWaitingForReply}
            onSendMessage={handleSendMessage}
          />
        </div>
        <div className="flex-1 bg-white">
          <WorkspacePanel previewUrl={previewUrl} />
        </div>
      </div>
    </div>
  );
}

export default App;