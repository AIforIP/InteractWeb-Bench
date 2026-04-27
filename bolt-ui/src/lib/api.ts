// [修改点] 删掉了 Message，只保留 AgentAction
import type { AgentAction } from '../types';

export const sendMessageToBackend = async (
  userPrompt: string,
  // onChunk 使用 any 类型以兼容 is_question 字段
  onChunk: (chunk: any) => void,
  onAction: (action: AgentAction) => void,
  onComplete: () => void
) => {
  try {
    const response = await fetch('http://localhost:5000/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt: userPrompt }),
    });

    if (!response.body) throw new Error("No response body");

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        if (!line.trim()) continue;
        try {
          const data = JSON.parse(line);

          // 只要有内容、提问信号或思考过程，都传给前端
          if (data.content || data.is_question || data.thought) {
            onChunk(data);
          }

          // 处理动作
          if (data.action) {
            onAction(data.action);
          }
        } catch (e) {
          console.error("JSON Parse Error:", e, line);
        }
      }
    }

    onChunk({ isStreaming: false });
    onComplete();

  } catch (error) {
    console.error("Network Error:", error);
    onChunk({ content: `\n\n[连接失败]: 无法连接到 Python 后端。\n请确认您已在 E:\\Agent_work 下运行了 'python server.py'` });
    onChunk({ isStreaming: false });
    onComplete();
  }
};