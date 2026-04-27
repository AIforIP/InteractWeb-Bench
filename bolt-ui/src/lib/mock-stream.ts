// src/lib/mock-stream.ts
import type { Message, AgentAction } from '../types';

// 模拟后端流式响应
export const simulateBackendResponse = (
  _userPrompt: string, // 加下划线前缀，告诉 TS 忽略"未使用变量"的报错
  onChunk: (chunk: Partial<Message>) => void,
  onAction: (action: AgentAction) => void,
  onComplete: () => void
) => {
  const thoughtText = "Thinking: User wants a snake game. I need to setup Vite project structure, install dependencies, and create the game logic using HTML5 Canvas...";
  const responseText = "I will create a Snake Game using React and HTML5 Canvas for you. \n\nI am setting up the project structure now...";

  let currentThought = "";
  let currentContent = "";

  // 1. 模拟思考过程 (Stream Thought)
  let i = 0;
  const thoughtInterval = setInterval(() => {
    currentThought += thoughtText[i];
    onChunk({ thought: currentThought, isStreaming: true });
    i++;
    if (i >= thoughtText.length) {
      clearInterval(thoughtInterval);

      // 2. 思考完后，模拟生成正文 (Stream Content)
      let j = 0;
      const contentInterval = setInterval(() => {
        currentContent += responseText[j];
        onChunk({ content: currentContent, isStreaming: true });
        j++;
        if (j >= responseText.length) {
          clearInterval(contentInterval);

          // 3. 模拟执行 Actions (Shell/File operations)
          // 模拟 npm install
          setTimeout(() => {
            onAction({ type: 'shell', status: 'running', detail: 'npm install' });

            setTimeout(() => {
              onAction({ type: 'shell', status: 'completed', detail: 'npm install' });

              // 模拟写文件
              onAction({ type: 'file', status: 'running', detail: 'src/App.tsx' });

              setTimeout(() => {
                onAction({ type: 'file', status: 'completed', detail: 'src/App.tsx' });

                // 模拟启动服务器
                onAction({ type: 'start', status: 'running', detail: 'npm run dev' });

                setTimeout(() => {
                  onAction({ type: 'start', status: 'completed', detail: 'Server running on port 3000' });
                  onChunk({ isStreaming: false }); // 结束流
                  onComplete();
                }, 1500);
              }, 1000);
            }, 2000);
          }, 500);
        }
      }, 30); // 打字速度
    }
  }, 20); // 思考速度
};