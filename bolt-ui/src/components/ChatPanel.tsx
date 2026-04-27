import React, { useEffect, useRef, useState } from 'react';
import { Send, Bot, User, Loader2 } from 'lucide-react';
import type { Message } from '../types';

interface ChatPanelProps {
  messages: Message[];
  isGenerating: boolean;
  isWaitingForReply: boolean;
  onSendMessage: (content: string) => void;
}

export default function ChatPanel({ messages, isGenerating, isWaitingForReply, onSendMessage }: ChatPanelProps) {
  const [input, setInput] = React.useState('');
  // [新增] 控制加载动画显示的本地状态
  const [showLoading, setShowLoading] = useState(false);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (input.trim() && !isGenerating) {
      onSendMessage(input);
      setInput('');
    }
  };

  // 1. 自动滚动
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isGenerating, showLoading]); // showLoading 变化时也要滚到底部

  // 2. 自动聚焦输入框
  useEffect(() => {
    if (isWaitingForReply && !isGenerating) {
      inputRef.current?.focus();
    }
  }, [isWaitingForReply, isGenerating]);

  // 获取最后一条消息及其内容
  const lastMsg = messages[messages.length - 1];
  const lastMsgContent = lastMsg?.content || '';
  const isLastMsgAssistant = lastMsg?.role === 'assistant';

  // [核心逻辑] 智能加载动画控制
  useEffect(() => {
    // 如果不在生成状态，强制关闭动画
    if (!isGenerating) {
      setShowLoading(false);
      return;
    }

    // 情况 A: 刚开始生成，还没吐出第一个字 -> 必须立即显示动画
    if (isLastMsgAssistant && !lastMsgContent) {
      setShowLoading(true);
      return;
    }

    // 情况 B: 正在输出文字 -> 暂时隐藏动画（因为文字在动就是反馈）
    setShowLoading(false);

    // 情况 C: 开启计时器，如果 3秒内 没有新的文字更新，说明卡住了 -> 显示动画
    const timer = setTimeout(() => {
      setShowLoading(true);
    }, 3000); // <--- 您要求的 3 秒阈值

    // 清理函数：如果 content 变了（有新字了），会触发 useEffect 重新执行，
    // 从而清除上一个 timer，实现“防抖”效果。
    return () => clearTimeout(timer);

  }, [isGenerating, lastMsgContent, isLastMsgAssistant]); // 监听内容变化

  return (
    <div className="flex flex-col h-full bg-[#1e1e1e] text-gray-300 font-sans text-sm">

      {/* 消息列表区域 */}
      <div className="flex-1 overflow-y-auto p-4 space-y-6">
        {messages.map((msg) => {
          // 依然隐藏空内容的 Agent 消息
          if (msg.role === 'assistant' && !msg.content) return null;

          return (
            <div key={msg.id} className={`flex gap-3 ${msg.role === 'user' ? 'flex-row-reverse' : 'flex-row'}`}>

              {/* 头像 */}
              <div className={`w-9 h-9 rounded-md flex items-center justify-center shrink-0 shadow-sm ${
                msg.role === 'user' ? 'bg-blue-600' : 'bg-[#383838]'
              }`}>
                {msg.role === 'user' ? <User size={18} className="text-white" /> : <Bot size={18} className="text-gray-300" />}
              </div>

              {/* 气泡 */}
              <div className={`max-w-[85%] rounded-lg px-4 py-2.5 whitespace-pre-wrap leading-relaxed shadow-sm break-words ${
                  msg.role === 'user' 
                    ? 'bg-blue-600 text-white' 
                    : 'bg-[#383838] text-gray-100 border border-gray-700'
                }`}
              >
                {msg.content}
              </div>
            </div>
          );
        })}

        {/* ========================================================================
           [动态渲染]
           只有当本地状态 showLoading 为 true 时才渲染这个动画
           ======================================================================== */}
        {showLoading && (
          <div className="flex gap-3 flex-row animate-in fade-in slide-in-from-bottom-2 duration-300">
             {/* 占位逻辑：如果上一条是 Agent，就隐藏头像保持对齐 */}
             <div className={`w-9 h-9 rounded-md flex items-center justify-center shrink-0 shadow-sm ${
               isLastMsgAssistant ? 'invisible' : 'bg-[#383838]'
             }`}>
               <Bot size={18} className="text-gray-300" />
             </div>

             {/* 思考气泡 */}
             <div className="bg-[#383838] border border-gray-700 rounded-lg px-4 py-3 flex items-center gap-1.5 h-10 w-20">
               <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce [animation-delay:-0.3s]"></div>
               <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce [animation-delay:-0.15s]"></div>
               <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
             </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* 底部输入框 */}
      <div className="p-4 border-t border-gray-700 bg-[#1e1e1e]">
        <form onSubmit={handleSubmit} className="relative flex gap-2">
          <input
            ref={inputRef}
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder={
                isGenerating
                ? "Agent 正在努力工作中..."
                : isWaitingForReply
                    ? "请在此输入您的回答..."
                    : "发送指令..."
            }
            disabled={isGenerating}
            className={`flex-1 bg-[#2d2d2d] text-white rounded-md pl-4 pr-4 py-3 focus:outline-none transition-all border ${
                !isGenerating && isWaitingForReply
                ? 'border-blue-500/60 ring-1 ring-blue-500/20' 
                : 'border-gray-600 focus:border-blue-500'
            } disabled:opacity-50`}
          />

          <button
            type="submit"
            disabled={!input.trim() || isGenerating}
            className={`px-4 rounded-md transition-colors flex items-center justify-center ${
                input.trim() && !isGenerating
                ? 'bg-blue-600 hover:bg-blue-500 text-white' 
                : 'bg-[#2d2d2d] text-gray-500 cursor-not-allowed'
            }`}
          >
            {isGenerating ? (
              <Loader2 size={18} className="animate-spin text-gray-400" />
            ) : (
              <Send size={18} />
            )}
          </button>
        </form>
      </div>
    </div>
  );
}