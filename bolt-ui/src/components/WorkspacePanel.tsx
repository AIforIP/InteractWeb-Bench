import { useState } from 'react';
import { Play, RotateCw, Loader2 } from 'lucide-react';
// 注意：删掉了 Download 图标的引用

interface WorkspacePanelProps {
  previewUrl: string;
}

export default function WorkspacePanel({ previewUrl }: WorkspacePanelProps) {
  const [iframeKey, setIframeKey] = useState(0);
  const [isLoading, setIsLoading] = useState(false);

  // 保持 Start Server 的修复逻辑 (前端部分)
  const handleStartServer = async () => {
    setIsLoading(true);
    try {
      await fetch('http://localhost:5000/api/preview/start', { method: 'POST' });

      // 延迟 3秒 等待 Vite 启动
      setTimeout(() => {
        setIframeKey(prev => prev + 1);
        setIsLoading(false);
      }, 3000);
    } catch (error) {
      console.error("Failed to start preview:", error);
      setIsLoading(false);
      alert("启动失败，请检查 server.py 是否运行。");
    }
  };

  return (
    <div className="h-full flex flex-col border-l border-bolt-border">
      {/* 顶部工具栏 */}
      <div className="h-12 border-b border-bolt-border flex items-center justify-between px-4 bg-[#f8f9fa]">
        <div className="flex items-center gap-2">
           <span className="text-sm font-bold text-gray-700">Preview</span>
           <span className="text-xs text-gray-500 font-mono">(Port 3000)</span>
        </div>

        <div className="flex items-center gap-2">

          {/* [已删除] 这里原本有个下载按钮，现在删掉了 */}

          {/* 启动按钮 */}
          <button
            onClick={handleStartServer}
            disabled={isLoading}
            className="flex items-center gap-1 px-3 py-1.5 bg-green-600 hover:bg-green-700 text-white text-xs font-medium rounded transition-colors disabled:opacity-50 shadow-sm"
          >
            {isLoading ? <Loader2 className="animate-spin w-3 h-3" /> : <Play className="w-3 h-3 fill-current" />}
            {isLoading ? "Starting..." : "Start Server"}
          </button>

          {/* 刷新 iframe */}
          <button
            onClick={() => setIframeKey(p => p + 1)}
            className="p-1.5 text-gray-600 hover:bg-gray-200 rounded transition-colors"
            title="Reload Preview"
          >
            <RotateCw className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* 预览区域 */}
      <div className="flex-1 bg-gray-100 relative">
        <iframe
          key={iframeKey}
          src={previewUrl}
          className="w-full h-full border-none bg-white"
          title="Workspace Preview"
          sandbox="allow-forms allow-modals allow-popups allow-presentation allow-same-origin allow-scripts"
        />
      </div>
    </div>
  );
}