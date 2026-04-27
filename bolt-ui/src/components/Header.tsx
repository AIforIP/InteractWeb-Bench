import { useState } from 'react';
import { Download, Share2, Loader2, Zap } from 'lucide-react';

export default function Header() {
  const [isDownloading, setIsDownloading] = useState(false);

  // [复用之前的下载逻辑]
  const handleDownload = async () => {
    setIsDownloading(true);
    try {
      const response = await fetch('http://localhost:5000/api/download');
      if (!response.ok) throw new Error('Download failed');

      // 1. 转为 Blob
      const blob = await response.blob();

      // 2. 创建临时链接下载
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = "project_code.zip";
      document.body.appendChild(a);
      a.click();

      // 3. 清理
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (error) {
      console.error("Download error:", error);
      alert("下载失败，请确保后端服务 (server.py) 正在运行。");
    } finally {
      setIsDownloading(false);
    }
  };

  return (
    <header className="h-14 bg-bolt-elements-background-depth-1 border-b border-bolt-border flex items-center justify-between px-4 z-10 relative">
      {/* 左侧 Logo */}
      <div className="flex items-center gap-2">
        <div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center">
            <Zap className="w-5 h-5 text-white fill-current" />
        </div>
        <span className="font-bold text-xl text-white tracking-tight">Bolt.Local</span>
        <span className="px-2 py-0.5 rounded-full bg-blue-500/20 text-blue-400 text-xs font-medium">Beta</span>
      </div>

      {/* 右侧按钮组 */}
      <div className="flex items-center gap-3">
        {/* Share 按钮 (示意) */}
        <button className="flex items-center gap-2 px-3 py-1.5 text-gray-400 hover:text-white hover:bg-white/5 rounded-md transition-colors text-sm font-medium">
          <Share2 size={16} />
          <span>Share</span>
        </button>

        {/* [核心修改] 原来的绿色按钮 -> 现在绑定了 handleDownload */}
        <button
          onClick={handleDownload}
          disabled={isDownloading}
          className="flex items-center gap-2 px-4 py-2 bg-[#2ea043] hover:bg-[#2c974b] text-white rounded-md transition-colors shadow-sm disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {isDownloading ? (
            <Loader2 size={16} className="animate-spin" />
          ) : (
            <Download size={16} />
          )}
          <span className="text-sm font-semibold">Download Code</span>
        </button>
      </div>
    </header>
  );
}