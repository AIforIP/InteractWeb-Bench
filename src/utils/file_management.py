import os
import re
from pathlib import Path

vite_file_content = """import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',
    watch: {
      usePolling: true,
      interval: 1000
    }
  }
})"""


def extract_and_write_files(response: str, workspace_dir: str):

    os.makedirs(workspace_dir, exist_ok=True)
    pattern = r'<boltAction\s+type="file"\s+filePath="([^"]+)">\s*(.*?)</boltAction>'
    matches = re.findall(pattern, response, flags=re.DOTALL | re.IGNORECASE)

    for file_path, file_content in matches:
        decoded_content = file_content.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")
        safe_file_path = file_path.lstrip("./\\")
        full_path = os.path.join(workspace_dir, safe_file_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(decoded_content)
        print(f"Created: {full_path}")

    full_path = os.path.join(workspace_dir, "vite.config.js")
    if not os.path.isfile(full_path):
        os.makedirs(os.path.dirname(full_path), exist_ok=True)

        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(vite_file_content)
        print(f"Created fallback config: {full_path}")


def get_sorted_file_paths(workspace_root):
    workspace_root = Path(workspace_root).resolve()
    all_files = []

    for dirpath, dirnames, filenames in os.walk(workspace_root):
        dirnames[:] = [d for d in dirnames if d != "node_modules"]
        dirnames.sort()
        filenames.sort()

        rel_dir = Path(dirpath).relative_to(workspace_root)
        for filename in filenames:
            file_path = (rel_dir / filename).as_posix()
            all_files.append(file_path)

    def sort_key(path):
        parts = path.split('/')
        return (len(parts), parts)

    return sorted(all_files, key=sort_key, reverse=True)