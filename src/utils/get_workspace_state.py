import json
import re
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional, Any
import os
import time

_EXCLUDED_DIRS: Set[str] = {"node_modules", "dist", ".next"}
_EXCLUDED_FILES: Set[str] = {"package-lock.json"}


def remove_dir(directory):
    for _ in range(5):
        try:
            shutil.rmtree(directory)
            return True
        except:
            time.sleep(5)
    return False


def _is_excluded(path: Path, root: Path) -> bool:
    """
    Return True if *path* should be skipped according to the exclusion rules.
    """
    # Skip explicit filenames
    if path.name in _EXCLUDED_FILES:
        return True

    # Skip anything that resides inside an excluded directory
    for part in path.relative_to(root).parts:
        if part in _EXCLUDED_DIRS:
            return True
    return False


def directory_to_dict(root_dir: str) -> Dict[str, str]:
    """
    Traverse *root_dir* recursively, excluding specified files/dirs.
    Returns a mapping {relative_posix_path: file_content}.
    """
    root = Path(root_dir).expanduser().resolve()
    if not root.is_dir():
        # 这里改为返回空字典而不是报错，防止读取非目录时崩溃
        return {}

    file_map: Dict[str, str] = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if _is_excluded(path, root):
            continue

        try:
            rel_path = path.relative_to(root).as_posix()
            file_map[rel_path] = path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            print(f"Error reading file {path}: {e}")
    return file_map


def dict_to_directory(
        file_map: Dict[str, str],
        target_dir: str,
        *,
        overwrite: bool = True
) -> None:
    """
    Recreate a directory tree at *target_dir* from *file_map*.
    """
    target = Path(target_dir).expanduser().resolve()

    if target.exists():
        if overwrite:
            remove_dir(target)
        else:
            raise FileExistsError(
                f"{target} already exists. Set overwrite=True to replace it."
            )
    target.mkdir(parents=True, exist_ok=True)

    for rel_path, content in file_map.items():
        try:
            file_path = target / rel_path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding="utf-8")
        except Exception as e:
            print(f"Error writing file {rel_path}: {e}")


# ---------------------------------------------------------------------------
#  3. Restore workspace from the latest stepN.json log
# ---------------------------------------------------------------------------
_STEP_RE = re.compile(r"step(\d+)\.json$", re.IGNORECASE)


# 修复了原代码中多余的 self 参数
def _extract_step_index(filename: str) -> int:
    m = re.search(r"step(\d+)\.json$", filename)
    return int(m.group(1)) if m else -1


def restore_from_last_step(
        log_dir: str, workspace_dir: str, max_steps: int = 20
) -> Tuple[List[dict], str, int, float, float, dict]:
    """
    Locate the highest-numbered stepN.json in log_dir, rebuild workspace_dir.

    Returns:
        (messages, gui_instruction, step_idx, screenshot_grade, webvoyager_grade, nodes)
    """
    log_path = Path(log_dir).expanduser().resolve()
    # 默认返回值
    default_ret = ([], "", -1, 0, 0, {})

    if not log_path.is_dir():
        return default_ret

    # Gather and sort the step files by numeric suffix
    # 修复了原代码中 lambda 参数类型提示引发的潜在问题
    try:
        step_files = sorted(
            (
                p
                for p in log_path.iterdir()
                if p.is_file() and _STEP_RE.match(p.name) and int(_STEP_RE.match(p.name).group(1)) < max_steps
            ),
            key=lambda p: int(_STEP_RE.match(p.name).group(1)),
        )
    except Exception as e:
        print(f"Error finding step files: {e}")
        return default_ret

    if not step_files:
        return default_ret

    nodes = {}
    # 稍微优化了读取 nodes 的逻辑，防止因单个文件损坏导致整个恢复失败
    for file in step_files:
        try:
            with open(file, "r", encoding="utf-8") as f:
                d = json.load(f)
            nodes[file.name] = {
                "screenshot_grade": d.get("screenshot_grade", 0),
                "webvoyager_grade": d.get("webvoyager_grade", 0),
                "pre": d.get("pre", -1),
                "has_error": d.get("has_error", False)
            }
        except Exception:
            # 如果某个文件坏了，忽略它
            continue

    if not step_files:
        return default_ret

    latest = step_files[-1]
    print(f"[{os.path.basename(log_dir)}] Found latest step log: {latest.name}")

    try:
        step_idx = int(_STEP_RE.match(latest.name).group(1))
        with latest.open(encoding="utf-8") as f:
            data = json.load(f)

        # Recreate workspace
        if os.path.exists(workspace_dir):
            remove_dir(workspace_dir)
        os.makedirs(workspace_dir, exist_ok=True)

        # [关键修复] 兼容性检查：优先查找 workspace_files，没有则查找 files
        files_data = data.get("workspace_files")
        if files_data is None:
            files_data = data.get("files")

        if files_data is not None:
            dict_to_directory(files_data, workspace_dir, overwrite=True)
            print(f"Restored workspace files from step {step_idx}")
        else:
            print(f"[Warn] No 'workspace_files' or 'files' found in {latest.name}, workspace may be empty.")

        # Return metadata
        return (
            data.get("messages", []),
            data.get("gui_instruction", ""),
            step_idx,
            data.get("screenshot_grade", 0),
            data.get("webvoyager_grade", 0),
            nodes
        )
    except Exception as e:
        print(f"Error restoring from {latest.name}: {e}")
        return default_ret


if __name__ == "__main__":
    # Test block
    pass