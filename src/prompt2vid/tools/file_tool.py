"""文件工具：工作目录内读写 JSON/文本、检查文件存在性。"""

from __future__ import annotations

import json
import os
from pathlib import Path

from crewai.tools import tool


def _workspace() -> Path:
    ws = os.getenv("PROMPT2VID_WORKSPACE", "").strip()
    path = Path(ws) if ws else Path("output") / "run_default"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _resolve(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = _workspace() / path
    return path


@tool("Write JSON file in workspace")
def write_json(path: str, data_json: str) -> str:
    """把 JSON 内容写入工作目录文件。

    Args:
        path: 目标文件路径（相对工作目录或绝对路径）。
        data_json: JSON 字符串（将被解析校验后写回）。

    Returns:
        写入后的文件绝对路径。
    """
    dest = _resolve(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    parsed = json.loads(data_json)
    dest.write_text(
        json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return str(dest.resolve())


@tool("Read JSON file from workspace")
def read_json(path: str) -> str:
    """读取工作目录中的 JSON 文件。

    Args:
        path: JSON 文件路径（相对工作目录或绝对路径）。

    Returns:
        文件的 JSON 内容字符串。
    """
    src = _resolve(path)
    if not src.is_file():
        raise FileNotFoundError(f"文件不存在: {src}")
    return src.read_text(encoding="utf-8")


@tool("Write text file in workspace")
def write_text(path: str, content: str) -> str:
    """把文本内容写入工作目录文件。

    Args:
        path: 目标文件路径。
        content: 文本内容。

    Returns:
        写入后的文件绝对路径。
    """
    dest = _resolve(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(content, encoding="utf-8")
    return str(dest.resolve())


@tool("Read text file from workspace")
def read_text(path: str) -> str:
    """读取工作目录中的文本文件。

    Args:
        path: 文本文件路径。

    Returns:
        文件内容字符串。
    """
    src = _resolve(path)
    if not src.is_file():
        raise FileNotFoundError(f"文件不存在: {src}")
    return src.read_text(encoding="utf-8")


@tool("Check file existence and size")
def check_file(path: str) -> str:
    """检查文件是否存在并返回大小信息。

    Args:
        path: 文件路径。

    Returns:
        JSON 字符串：exists、size_bytes、absolute_path。
    """
    target = _resolve(path)
    exists = target.is_file()
    return json.dumps(
        {
            "exists": exists,
            "size_bytes": target.stat().st_size if exists else 0,
            "absolute_path": str(target.resolve()),
        },
        ensure_ascii=False,
    )


TOOLS = [write_json, read_json, write_text, read_text, check_file]
