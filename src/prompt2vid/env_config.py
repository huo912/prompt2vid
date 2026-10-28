"""环境配置加载：多份 GPU 档位配置文件的解析与切换。

配置按显式程度解析（先到先得）：
1. 显式传入的 env_file（CLI --env-file，值为文件路径或档位简名）；
2. 真实环境变量 PROMPT2VID_ENV_FILE（文件路径或档位简名）；
3. 真实环境变量 PROMPT2VID_PROFILE（档位简名，如 4090 → .env.4090）；
4. 默认 .env。

档位简名：不含路径分隔符且不以 . 开头时视为简名，展开为 .env.<简名>
（如 `4090` → `.env.4090`）；否则按字面路径处理。显式指定的文件不存在
时抛 FileNotFoundError（带可选档位列表）——宁可失败也不能静默落回
错误档位的默认 .env，否则会拿 2B/13B 的配置去打另一张卡。

进程内幂等：首次成功加载后记住所选文件，后续无参调用直接返回，
避免 build_crew 等处的二次加载把配置换回默认 .env；显式传参始终生效。
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# 显式指定配置文件 / 档位的环境变量（真实环境变量，不是 .env 里的值：
# 解析发生在加载任何 .env 之前，server / k8s / systemd 用它们切换档位）
ENV_FILE_VAR = "PROMPT2VID_ENV_FILE"
PROFILE_VAR = "PROMPT2VID_PROFILE"
DEFAULT_ENV_FILE = ".env"
# 模板文件不算档位配置（简名列表里排除）
_EXAMPLE_FILE = ".env.example"

_loaded_file: Path | None = None


def _expand(value: str) -> list[Path]:
    """把路径或档位简名展开为候选文件列表（按优先级）。"""
    value = value.strip()
    if not value:
        return []
    has_sep = "/" in value or "\\" in value
    if has_sep or value.startswith("."):
        return [Path(value)]
    # 纯名字（如 4090 / local）→ 档位简名 .env.<name>，字面路径兜底
    return [Path(f".env.{value}"), Path(value)]


def _find_existing(value: str) -> Path | None:
    for candidate in _expand(value):
        if candidate.is_file():
            # 绝对路径：CWD 之后变化（server 场景）也不影响记忆与展示
            return candidate.resolve()
    return None


def available_profiles() -> list[str]:
    """列出当前目录下可用的档位配置（.env.4060 / .env.4090 / .env.5090 …）。"""
    return sorted(
        p.name for p in Path(".").glob(".env.*") if p.name != _EXAMPLE_FILE
    )


def _missing_error(value: str, source: str) -> FileNotFoundError:
    profiles = available_profiles()
    hint = (
        f"可用档位配置: {', '.join(profiles)}" if profiles
        else "当前目录下没有任何 .env.<档位> 配置文件"
    )
    return FileNotFoundError(
        f"{source} 指定的配置文件不存在: {value!r}（{hint}）"
    )


def resolve_env_file(env_file: str | Path | None = None) -> Path | None:
    """解析应加载的配置文件；没有可用配置时返回 None（表示全用默认值）。

    显式指定（参数 / 环境变量）的文件不存在时抛 FileNotFoundError。
    """
    if env_file is not None:
        found = _find_existing(str(env_file))
        if found is None:
            raise _missing_error(str(env_file), "env_file")
        return found
    for var in (ENV_FILE_VAR, PROFILE_VAR):
        raw = os.getenv(var, "").strip()
        if not raw:
            continue
        found = _find_existing(raw)
        if found is None:
            raise _missing_error(raw, f"环境变量 {var}")
        return found
    default = Path(DEFAULT_ENV_FILE)
    return default.resolve() if default.is_file() else None


def load_env(env_file: str | Path | None = None) -> Path | None:
    """加载环境配置到 os.environ，返回实际加载的文件（无配置时 None）。

    真实环境变量优先（load_dotenv 默认不覆盖已存在的值）。
    """
    global _loaded_file
    if env_file is None and _loaded_file is not None:
        return _loaded_file
    path = resolve_env_file(env_file)
    if path is not None:
        load_dotenv(path)
        _loaded_file = path
    return path


def loaded_env_file() -> Path | None:
    """本进程当前生效的配置文件（未加载过时解析一次但不加载）。"""
    if _loaded_file is not None:
        return _loaded_file
    return resolve_env_file(None)
