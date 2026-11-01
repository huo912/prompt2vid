"""从上游仓库下载并安装第三方 Agent Skills（默认 ``anthropics/skills``）。

为什么要"下载"而不是"手写"：Agent Skills 是开放标准（agentskills.io），
上游已经沉淀了一批通用 skill（文档协作、web 应用测试、skill 自举等），
直接复用比重新造轮子划算。项目自带的领域 skill 在 ``config/skills/``
（不联网也有），下载的第三方 skill 落在 ``config/skills-external/``，
通过 ``.gitignore`` 排除，镜像里也不打包——保证离线可运行。

下载通道（本机实测）：
- GitHub 直连不可用；
- ``https://ghfast.top/https://github.com/...`` 镜像可用（git clone 已验证）；
- ``api.github.com`` / ``codeload.github.com`` 走镜像返回 403，因此不用
  tarball/API，只用 ``git clone --depth 1``。

用法::

    prompt2vid --list-skills                # 看项目 skill 与可下载清单
    prompt2vid --sync-skills                # 安装推荐集
    prompt2vid --sync-skills skill-creator  # 只装指定项
    prompt2vid --sync-skills --skills-force # 覆盖重装

环境变量：
- ``PROMPT2VID_SKILLS_REPO``：替换上游仓库 URL（自建镜像/内网 fork）；
- ``PROMPT2VID_SKILLS_MIRROR``：加速前缀，``off`` 表示直连；
- ``PROMPT2VID_SKILLS_EXTERNAL_DIR``：安装目标目录；
- ``PROMPT2VID_SKILLS_SOURCE``：本地已 checkout 的仓库路径（离线安装）。
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .skills import default_config_dir

DEFAULT_REPO = "https://github.com/anthropics/skills.git"
# 本机实测可用的 GitHub 加速前缀；第一个失败会自动换下一个，最后直连兜底。
DEFAULT_MIRRORS = ("https://ghfast.top/", "https://gh-proxy.com/")
EXTERNAL_DIR_NAME = "skills-external"
MANIFEST_NAME = "SYNC_MANIFEST.json"

REPO_VAR = "PROMPT2VID_SKILLS_REPO"
MIRROR_VAR = "PROMPT2VID_SKILLS_MIRROR"
EXTERNAL_DIR_VAR = "PROMPT2VID_SKILLS_EXTERNAL_DIR"
SOURCE_VAR = "PROMPT2VID_SKILLS_SOURCE"

CLONE_TIMEOUT = 90


class SkillSyncError(RuntimeError):
    """下载/安装外部 skill 失败（含可操作提示）。"""


@dataclass(frozen=True)
class ExternalSkill:
    """一条可下载的上游 skill 记录。"""

    name: str
    source_path: str
    license: str
    why: str
    recommended: bool = False


# 人工筛选过的清单：只收与本项目（短视频流水线 / 开发者体验）相关的条目。
# 上游还有 docx/pdf/pptx/xlsx 等办公文档类 skill，与本项目无关，刻意不收。
EXTERNAL_SKILLS: dict[str, ExternalSkill] = {
    item.name: item
    for item in (
        ExternalSkill(
            name="skill-creator",
            source_path="skills/skill-creator",
            license="Apache-2.0（见 LICENSE.txt）",
            why="按规范创建/校验/评测新 skill —— 给 config/skills/ 加新领域 skill 时用",
            recommended=True,
        ),
        ExternalSkill(
            name="doc-coauthoring",
            source_path="skills/doc-coauthoring",
            license="Anthropic 许可（见仓库 THIRD_PARTY_NOTICES.md）",
            why="结构化文档协作流程 —— 写 README / 设计文档 / 方案时用",
            recommended=True,
        ),
        ExternalSkill(
            name="webapp-testing",
            source_path="skills/webapp-testing",
            license="Apache-2.0（见 LICENSE.txt）",
            why="Playwright 驱动本地 web 应用做端到端验证 —— 给 web 控制台写 e2e 时用",
            recommended=True,
        ),
        ExternalSkill(
            name="theme-factory",
            source_path="skills/theme-factory",
            license="Anthropic 许可（见 LICENSE.txt）",
            why="成套配色/字体主题 —— 美化 web 控制台与报告页时用",
            recommended=False,
        ),
        ExternalSkill(
            name="canvas-design",
            source_path="skills/canvas-design",
            license="Anthropic 许可（见 LICENSE.txt）",
            why="版式设计原则 —— 生成海报/封面图时的排版参考（体积较大）",
            recommended=False,
        ),
    )
}


@dataclass
class SyncReport:
    """一次同步的结果汇总。"""

    target: Path
    installed: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)
    source: str = ""
    commit: str = ""

    @property
    def ok(self) -> bool:
        return not self.failed

    def format(self) -> str:
        lines = [f"外部 skill 目录: {self.target}"]
        if self.source:
            lines.append(f"来源: {self.source}" + (f" @ {self.commit[:8]}" if self.commit else ""))
        if self.installed:
            lines.append(f"已安装 {len(self.installed)} 个: {', '.join(self.installed)}")
        if self.skipped:
            lines.append(f"已存在跳过 {len(self.skipped)} 个: {', '.join(self.skipped)}")
        for name, err in self.failed.items():
            lines.append(f"失败 {name}: {err}")
        return "\n".join(lines)


def mirror_prefixes() -> list[str]:
    """镜像前缀列表（``PROMPT2VID_SKILLS_MIRROR`` 可逗号分隔多个，``off`` 直连）。"""
    raw = os.getenv(MIRROR_VAR, "").strip()
    if raw:
        if raw.lower() in {"off", "none", "direct", "0", "false"}:
            return []
        return [item.strip() for item in raw.split(",") if item.strip()]
    return list(DEFAULT_MIRRORS)


def repo_urls() -> list[str]:
    """候选下载 URL（按优先级）：镜像加速 → 直连兜底。

    ``PROMPT2VID_SKILLS_REPO`` 指向非 GitHub 地址（自建镜像 / 内网 fork）时只试该地址。
    """
    base = os.getenv(REPO_VAR, "").strip() or DEFAULT_REPO
    if base.startswith(("http://", "https://")) and "github.com" not in base:
        return [base]

    candidates = [f"{prefix}{base}" for prefix in mirror_prefixes()]
    candidates.append(base)
    seen: set[str] = set()
    ordered: list[str] = []
    for url in candidates:
        if url in seen:
            continue
        seen.add(url)
        ordered.append(url)
    return ordered


def repo_url() -> str:
    """首选下载 URL（展示用；实际会按 :func:`repo_urls` 逐个尝试）。"""
    urls = repo_urls()
    return urls[0] if urls else DEFAULT_REPO


def target_dir(config_dir: Path | None = None) -> Path:
    """外部 skill 安装目录：环境变量优先，否则 ``config/skills-external``。"""
    raw = os.getenv(EXTERNAL_DIR_VAR, "").strip()
    if raw:
        return Path(raw).expanduser()
    base = config_dir or default_config_dir() or Path("config")
    return base / EXTERNAL_DIR_NAME


def _git(*args: str, cwd: Path | None = None) -> str:
    if shutil.which("git") is None:
        raise SkillSyncError("未找到 git，无法下载外部 skill（请安装 git 或手工放置）")
    env = os.environ.copy()
    env.setdefault("GIT_TERMINAL_PROMPT", "0")
    proc = subprocess.run(  # noqa: S603 - 参数全部由本模块构造
        [
            "git",
            # 低速阈值：镜像挂住时 30s 内自动失败，而不是默认长时间挂起
            "-c", "http.lowSpeedLimit=1000",
            "-c", "http.lowSpeedTime=30",
            *args,
        ],
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        timeout=CLONE_TIMEOUT,
        check=False,
        env=env,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip().splitlines()
        message = detail[-1] if detail else f"exit {proc.returncode}"
        raise SkillSyncError(f"git {' '.join(args)} 失败: {message}")
    return proc.stdout.strip()


def clone_repo(dest: Path) -> tuple[Path, str]:
    """把上游仓库浅克隆到 ``dest``，返回 ``(源根目录, commit)``。

    优先使用 ``PROMPT2VID_SKILLS_SOURCE`` 指向的本地 checkout（离线可用）；
    联网时按 :func:`repo_urls` 逐个镜像尝试，全部失败才报错。
    """
    local = os.getenv(SOURCE_VAR, "").strip()
    if local:
        src = Path(local).expanduser()
        if not (src / "skills").is_dir():
            raise SkillSyncError(f"{SOURCE_VAR} 指向的目录不像上游仓库: {src}")
        commit = ""
        try:
            commit = _git("rev-parse", "HEAD", cwd=src)
        except SkillSyncError:
            pass
        return src, commit

    errors: list[str] = []
    for url in repo_urls():
        try:
            _git("clone", "--depth", "1", url, str(dest))
        except SkillSyncError as exc:
            errors.append(f"{url} -> {exc}")
            continue
        commit = ""
        try:
            commit = _git("rev-parse", "HEAD", cwd=dest)
        except SkillSyncError:
            pass
        return dest, commit

    hint = (
        "；已尝试全部镜像。可换镜像：PROMPT2VID_SKILLS_MIRROR=https://<prefix>/ "
        "或设置 PROMPT2VID_SKILLS_SOURCE=<本地已克隆的仓库路径> 离线安装"
    )
    raise SkillSyncError("git clone 全部通道失败: " + " | ".join(errors) + hint)


def _select(names: list[str] | None) -> list[ExternalSkill]:
    if not names:
        return [s for s in EXTERNAL_SKILLS.values() if s.recommended]
    unknown = [n for n in names if n not in EXTERNAL_SKILLS]
    if unknown:
        available = ", ".join(sorted(EXTERNAL_SKILLS))
        raise SkillSyncError(f"未知外部 skill: {unknown}（可选: {available}）")
    return [EXTERNAL_SKILLS[n] for n in names]


def _write_manifest(target: Path, report: SyncReport) -> Path:
    path = target / MANIFEST_NAME
    data: dict[str, object] = {}
    if path.is_file():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                data = loaded
        except (OSError, json.JSONDecodeError):
            data = {}
    skills = data.get("skills")
    entries = skills if isinstance(skills, dict) else {}
    for name in report.installed:
        spec = EXTERNAL_SKILLS[name]
        entries[name] = {
            "source": spec.source_path,
            "license": spec.license,
            "why": spec.why,
            "repo": report.source,
            "commit": report.commit,
        }
    data.update(
        {
            "standard": "agentskills.io",
            "repo": report.source,
            "commit": report.commit,
            "synced_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "skills": entries,
        }
    )
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def sync_skills(
    names: list[str] | None = None,
    target: Path | None = None,
    force: bool = False,
    source_dir: Path | None = None,
) -> SyncReport:
    """下载并安装外部 skill。

    Args:
        names: 要安装的 skill 名；``None`` 表示推荐集。
        target: 安装目录（默认 ``config/skills-external``）。
        force: 目标目录已存在时是否覆盖。
        source_dir: 已下载/克隆好的上游仓库根目录（测试用，跳过网络）。

    Raises:
        SkillSyncError: 清单不合法、git 缺失、克隆失败。
    """
    selected = _select(names)
    dest_root = target or target_dir()
    report = SyncReport(target=dest_root)

    todo = [s for s in selected if force or not (dest_root / s.name).is_dir()]
    report.skipped = [s.name for s in selected if s not in todo]
    if not todo:
        report.source = repo_url()
        return report

    tmp_ctx: tempfile.TemporaryDirectory[str] | None = None
    try:
        if source_dir is not None:
            repo_root, commit = source_dir, ""
        else:
            tmp_ctx = tempfile.TemporaryDirectory(prefix="p2v-skills-")
            repo_root, commit = clone_repo(Path(tmp_ctx.name) / "repo")
        report.source = repo_url() if source_dir is None else str(repo_root)
        report.commit = commit

        dest_root.mkdir(parents=True, exist_ok=True)
        for spec in todo:
            src = repo_root / spec.source_path
            if not (src / "SKILL.md").is_file():
                report.failed[spec.name] = f"上游缺少 {spec.source_path}/SKILL.md"
                continue
            dest = dest_root / spec.name
            try:
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(src, dest)
            except OSError as exc:
                report.failed[spec.name] = str(exc)
                continue
            report.installed.append(spec.name)
    finally:
        if tmp_ctx is not None:
            tmp_ctx.cleanup()

    if report.installed:
        _write_manifest(dest_root, report)
    return report


def describe_registry(external_dir: Path | None = None) -> str:
    """列出可下载清单 + 本地安装状态（``--list-skills`` 用）。"""
    dest = external_dir or target_dir()
    lines = [
        "上游仓库: " + " 或 ".join(repo_urls()),
        f"安装目录: {dest}",
        "可下载的外部 skill:",
    ]
    for spec in EXTERNAL_SKILLS.values():
        mark = "已安装" if (dest / spec.name).is_dir() else "未安装"
        tag = "（推荐）" if spec.recommended else ""
        lines.append(f"  - {spec.name} [{mark}]{tag}  许可: {spec.license}")
        lines.append(f"      {spec.why}")
    return "\n".join(lines)


def installed_skills(external_dir: Path | None = None) -> list[str]:
    """列出安装目录下已就位的 skill 名。"""
    dest = external_dir or target_dir()
    if not dest.is_dir():
        return []
    return sorted(
        child.name for child in dest.iterdir() if (child / "SKILL.md").is_file()
    )
