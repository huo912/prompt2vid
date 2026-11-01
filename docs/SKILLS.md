# Agent Skills（技能包）

本文说明 prompt2vid 如何接入 **Agent Skills** 开放标准（[agentskills.io](https://agentskills.io)），
包括项目内置的 8 个领域 skill、`always_on` 的取舍、外部 skill 的下载方式，
以及自己写新 skill 的规范。

> 一句话区分：**Tools 是"能做什么"（可调用的函数），Skills 是"该怎么做"（注入 prompt 的方法论）。**
> skills 不提供任何新能力，只是把领域知识喂给 LLM。

---

## 1. 为什么需要 skills

本项目的 10 个环节里，只有 4 个真正由 LLM 产出内容：

| 环节 | 执行者 | 说明 |
| --- | --- | --- |
| 需求分析 / 剧本 / 分镜 / 发布文案 | LLM（4 个 agent） | 创造力活儿，prompt 质量直接决定成片质量 |
| 首帧图 / 图生视频 / 配音 / 口型 / 剪辑 / 质检 | 代码（`stages.py`） | 确定性执行，不需要 LLM |

也就是说，**skills 只对那 4 个创意环节有用**。而这 4 个环节恰好是最容易翻车的地方：

- 分镜的 `image_prompt` 写成抽象中文 → SDXL 出图构图平淡、元素缺失；
- 分镜的 `motion_prompt` 写"人跑步" → LTX 直接肢体崩坏；
- `storyboard.json` 少一层 `shots` 包装 → 下游解析失败、整轮白跑；
- 旁白写成书面语 → TTS 念不顺、字数超时；
- 发布标题堆词 → 被限流。

这些知识原先散落在 `config/tasks/*.yaml` 的 prompt 里，越写越长、多个 agent 重复。
引入 skills 后：**task YAML 只留输出契约与硬性字段，方法论下沉到 skill**。

---

## 2. 目录与加载机制

```
config/
├── skills/                    # 项目内置 skill（随仓库与镜像分发，离线可用）
│   ├── sdxl-prompt-engineering/
│   │   ├── SKILL.md
│   │   └── references/prompt-vocab.md
│   ├── ltx-motion-prompting/
│   ├── character-consistency/
│   ├── storyboard-schema-guide/
│   ├── chinese-short-video-script/
│   ├── visual-style-presets/
│   ├── douyin-metadata-and-compliance/
│   └── delivery-qa-spec/
└── skills-external/           # --sync-skills 下载的第三方 skill（.gitignore 排除）
    ├── skill-creator/
    ├── doc-coauthoring/
    └── SYNC_MANIFEST.json
```

加载流程（`src/prompt2vid/skills.py`）：

1. `find_skills_dirs(config_dir)` → 依次返回 `config/skills`、`config/skills-external`；
2. `load_catalog(dirs)` → `crewai.skills.discover_skills()` 扫每个子目录的 `SKILL.md`，
   只解析 frontmatter（METADATA 级，成本极低），得到 `{name: Skill}`；
3. `crew.build_agents()` 读取每个 agent YAML 的 `skills:` 列表，
   `resolve_skills()` 挑出子集传给 `Agent(skills=[...])`；
4. CrewAI 在系统提示里注入 `<skills>` / `<available_skills>` 块，并自动挂一个
   `load_skill` 工具供模型按需加载——**渐进式披露**：平时只看到一行描述，
   判定相关时才读全文。

> ⚠️ 常见坑：`discover_skills(path)` 扫描的是 `path` 的**子目录**。
> 直接传单个 skill 目录会静默返回空列表——必须传父目录。

### 与 tools 的关系

`crew.py::_collect_tools()` 的装配逻辑**完全没变**。skills 走 `Agent(skills=...)`，
tools 走 `Agent(tools=...)`，两者正交。所以：

- skills 目录不存在 → agent 行为与引入前**完全一致**（向后兼容）；
- `PROMPT2VID_SKILLS=off` → 一键回到无 skills 状态（应急开关）。

---

## 3. 内置 skill 清单

| Skill | 挂载 agent | 解决什么问题 |
| --- | --- | --- |
| `sdxl-prompt-engineering` | storyboard_planner | 三段式提示词结构、权重语法、竖屏 768×1344 构图、抽象词黑名单 |
| `ltx-motion-prompting` | storyboard_planner（**常驻**） | 时长↔帧数换算（8n+1）、可用/禁用运动词、四类必崩镜头规避 |
| `character-consistency` | storyboard_planner | 角色卡写法、逐字复现规则、跨镜长相一致的构图要求 |
| `storyboard-schema-guide` | storyboard_planner（**常驻**） | `storyboard.json` 字段契约与 5 种典型写错方式 |
| `chinese-short-video-script` | script_writer（常驻） | 4.5 字/秒字数预算、3 秒钩子、TTS 不友好的写法清单 |
| `visual-style-presets` | requirement_analyst | 风格预设与 checkpoint/LoRA/画幅档位的对应关系 |
| `douyin-metadata-and-compliance` | douyin_publisher（常驻） | 标题字数句式、标签取法、敏感词自查、`dry_run` 语义 |
| `delivery-qa-spec` | quality_inspector | 六项检查的判定线与失败归因表（与 `stages.run_quality_checks` 一一对应） |

### 为什么有的 skill 是 `always_on`

默认的渐进式披露依赖**模型主动调用 `load_skill`**。本地小模型（7B/14B 量化）
经常不调用工具就直接作答，导致 skill 形同虚设。因此对本项目"不照做就一定出错"
的三处（运动提示词、分镜 JSON 结构、发布合规）标记为常驻：

```yaml
skills:
  - name: ltx-motion-prompting
    always_on: true
```

`always_on` 会用 `activate_skill()` 把 skill 提升到 INSTRUCTIONS 级，全文进入系统提示。
权衡：多占几百 token，换来稳定性。想临时全开/全关：

```bash
PROMPT2VID_SKILLS_ALWAYS_ON=sdxl-prompt-engineering,visual-style-presets  # 追加常驻
PROMPT2VID_SKILLS=off                                                    # 整体关闭
```

---

## 4. 下载外部 skill

```bash
prompt2vid --list-skills                  # 看内置 skill + 可下载清单 + 安装状态
prompt2vid --sync-skills                  # 安装推荐集（3 个）
prompt2vid --sync-skills theme-factory    # 只装指定项
prompt2vid --sync-skills --skills-force   # 覆盖重装
```

下载源是 `anthropics/skills`（Agent Skills 官方示例仓库）。由于本机 GitHub 直连不可用，
默认走镜像前缀，失败自动换下一个通道：

| 环境变量 | 作用 |
| --- | --- |
| `PROMPT2VID_SKILLS_MIRROR` | 逗号分隔的镜像前缀；`off` 表示直连 |
| `PROMPT2VID_SKILLS_REPO` | 替换上游仓库地址（自建镜像 / 内网 fork） |
| `PROMPT2VID_SKILLS_SOURCE` | 指向**本地已克隆**的仓库根目录 → 完全离线安装 |
| `PROMPT2VID_SKILLS_EXTERNAL_DIR` | 安装目标目录（默认 `config/skills-external`） |

安装后用 `SYNC_MANIFEST.json` 记录来源、commit 与各自许可。

**下载的 skill 默认不挂载**——它们和本项目的领域无关（如 `webapp-testing` 是给
开发者测前端用的）。要用就在 agent YAML 里加一行：

```yaml
# config/agents/script_writer.yaml
skills:
  - chinese-short-video-script
  - doc-coauthoring        # 写文档/方案时有用
```

离线环境（k8s 镜像内）完全不打包 `skills-external/`：8 个内置 skill 已覆盖全部
「不照做就出错」的知识点，外部 skill 纯属开发者体验增强。

---

## 5. 写一个新 skill

规范要点（[spec](https://agentskills.io)）：

- 目录名 **必须**等于 frontmatter 的 `name`（CrewAI 会校验，不一致直接报错）；
- `name`：1–64 字符，小写字母/数字/连字符；
- `description`：1–1024 字符，写清 **做什么 + 什么时候用**（模型靠它判断是否加载）；
- 正文 < 500 行；细节放 `references/`（只允许一层嵌套），脚本放 `scripts/`；
- `allowed-tools` 是实验字段，**不会**给 agent 授予工具权限，别指望它。

```markdown
---
name: my-domain-skill
description: 一句话说明做什么；再说明在什么情况下应该加载（"当……时使用"）。
license: MIT
metadata:
  author: prompt2vid
  version: "1.0"
---

# 标题
## 1. 硬规则（不照做就出错）
## 2. 对照表 / 模板
## 3. 自检清单
```

写完校验：

```bash
prompt2vid --check-only | grep skills        # 预检项「Agent Skills」列出解析结果
prompt2vid --list-skills                     # 看描述是否被正确解析
PYTHONPATH=src .venv/bin/python3.12 -m pytest tests/test_skills.py -q
```

挂载到 agent 后建议看一眼 `run.log` 首次调用的系统提示长度——常驻 skill 越多，
小模型的可用上下文越少，别把分镜表挤出窗口。

---

## 6. 排错

| 现象 | 原因与处理 |
| --- | --- |
| 预检 `NG Agent Skills: 0 个可用, N 个解析失败` | frontmatter 缺 `description` 或 `name` ≠ 目录名，看 `hint` 里的具体报错 |
| `KeyError: 未知 skill: xxx` | agent YAML 引用了不存在的 skill，`--list-skills` 核对拼写 |
| skill 好像没生效 | 先确认不是 `PROMPT2VID_SKILLS=off`；再检查该 skill 是否 `always_on`（否则取决于模型是否调用 `load_skill`） |
| `--sync-skills` 卡住/失败 | 换镜像：`PROMPT2VID_SKILLS_MIRROR=https://ghfast.top/`；或先手工 `git clone` 后设 `PROMPT2VID_SKILLS_SOURCE` |
| 同名 skill 冲突 | 先出现的目录优先（`config/skills` > `config/skills-external`），日志有 warning |

> 注意别把这里的 skill 和 **A2A Agent Card / kagent** 里的能力描述搞混：那是对外
> 暴露的服务元数据，与 agent 内部 prompt 增强无关。
