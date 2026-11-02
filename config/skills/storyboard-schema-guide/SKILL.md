---
name: storyboard-schema-guide
description: storyboard.json 的完整字段契约与常见写错方式（裸数组、total_duration_seconds 混入 shots、id 重复、时长总和不符）。当需要把剧本拆成分镜表、或下游报「分镜表缺少非空 shots 数组」「字段缺失」时使用。
license: MIT
metadata:
  author: prompt2vid
  version: "1.0"
compatibility: 与 src/prompt2vid/storyboard.py 的宽容解析配套
---

# storyboard.json 字段契约

## 唯一正确的顶层结构

**顶层必须是对象（`{` 开头、`}` 结尾）**，只有三个键：

```json
{
  "characters": [],
  "shots": [],
  "total_duration_seconds": 16
}
```

| 键 | 类型 | 必需 | 说明 |
| --- | --- | --- | --- |
| `characters` | array | 有人物时必需 | 角色卡数组，见 `character-consistency` |
| `shots` | array | ✅ 必需且非空 | 分镜数组 |
| `total_duration_seconds` | number | 建议 | `shots[].duration_seconds` 之和 |

## `shots[]` 每项字段

| 字段 | 类型 | 必需 | 说明 |
| --- | --- | --- | --- |
| `id` | int | ✅ | 从 1 开始连续编号，不重复 |
| `characters` | int[] | ✅（空镜写 `[]`） | 本镜出镜角色 id |
| `dialogue` | array | 有人说话时 ✅ | 台词行数组 `{"speaker", "text"}`，见下节 |
| `narration` | string | 有 dialogue 时可留空 | 本镜旁白（中文，与 script.md 逐镜文案一致） |
| `visual` | string | ✅ | 画面描述（中文，给人看的） |
| `image_prompt` | string | ✅ | 英文 SDXL 首帧提示词 |
| `motion_prompt` | string | ✅ | 英文 LTX 运动描述 |
| `duration_seconds` | number | ✅ | 3~5 秒 |

## `dialogue`：让人物真的开口

每一行台词是一个对象，`speaker` 写**角色 id** 或字串 `"旁白"`：

```json
"dialogue": [
  {"speaker": 1, "text": "你们这届凡人，有点意思。"},
  {"speaker": 2, "text": "陛下，人间已经变天了。"},
  {"speaker": "旁白", "text": "三千年前，他也说过这句话。"}
]
```

规则：

1. **一句一行**，按播放顺序排列；一镜 1~3 句，单句 6~18 字（单镜 3~5 秒）。
2. `speaker` 必须是 `characters` 里已定义的 id，或 `"旁白"`。写了不存在的
   发言人会被降级成旁白（不报错，但那句话就没有角色声线了）。
3. 说话角色必须同时出现在本镜 `characters` 里。
4. 没有 `dialogue` 的镜头 = 整镜用 `narration` 配一条旁白（旧行为，仍然支持）。
5. 每句在前端得到自己的字幕时间轴，不会整镜一条长字幕。

## `characters[].voice`：不同角色不同嗓子

角色卡可选写音色，不写则由代码按角色 id 自动分配：

| 取值 | 听感 |
| --- | --- |
| `zh-CN-YunxiNeural` | 男声，年轻偏轻快 |
| `zh-CN-YunjianNeural` | 男声，中年解说腔 |
| `zh-CN-YunyangNeural` | 男声，新闻播报 |
| `zh-CN-XiaoyiNeural` | 女声，年轻偏甜 |
| `zh-CN-XiaochenNeural` | 女声，温柔 |
| `zh-CN-XiaohanNeural` | 女声，沉稳 |

IndexTTS-2.5 引擎用 `timbre`（插件音色文件名，如 `甜美女声.mp3`）；
未配置时全片用全局音色。

## 六种最常见的写错方式

### 1. 顶层写成裸数组

```json
[{"id": 1, ...}, {"id": 2, ...}]        // ✗ 解析器会把第一镜当成整张表
```

必须包一层：`{"shots": [ ... ]}`。

### 2. 把总计混进 shots

```json
{"shots": [{"id": 1, ...}, {"total_duration_seconds": 16}]}   // ✗ 被当成一个没有画面的镜头
```

`total_duration_seconds` 与 `shots` 平级。

### 3. 用 id 当字符串或从 0 开始

```json
{"id": "shot_1"}       // ✗ 用整数 1
{"id": 0}              // ✗ 从 1 开始
```

### 4. 时长总和与需求不符

需求 `duration_seconds=20` → 5 镜 × 4 秒 = 20。若写 4 镜 × 4 秒 = 16，
下游按分镜时长生成，成片就短 4 秒，质检 `duration_match`（±2s）会失败。

### 5. JSON 外面套 Markdown

```json
```json
{ ... }
```
```

把 JSON 写进文件时不要带代码块围栏，直接写裸 JSON。

### 6. 把人物台词写成旁白（“人物不说话”最常见的原因）

```json
{"narration": "玉皇大帝说：你们这届凡人有点意思", "dialogue": []}   // ✗ 成片只有旁白
{"dialogue": [{"speaker": 1, "text": "你们这届凡人，有点意思。"}]}    // ✓ 人物开口
```

只要画面里有人物、剧本里有台词，就必须写 `dialogue`；同时把
`motion_prompt` 写成带口部动作的版本（如 `the character is speaking,
lips moving naturally`），否则 LTX 会生成一张不动的脸，口型同步没目标可对齐。

## 写文件方式

必须调用 `write_json` 工具（参数 `path="storyboard.json"`，
`data_json` 是完整 JSON 字符串）。只把 JSON 贴在最终回答里而不调用工具，
本步骤失败——虽然 `llm_output.py` 会尝试抢救，但抢救不保证字段完整。

## 交付前自检

- [ ] 顶层是对象，含非空 `shots`？
- [ ] 每镜字段齐全（有人物说话时含 `dialogue`）？
- [ ] 说话角色的 `speaker` 是本镜已定义的 id，且该角色在本镜 `characters` 里？
- [ ] 台词单句 6~18 字、说话角色的 `motion_prompt` 带口部动作？
- [ ] `id` 从 1 连续、无重复？
- [ ] `duration_seconds` 之和 = `total_duration_seconds` = 需求时长？
- [ ] 镜头数在 4~8 之间？
- [ ] 有人物时 `characters` 数组非空、`shots[].characters` 正确引用？
- [ ] 没有 Markdown 围栏、没有注释、没有省略号？
