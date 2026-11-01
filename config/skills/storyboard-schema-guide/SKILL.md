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
| `narration` | string | ✅ | 本镜旁白（中文，与 script.md 逐镜文案一致） |
| `visual` | string | ✅ | 画面描述（中文，给人看的） |
| `image_prompt` | string | ✅ | 英文 SDXL 首帧提示词 |
| `motion_prompt` | string | ✅ | 英文 LTX 运动描述 |
| `duration_seconds` | number | ✅ | 3~5 秒 |

## 五种最常见的写错方式

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

## 写文件方式

必须调用 `write_json` 工具（参数 `path="storyboard.json"`，
`data_json` 是完整 JSON 字符串）。只把 JSON 贴在最终回答里而不调用工具，
本步骤失败——虽然 `llm_output.py` 会尝试抢救，但抢救不保证字段完整。

## 交付前自检

- [ ] 顶层是对象，含非空 `shots`？
- [ ] 每镜 7 个字段齐全？
- [ ] `id` 从 1 连续、无重复？
- [ ] `duration_seconds` 之和 = `total_duration_seconds` = 需求时长？
- [ ] 镜头数在 4~8 之间？
- [ ] 有人物时 `characters` 数组非空、`shots[].characters` 正确引用？
- [ ] 没有 Markdown 围栏、没有注释、没有省略号？
