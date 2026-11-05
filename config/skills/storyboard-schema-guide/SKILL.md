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
| `duration_seconds` | number | ✅ | 3~8 秒（硬顶是 LTX 单镜上限约 10.2 秒，长台词优先拆镜而不是拉长单镜） |

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

1. **一句一行**，按播放顺序排列；一镜 1~3 句。单句和整镜的字数受
   **台词字数预算**约束（见下节），不能凭感觉写长句。
2. `speaker` 必须是 `characters` 里已定义的 id，或 `"旁白"`。写了不存在的
   发言人会被降级成旁白（不报错，但那句话就没有角色声线了）。
3. 说话角色必须同时出现在本镜 `characters` 里。
4. **每句写 `emotion`（中文情感短语，8~15 字）**——配音引擎会按它调整
   语气的起伏抑扬，不写则全片用统一的口语化基调（容易像念稿）。
   例：`"emotion": "好奇地打量，语调带笑意"`、`"emotion": "不耐烦地
   拖长音"`、`"emotion": "哭腔，声音发颤"`。写情绪而不是动作。
5. 没有 `dialogue` 的镜头 = 整镜用 `narration` 配一条旁白（旧行为，仍然支持）。
6. 每句在前端得到自己的字幕时间轴，不会整镜一条长字幕。

## 台词字数预算：写多少字才放得进这一镜

中文旁白约 **2.6 字/秒**，每镜还要留约 0.4 秒收尾静音，一镜能容纳的
台词是硬上限，不是建议：

| `duration_seconds` | 本镜台词合计上限 |
| --- | --- |
| 3 秒 | 6 字 |
| 4 秒 | 9 字 |
| 5 秒 | 11 字 |
| 6 秒 | 14 字 |
| 8 秒 | 19 字 |

算法：`字数 ≤ 2.6 × (duration_seconds − 0.4)`（写两句再减 0.25 秒的句间停顿）。
超过这个长度，配音比画面长，下游只能给旁白提速或降速画面来凑，
**口型必然对不上、成片也会比需求时长更长**。

整片同理：所有镜头台词字数之和 ≤ `2.6 × (需求时长 − 0.4 × 镜头数)`。
需求 20 秒、5 镜 → 台词总共不超过约 46 字。要表达的内容更多就
**多加镜头或提炼句子**，而不是把台词塞长。分镜表写出超预算台词时，
`storyboard.py` 会告警并把该镜时长抬到够用长度（但单镜还有 LTX 帧数
上限约 10.2 秒，超出的部分仍然只能靠后期补）。

同一个需求时长下，**镜头数越多、每镜台词越少**才不会撑爆成片：需求 25 秒
写 6 镜平均 4.2 秒，就按上表每镜 9~11 字写，全片台词合计别超过 60 字。

## `characters[].voice` / `characters[].timbre`：不同角色不同嗓子

角色卡可选写音色；**说话角色必须写 `timbre`（IndexTTS 引擎，4090/5090
档默认）并按人设选**——不写则由代码按角色 id 轮转默认音色（能区分角色
但不保证性别匹配）。同一张卡两个都写时 IndexTTS 用 `timbre`，edge-tts
降级时用 `voice`。

`voice`（edge-tts 引擎，4060 档或 IndexTTS 不可用时）：

| 取值 | 听感 |
| --- | --- |
| `zh-CN-YunxiNeural` | 男声，年轻偏轻快 |
| `zh-CN-YunjianNeural` | 男声，中年解说腔 |
| `zh-CN-YunyangNeural` | 男声，新闻播报 |
| `zh-CN-XiaoyiNeural` | 女声，年轻偏甜 |
| `zh-CN-XiaochenNeural` | 女声，温柔 |
| `zh-CN-XiaohanNeural` | 女声，沉稳 |

`timbre`（IndexTTS-2.5 引擎，插件 TimbreModel 目录标准音色，按人设挑）：

| 取值 | 听感 |
| --- | --- |
| `钟离.wav` | 男，低沉威严、帝王气度 |
| `北京大爷.mp3` | 男，京腔喜剧感、唠嗑 |
| `播音中年男.mp3` | 男，中年播音员 |
| `温润男声.mp3` | 男，温润 |
| `电台男主播.mp3` | 男，磁性电台腔 |
| `徐志胜.wav` | 男，喜剧综艺腔 |
| `随性男青.mp3` | 男，随性青年 |
| `沉稳高管.mp3` | 男，沉稳商务 |
| `甜美女声.mp3` | 女，甜美 |
| `御姐配音.wav` | 女，成熟御姐 |
| `温柔学姐.mp3` | 女，温柔 |
| `阅历姐姐.mp3` | 女，沉稳阅历 |
| `温暖闺蜜.mp3` | 女，邻家温暖 |
| `傲娇御姐.mp3` | 女，傲娇 |
| `花甲奶奶.mp3` | 女，老年奶奶 |

选角规则：按角色的性别 + 年龄 + 气质挑最接近的一档（老年帝王选
`钟离.wav`、东北糙汉选 `北京大爷.mp3`、少女选 `甜美女声.mp3`……）；
不同角色**必须不同** timbre；不说话的角色（旁白镜出场）不用写。

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
下游按分镜时长生成，成片就短 4 秒，质检 `duration_match`（±3s）会失败。
反向也一样：台词写太长（见「台词字数预算」）会把成片顶长，同样过不了。

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
- [ ] 每镜台词字数 ≤ `2.6 × (duration_seconds − 0.4)`、全片台词字数 ≤ `2.6 × (需求时长 − 0.4 × 镜头数)`？
- [ ] 说话角色的 `motion_prompt` 带口部动作？
- [ ] `id` 从 1 连续、无重复？
- [ ] `duration_seconds` 之和 = `total_duration_seconds` = 需求时长？
- [ ] 镜头数在 4~8 之间？
- [ ] 有人物时 `characters` 数组非空、`shots[].characters` 正确引用？
- [ ] 没有 Markdown 围栏、没有注释、没有省略号？
