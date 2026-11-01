# SDXL 提示词词汇表（竖屏短视频）

## 景别（Shot size）

| 英文 | 中文 | 稳定性 | 适用 |
| --- | --- | --- | --- |
| `extreme close-up` | 大特写 | 高 | 眼睛、手部道具、食物纹理 |
| `close-up` | 特写 | 高 | 表情、情绪转折 |
| `medium close-up` | 近景 | 高 | 半身 + 少量环境（**默认首选**） |
| `medium shot` | 中景 | 中高 | 腰部以上 + 环境 |
| `medium full shot` | 中全景 | 中 | 膝以上 |
| `full body shot` | 全身 | 低 | 只在必须有全身信息时用 |
| `wide shot` / `establishing shot` | 远景 / 建立镜头 | 低 | 空镜、城市全景 |

## 机位与角度（Camera）

`eye level`（默认）、`slightly low angle`（显气势）、`slightly high angle`（显亲和）、
`over-the-shoulder`、`dutch angle`（慎用）、`top-down flat lay`（静物）

## 机位距离感（竖屏）

`vertical composition`、`centered subject`、`rule of thirds`、
`subject fills upper two thirds`、`shallow depth of field, blurred background`

## 光线（Lighting）

- 自然：`soft morning light`、`warm golden hour`、`overcast diffused light`、
  `dappled sunlight through leaves`
- 人造：`warm practical lamp light`、`neon glow`、`studio softbox lighting`
- 戏剧：`cinematic rim light`、`strong backlight, lens flare`、`chiaroscuro`

## 色彩与影调（Color / Tone）

`muted warm palette`、`cool blue palette`、`high saturation`、`desaturated`、
`pastel tones`、`teal and orange grade`、`monochrome with one accent color`

## 质感与媒介（Texture / Medium）

- 写实：`35mm film still`、`documentary photography`、`shallow depth of field`
- 动漫：`2d cel shaded`、`clean line art`、`anime key visual`
- 插画：`storybook illustration`、`gouache texture`、`flat vector illustration`
- 复古：`vintage Kodachrome look`、`slight film grain`

> 注意：媒介词与 `PROMPT2VID_STYLE` 档位冲突时以档位为准（代码层会追加并排斥）。

## 环境与道具（高频场景词）

公园：`park path`、`ginkgo leaves`、`wooden bench`、`pond with lotus`
街道：`narrow alley`、`storefront signs`、`wet asphalt reflections`、`bicycle`
室内：`window seat`、`wooden desk`、`steaming mug`、`bookshelf`
自然：`misty mountain ridge`、`rice terrace`、`seaside rocks`、`snow covered pines`

## 情绪的可画化改写（情绪 → 可见细节）

| 情绪 | 不要写 | 改写成 |
| --- | --- | --- |
| 孤独 | `lonely` | `sitting alone on a long bench, empty space beside her` |
| 温暖 | `warm feeling` | `both hands wrapped around a steaming mug, soft smile` |
| 紧张 | `nervous` | `fingers gripping the edge of the table, eyes wide` |
| 疲惫 | `tired` | `slumped shoulders, half-closed eyes, loosened tie` |
| 怀旧 | `nostalgic` | `old photo in hand, faded colors, dust in sunbeam` |
