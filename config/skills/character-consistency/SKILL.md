---
name: character-consistency
description: 角色卡（characters 数组）与跨镜长相一致性的写法规范：英文外观描述要素、逐字复现规则、出镜角色 id 标注、IPAdapter 开启时的构图要求。当分镜里有人物/动物拟人角色，或成片出现同一角色前后长相不一致时使用。
license: MIT
metadata:
  author: prompt2vid
  version: "1.0"
compatibility: 需要 storyboard.json 的 characters 字段；IPAdapter 相关段落仅在 COMFYUI_IPADAPTER=true 时适用
---

# 角色一致性规范

全片角色长得一样，**只靠一件事**：每一镜的 `image_prompt` 里逐字出现同一段
英文外观描述。没有别的魔法。

## 1. 角色卡写法（`characters[].description`）

必须是**英文**、**具体到可复现**、**一段固定文本**，覆盖这 6 个维度：

1. 年龄与体型：`a 24-year-old slim woman`
2. 脸型与肤色：`oval face, fair skin, slight freckles`
3. 发型发色（含长度）：`shoulder-length straight black hair with side bangs`
4. 服装（款式 + 颜色，上下装分开）：`cream knit sweater, dark blue jeans`
5. 标志性配饰：`round silver glasses, small gold hoop earrings`
6. 全片不变的特征：`a small mole under left eye`

禁止写法：

| 禁止 | 原因 |
| --- | --- |
| `a beautiful girl` | 无信息，每镜都不同 |
| `年轻女性` / 任何中文 | SDXL 文本编码器不懂中文 |
| `she looks sad` | 情绪不属于外观，且会锁死所有镜头的表情 |
| `wearing different clothes each scene` | 会诱导模型换装 |
| 把 `visual`（画面描述）整段抄进 description | 描述里含动作/镜头信息，会污染每一镜 |

参考模板：[assets/character-card-template.json](assets/character-card-template.json)

## 2. 逐字复现规则（硬性）

`image_prompt` 必须**原样包含**该镜所有出场角色的 `description` 原文：

- 一字不改，不改大小写，不改标点，不翻译
- 多个角色按 `characters` 数组的 id 升序拼接，用 `, ` 分隔
- 角色段放在 prompt **最前面**，后面再接动作 / 场景 / 镜头段
- `description` 里已经有的词（发色、服装）不要在后面重复第二遍

反例：

```json
{"image_prompt": "a woman with black hair, medium shot"}          // ✗ 没有逐字复现
{"image_prompt": "A 24-year-old slim woman... (改了标点)"}          // ✗ 改写了原文
```

## 3. 出镜角色标注（`shots[].characters`）

- 列出该镜真实出镜角色的 id，空镜写 `[]` 或不写（**不要为了凑字段写全部角色**）
- 只写 id，不写名字
- 角色首次出场的镜头，构图给到 `medium shot` 及以上，让观众记住长相

## 4. 无人物的镜头

纯风景 / 纯道具空镜不要写 `characters`，也不要在 prompt 里塞人物描述——
空镜里出现"半个人"是最常见的失败。

## 5. 开启 IPAdapter 时（`COMFYUI_IPADAPTER=true`）

代码会自动生成每个角色的定妆照并注入参考图（权重 0.8），此时：

- 角色卡里的**服装颜色要显眼且唯一**（IPAdapter 对大面积色块最敏感）
- 避免遮挡脸部的构图：`face clearly visible`，不要 `back view`、`from behind`
- 不要在 description 里写会变的东西（湿发、逆光剪影）
- 每个角色的定妆照只生成一次；角色卡写得含糊 = 参考图也含糊

## 6. 自检

- [ ] 每个出场角色的 description 都是英文、≥ 20 个词？
- [ ] 6 个维度（年龄体型 / 脸 / 发型 / 服装 / 配饰 / 标志特征）都有？
- [ ] 每镜 image_prompt 逐字含本镜全部角色 description？
- [ ] `shots[].characters` 只列真实出镜的角色？
- [ ] 没有把动作、情绪、镜头信息写进 description？
