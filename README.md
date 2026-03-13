# AIGC 文本检测接口说明

## 部署信息

- **URL**


http://127.0.0.1:18082/predict


- **端口**


18082


⚠️ 当前端口 **未开放防火墙**，因此需要使用本地地址：


127.0.0.1


---

# 请求方式

## HTTP 方法


POST


## 请求头

```http
Content-Type: application/json
```
请求体格式

目前为一个简单 JSON 字典，仅包含一个字段：
```
{
  "texts": "这是一个测试文本"
}
```
Bash 测试脚本

在服务器 bash 中运行：
```
curl -X POST "http://127.0.0.1:18082/predict" \
-H "Content-Type: application/json" \
-d '{"texts":"这是一个测试文本"}'
```
返回格式
```
{
  "results": [
    {
      "score": 0.18978093564510345,
      "probability": 0.5473033427967766,
      "is_ai": true
    }
  ]
}
```
字段说明：

字段	说明
score	模型原始预测分数（业务上通常不需要）
probability	AI生成概率（0-1之间，越接近1表示越可能是AI生成）
is_ai	根据算法端阈值判断的结果

⚠️ 推荐后端使用 probability 自行判断是否为 AI 文本，而不是直接使用 is_ai。

真实数据测试样例

请求：
```
{
  "texts": "In this article we analyze the impact of B-physics and Higgs physics at LEPon standard and non-standard Higgs bosons searches at the Tevatron and the LHC,within the framework of minimal flavor violating supersymmetric models. TheB-physics constraints we consider come from the experimental measurements ofthe rare B-decays b -> s gamma and B_u -> tau nu and the experimental limit onthe B_s -> mu+ mu- branching ratio. We show that these constraints are severefor large values of the trilinear soft breaking parameter A_t, rendering thenon-standard Higgs searches at hadron colliders less promising. On the contrarythese bounds are relaxed for small values of A_t and large values of theHiggsino mass parameter mu, enhancing the prospects for the direct detection ofnon-standard Higgs bosons at both colliders. We also consider the availableATLAS and CMS projected sensitivities in the standard model Higgs searchchannels, and we discuss the LHC's ability in probing the whole MSSM parameterspace. In addition we also consider the expected Tevatron collidersensitivities in the standard model Higgs h -> b bbar channel to show that itmay be able to find 3 sigma evidence in the B-physics allowed regions for smallor moderate values of the stop mixing parameter"
}
```
该样本的 真实标签为：非 AI 生成文本。

错误码说明
## 输入错误

| 错误码 | 错误名称 | 说明 |
|------|------|------|
| 1001 | TEXT_TOO_LONG_CHAR | 传入文本字符长度过长 |
| 1002 | TEXT_TOO_LONG_TOKEN | 文本 token 数量超过模型最大限制 |
| 1003 | INVALID_REQUEST | 非法请求 |

## 模型错误

| 错误码 | 错误名称 | 说明 |
|------|------|------|
| 2001 | MODEL_INFERENCE_ERROR | 模型推理失败 |
| 2002 | MODEL_LOAD_ERROR | 模型加载失败 |

需要重点关注错误：

1002 TEXT_TOO_LONG_TOKEN

重点关注错误1002，实际一段文本的长度与token计算之间有差异，并且这需要一些huggingface和torch相关包才能计算，因为在后端你为了判断token长度安装这些库很没必要，所以直接在算法端判断，如果长度超过就返回这一错误，你需要自己写判断，将文本太长的结果返回给用户。
