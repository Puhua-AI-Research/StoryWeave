# 普华开放API平台 - 开发者指南

## 1. 平台简介

普华开放API平台针对生成式大模型的应用落地的多种场景需求，助力开发者和企业聚焦产品创新，提供高性能、易上手、安全可靠的大模型服务，覆盖文本、图像、视频等多模态场景。

### 主要特性

- **多模态支持**：文本生成、图像生成、视频生成、文本嵌入、OCR识别
- **OpenAI兼容**：完全兼容OpenAI SDK，快速接入
- **高性能稳定**：企业级可靠性保障
- **灵活配置**：支持多种参数定制化调整
- **实时流式**：支持流式输出，提升用户体验

## 2. 快速开始

### 2.1 安装依赖

使用 Python 的 OpenAI SDK：

```bash
pip install openai
```

### 2.2 基础配置

```python
from openai import OpenAI

# 配置 API 基础地址和密钥
base_url = "https://ph8.co/openai/v1"  # 或 https://ph8.co/v1
api_key = "your-api-key-here"

client = OpenAI(base_url=base_url, api_key=api_key)
```

### 2.3 API 端点列表

| 服务类型 | 端点路径 | 说明 |
|---------|---------|------|
| 文本生成 | `/v1/chat/completions` | 对话式文本生成 |
| 图像生成 | `/v1/images/generations` | 文生图、图生图 |
| 视频生成 | `/v1/videos` | 文生视频、图生视频 |
| 文本嵌入 | `/v1/embeddings` | 文本向量化 |
| OCR识别 | `/v1/chat/completions` | 图像文字识别 |

## 3. 文本生成（LLM）

### 3.1 基础对话

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://ph8.co/openai/v1",
    api_key="your-api-key"
)

response = client.chat.completions.create(
    model="model_name",  # 替换为实际模型名称
    messages=[
        {"role": "user", "content": "介绍一下北京的旅游景点"}
    ],
    max_tokens=2048,
    temperature=0.7,
    stream=False
)

print(response.choices[0].message.content)
```

### 3.2 流式输出

```python
response = client.chat.completions.create(
    model="model_name",
    messages=[
        {"role": "user", "content": "写一首关于春天的诗"}
    ],
    max_tokens=1024,
    temperature=0.7,
    stream=True  # 启用流式输出
)

# 逐块输出响应
for chunk in response:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
print()
```

### 3.3 参数说明

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `model` | string | 模型名称 | 必填 |
| `messages` | array | 对话消息列表 | 必填 |
| `max_tokens` | integer | 最大生成token数 | 2048 |
| `temperature` | float | 随机性控制（0-2） | 1.0 |
| `stream` | boolean | 是否流式输出 | false |
| `top_p` | float | 核采样参数（0-1） | 1.0 |

## 4. 图像生成

### 4.1 文生图（Text-to-Image）

```python
from openai import OpenAI
import base64

client = OpenAI(
    base_url="https://ph8.co/openai/v1",
    api_key="your-api-key"
)

result = client.images.generate(
    model="model_name",
    prompt="一只可爱的猫咪在花园里玩耍，阳光明媚",
    size="1024x1024",  # 图像尺寸
    n=1,  # 生成数量
    response_format="b64_json"  # 返回格式：url 或 b64_json
)

# 保存图像
image_base64 = result.data[0].b64_json
image_bytes = base64.b64decode(image_base64)
with open("output.png", "wb") as f:
    f.write(image_bytes)
```

### 4.2 图生图（Image-to-Image）

#### 基础图生图示例

```python
result = client.images.generate(
    model="model_name",  # 替换为实际的图生图模型名称
    prompt="改成爱心形状的泡泡",
    size="adaptive",  # 自适应尺寸
    response_format="url",
    extra_body={
        "image": "https://example.com/input-image.jpg",  # 参考图像
        "seed": 42,  # 随机种子
        "guidance_scale": 5.5,  # 引导强度
        "watermark": False  # 是否添加水印
    }
)

print(result.data[0].url)
```

#### 高级图生图示例（支持参考类型）

```python
result = client.images.generate(
    model="model_name",  # 替换为实际的图生图模型名称
    prompt="未来战士身穿科幻装甲，史诗场景",
    size="16:9",  # 比例：1:1, 16:9, 9:16, 4:3, 3:4
    response_format="url",
    extra_body={
        "image": "https://example.com/reference.png",  # 参考图片
        "image_reference": "subject",  # 参考类型：subject（主体特征）或 face（人物长相）
        "image_fidelity": 0.5,  # 参考强度 [0,1]
        "resolution": "2k",  # 分辨率：1k 或 2k
        "negative_prompt": "低质量，模糊"  # 负面提示词
    }
)
```

### 4.3 图像生成参数说明

| 参数 | 类型 | 说明 | 可选值 |
|------|------|------|--------|
| `model` | string | 模型名称 | 必填 |
| `prompt` | string | 图像描述提示词 | 必填 |
| `size` | string | 图像尺寸或比例 | "1024x1024", "16:9", "adaptive" 等 |
| `n` | integer | 生成图片数量 | 1-10 |
| `response_format` | string | 返回格式 | "url" 或 "b64_json" |
| `extra_body.image` | string | 参考图像（图生图） | URL 或 base64 |
| `extra_body.seed` | integer | 随机种子 | 任意整数 |
| `extra_body.guidance_scale` | float | 引导强度 | 1.0-20.0 |

## 5. 视频生成

### 5.1 文生视频（Text-to-Video）

```python
from openai import OpenAI
import sys
import time

client = OpenAI(
    base_url="https://ph8.co/openai/v1",
    api_key="your-api-key"
)

# 创建视频生成任务
video = client.videos.create(
    model="model_name",
    prompt="一辆青色跑车在沙漠公路上疾驰，热浪翻腾，烈日当空",
    size="1280x720",  # 视频尺寸
    seconds="8"  # 视频时长（秒）
)

print("视频生成已启动:", video)

# 进度条展示
progress = getattr(video, "progress", 0)
bar_length = 30

while video.status in ("in_progress", "queued"):
    video = client.videos.retrieve(video.id)
    progress = getattr(video, "progress", 0)
    
    filled_length = int((progress / 100) * bar_length)
    bar = "=" * filled_length + "-" * (bar_length - filled_length)
    status_text = "排队中" if video.status == "queued" else "生成中"
    
    sys.stdout.write(f"\r{status_text}: [{bar}] {progress:.1f}%")
    sys.stdout.flush()
    time.sleep(2)

sys.stdout.write("\n")

# 检查生成状态
if video.status == "failed":
    message = getattr(
        getattr(video, "error", None), "message", "视频生成失败"
    )
    print(message)
    exit(1)

print("视频生成完成:", video)

# 下载视频
print("正在下载视频...")
time.sleep(5)  # 等待视频准备就绪
content = client.videos.download_content(video.id, variant="video")
content.write_to_file("output.mp4")

print("视频已保存为 output.mp4")
```

### 5.2 图生视频（Image-to-Video）

#### 简化模式

```python
import base64
import requests

# 准备参考图像
response = requests.get("https://example.com/image.png")
image_base64 = base64.b64encode(response.content).decode('utf-8')
base64_image = f"data:image/png;base64,{image_base64}"

# 创建图生视频任务
video = client.videos.create(
    model="model_name",  # 替换为实际的视频生成模型名称
    prompt="一只酷猫骑摩托车穿越夜晚的城市",
    extra_body={
        "image": base64_image,  # 首帧图像
        "duration": 5,  # 时长（秒）：4-12
        "resolution": "1080p",  # 分辨率：720p 或 1080p
        "ratio": "16:9",  # 比例：16:9, 9:16, 1:1, adaptive
        "camerafixed": False,  # 镜头：True=静态，False=动态
        "seed": 12345,  # 随机种子
        "watermark": False  # 水印
    }
)
```

#### 高级模式（支持首帧/尾帧/参考图）

```python
video = client.videos.create(
    model="model_name",  # 替换为实际的视频生成模型名称
    prompt="一只酷猫骑摩托车穿越夜晚的城市",
    extra_body={
        "content": [
            {
                "type": "text",
                "text": "一只酷猫骑摩托车穿越夜晚的城市 --ratio 16:9 --duration 5 --seed 42"
            },
            {
                "type": "image_url",
                "image_url": {"url": "https://example.com/image.png"},
                "role": "first_frame"  # 选项：first_frame（首帧）, last_frame（尾帧）, reference_image（参考图）
            }
        ],
        "resolution": "1080p",
        "service_tier": "default",  # default（标准）或 flex（经济模式，价格更优惠）
        "generate_audio": False,  # 是否生成音频
        "return_last_frame": True,  # 是否返回最后一帧
        "execution_expires_after": 3600,  # 超时时间（秒）
        "watermark": False
    }
)
```

### 5.3 视频生成参数说明

| 参数 | 类型 | 说明 | 可选值 |
|------|------|------|--------|
| `model` | string | 模型名称 | 必填 |
| `prompt` | string | 视频描述提示词 | 必填 |
| `size` | string | 视频尺寸 | "1280x720", "1920x1080" 等 |
| `seconds` | string | 视频时长 | "4"-"12" |
| `extra_body.duration` | integer | 时长（秒） | 4-12 或 -1（自动） |
| `extra_body.resolution` | string | 分辨率 | "720p" 或 "1080p" |
| `extra_body.ratio` | string | 宽高比 | "16:9", "9:16", "1:1", "adaptive" |
| `extra_body.camerafixed` | boolean | 镜头固定 | true（静态）或 false（动态） |
| `extra_body.image` | string | 参考图像 | URL 或 base64 |

## 6. 文本嵌入（Embedding）

文本嵌入用于将文本转换为向量表示，常用于语义搜索、文本相似度计算等场景。

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://ph8.co/v1",
    api_key="your-api-key"
)

embedding = client.embeddings.create(
    model="model_name",
    input="普华开放API平台提供高性能的大模型服务"
)

# 获取向量
vector = embedding.data[0].embedding
print(f"向量维度: {len(vector)}")
print(f"向量前10个值: {vector[:10]}")
```

### 批量文本嵌入

```python
texts = [
    "第一段文本内容",
    "第二段文本内容",
    "第三段文本内容"
]

embedding = client.embeddings.create(
    model="model_name",
    input=texts
)

# 获取所有向量
for i, data in enumerate(embedding.data):
    print(f"文本 {i+1} 的向量维度: {len(data.embedding)}")
```

## 7. OCR 识别

OCR（光学字符识别）用于从图像中提取文字内容。

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://ph8.co/openai/v1",
    api_key="your-api-key"
)

response = client.chat.completions.create(
    model="model_name",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://example.com/receipt.png"  # 图片URL
                    }
                },
                {
                    "type": "text",
                    "text": "请识别图片中的文字内容"
                }
            ]
        }
    ],
    max_tokens=1024,
    temperature=0.1,
    stream=False
)

print(response.choices[0].message.content)
```

### 流式 OCR

```python
response = client.chat.completions.create(
    model="model_name",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "https://example.com/document.jpg"}},
                {"type": "text", "text": "OCR:"}
            ]
        }
    ],
    max_tokens=2048,
    temperature=0.1,
    stream=True
)

for chunk in response:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
print()
```

## 8. 错误处理

### 常见错误码

| 错误码 | 说明 | 解决方案 |
|--------|------|----------|
| 401 | 认证失败 | 检查 API Key 是否正确 |
| 403 | 权限不足 | 确认账户权限和配额 |
| 429 | 请求过于频繁 | 降低请求频率或升级套餐 |
| 500 | 服务器错误 | 稍后重试或联系技术支持 |

### 错误处理示例

```python
from openai import OpenAI, APIError, RateLimitError

client = OpenAI(
    base_url="https://ph8.co/openai/v1",
    api_key="your-api-key"
)

try:
    response = client.chat.completions.create(
        model="model_name",
        messages=[{"role": "user", "content": "你好"}]
    )
    print(response.choices[0].message.content)
    
except RateLimitError:
    print("请求频率超限，请稍后再试")
    
except APIError as e:
    print(f"API 错误: {e}")
    
except Exception as e:
    print(f"未知错误: {e}")
```

## 9. 最佳实践

### 9.1 环境变量配置

建议使用环境变量管理敏感信息：

```python
import os
from openai import OpenAI

base_url = os.environ.get("OPENAI_BASE_URL", "https://ph8.co/openai/v1")
api_key = os.environ.get("OPENAI_API_KEY")

client = OpenAI(base_url=base_url, api_key=api_key)
```

在终端设置环境变量：

```bash
export OPENAI_BASE_URL="https://ph8.co/openai/v1"
export OPENAI_API_KEY="your-api-key"
```

### 9.2 超时设置

设置请求超时避免长时间等待：

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://ph8.co/openai/v1",
    api_key="your-api-key",
    timeout=30.0  # 30秒超时
)
```

### 9.3 重试机制

实现自动重试提高稳定性：

```python
import time
from openai import OpenAI, APIError

client = OpenAI(
    base_url="https://ph8.co/openai/v1",
    api_key="your-api-key"
)

def call_with_retry(func, max_retries=3):
    for attempt in range(max_retries):
        try:
            return func()
        except APIError as e:
            if attempt == max_retries - 1:
                raise
            wait_time = 2 ** attempt  # 指数退避
            print(f"请求失败，{wait_time}秒后重试...")
            time.sleep(wait_time)

# 使用示例
response = call_with_retry(
    lambda: client.chat.completions.create(
        model="model_name",
        messages=[{"role": "user", "content": "你好"}]
    )
)
```

## 10. 常见问题（FAQ）

### Q1: 支持哪些编程语言？
A: 平台兼容 OpenAI SDK，支持 Python、Node.js、Go、Java 等多种语言。推荐使用官方 OpenAI SDK。

### Q2: API 有调用频率限制吗？
A: 有限制，具体根据您的套餐而定。建议实现请求频率控制和重试机制。

### Q3: 如何选择合适的模型？
A: 根据具体场景选择：
- 文本生成：选择适合对话或创作的 LLM
- 图像生成：文生图选择 t2i 模型，图生图选择 i2i 模型
- 视频生成：根据时长和质量要求选择对应模型

### Q4: 生成的内容可以商用吗？
A: 请参考服务条款，一般情况下生成内容的使用权归您所有。

### Q5: 如何优化生成质量？
A: 
- 提示词：使用清晰、详细的描述
- 参数调整：适当调整 temperature、guidance_scale 等参数
- 多次尝试：可以使用不同的 seed 值生成多个结果选择最佳

## 11. 技术支持

如有疑问或需要帮助，请访问我们的官方文档：

- 文档：https://ph8.co/docs

---

**版本**: v1.0  
**更新时间**: 2026年

