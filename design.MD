# StoryWeave 产品设计文档

> **产品名称**: StoryWeave（故事编织者）  
> **文档版本**: v2.3.0  
> **更新日期**: 2026-02-01  
> **产品定位**: AI 驱动的智能短剧创作平台

---

## 目录

- [一、产品概览](#一产品概览)
- [二、技术架构](#二技术架构)
- [三、核心功能模块](#三核心功能模块)
- [四、用户操作流程](#四用户操作流程)
- [五、数据架构设计](#五数据架构设计)
- [六、产品方案](#六产品方案)
- [七、商业价值](#七商业价值)
- [八、产品路线图](#八产品路线图)

---

## 一、产品概览

### 1.1 产品愿景

StoryWeave 致力于成为全球领先的 AI 驱动短剧创作平台，通过先进的人工智能技术降低内容创作门槛，让每个人都能轻松创作出专业级的短剧视频内容。

### 1.2 产品定义

StoryWeave 是一款面向内容创作者的 AI 短剧创作 SaaS 平台，提供从原始文本到成品视频的全流程自动化创作工具。平台整合了大语言模型、图像生成模型和视频生成模型，通过智能化的工作流引擎，实现剧本生成、分镜设计、素材制作和视频合成的一站式服务。

### 1.3 核心价值主张

#### 对创作者
- **零门槛创作**：无需编剧、绘画或视频制作技能，仅需提供故事文本
- **10倍效率提升**：传统短剧制作周期从数周缩短至数小时
- **成本节约 90%**：无需雇佣团队，AI 完成 90% 重复性工作
- **创意可控**：AI 生成初稿，人工精修优化，保持创意主导权

#### 对企业客户
- **批量生产能力**：支持同时创作多个项目，规模化内容产出
- **品质标准化**：统一的制作流程，保证内容质量一致性
- **灵活定制**：支持自定义风格、素材库和工作流
- **数据驱动**：完整的数据追踪，优化创作流程

### 1.4 目标用户画像

#### 核心用户群体

**1. 自媒体创作者**
- 年龄：25-40岁
- 痛点：内容产出压力大，制作成本高
- 需求：快速产出高质量短剧内容，提升账号活跃度

**2. 短剧创作团队**
- 规模：3-10人小团队
- 痛点：人力成本高，制作周期长
- 需求：提升团队效率，降低制作成本

**3. 影视制作公司**
- 规模：10-50人中型公司
- 痛点：前期创意验证成本高
- 需求：快速制作样片验证创意，降低试错成本

**4. 品牌营销团队**
- 行业：电商、教育、娱乐等
- 痛点：营销视频制作周期长，成本高
- 需求：快速产出符合品牌调性的营销短剧

#### 用户规模预估
- 第一年：5,000 注册用户，500 付费用户
- 第二年：20,000 注册用户，3,000 付费用户
- 第三年：100,000 注册用户，15,000 付费用户

### 1.5 产品定位

**市场定位**：AI 短剧创作工具的领导者

**竞争优势**：
1. **技术领先**：多模态 AI 协同工作流，业界首创
2. **全流程覆盖**：从文本到视频的完整链路
3. **开放生态**：支持多家 AI 服务商，用户自由选择
4. **易用性**：直观的可视化界面，无需技术背景

### 1.6 技术栈总览

#### 前端技术栈
| 技术类别 | 技术选型 | 版本 | 选型理由 |
|---------|---------|------|---------|
| 核心框架 | React | 18.x+ | 生态成熟，社区活跃，企业级首选 |
| 开发语言 | TypeScript | 5.x+ | 类型安全，提升代码质量 |
| 状态管理 | Redux Toolkit | 2.x+ | 官方推荐，简化 Redux 使用 |
| 路由管理 | React Router | 6.x+ | 声明式路由，易于维护 |
| UI 组件库 | Ant Design | 5.x+ | 企业级组件，开箱即用 |
| 数据可视化 | ECharts | 5.x+ | 功能强大，适合数据展示 |
| HTTP 请求 | Axios | 1.x+ | 功能完整，拦截器支持 |
| 实时通信 | Socket.io-client | 4.x+ | WebSocket 封装，自动重连 |
| 表单管理 | React Hook Form | 7.x+ | 性能优秀，API 简洁 |
| 样式方案 | CSS Modules + Sass | - | 模块化样式，避免冲突 |
| 构建工具 | Vite | 5.x+ | 极速构建，现代化工具 |

#### 后端技术栈
| 技术类别 | 技术选型 | 版本 | 选型理由 |
|---------|---------|------|---------|
| 核心框架 | FastAPI | 0.109+ | 现代化、高性能、原生异步支持 |
| 开发语言 | Python | 3.11+ | 简洁易读，AI 库丰富 |
| ORM | SQLAlchemy | 2.x+ | 强大的 ORM，支持异步 |
| 数据验证 | Pydantic | 2.x+ | 类型验证，FastAPI 内置 |
| 异步任务 | asyncio + 线程池 | - | 原生异步处理，轻量高效 |
| 定时任务 | APScheduler | 3.x+ | Python 定时任务调度库 |
| 消息队列 | Redis | 7.x+ | 高性能缓存和消息代理 |
| 数据库 | PostgreSQL | 15.x+ | 功能强大，支持 JSON 字段 |
| 文件存储 | MinIO / S3 | - | 对象存储，适合图片视频 |
| 实时通信 | FastAPI WebSocket | - | 原生 WebSocket 支持 |
| AI 集成 | LangChain | 0.1.x+ | AI 工作流编排 |
| 认证授权 | python-jose | 3.x+ | JWT Token 认证 |
| API 文档 | FastAPI 内置 | - | 自动生成 OpenAPI/Swagger 文档 |
| 监控日志 | Sentry | - | 错误追踪和性能监控 |

#### 基础设施
| 组件 | 技术选型 | 用途 |
|------|---------|------|
| 容器化 | Docker | 应用容器化部署 |
| 编排 | Docker Compose / K8s | 服务编排 |
| 反向代理 | Nginx | 静态文件服务和反向代理 |
| 对象存储 | 自建 MinIO/S3 | 图片视频存储 |

---

## 二、技术架构

### 2.1 整体架构设计

```
┌─────────────────────────────────────────────────────────────┐
│                         客户端层                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  Web 应用     │  │  移动端 H5    │  │  桌面客户端   │      │
│  │  (React)     │  │  (React)     │  │  (Electron)  │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                           │  HTTPS / WebSocket
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                         网关层                                │
│  ┌──────────────────────────────────────────┐               │
│  │  Nginx 反向代理 + 负载均衡                │               │
│  │  - SSL 终止                               │               │
│  │  - 静态资源服务                           │               │
│  │  - API 路由                               │               │
│  └──────────────────────────────────────────┘               │
└─────────────────────────────────────────────────────────────┘
                           │
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                      应用服务层                               │
│  ┌────────────────┐         ┌────────────────┐              │
│  │  FastAPI 应用   │         │  WebSocket     │              │
│  │  - REST API    │         │  - 实时通信     │              │
│  │  - 业务逻辑     │         │  - 任务推送     │              │
│  │  - 异步处理     │         │  - 原生支持     │              │
│  └────────────────┘         └────────────────┘              │
└─────────────────────────────────────────────────────────────┘
                           │
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                      任务处理层                               │
│  ┌────────────────┐         ┌────────────────┐              │
│  │  后台线程池     │         │  Redis         │              │
│  │  - AI 生成任务  │←────────│  - 任务状态     │              │
│  │  - 异步处理     │         │  - 缓存         │              │
│  │  - 定时调度     │         │  - Session     │              │
│  └────────────────┘         └────────────────┘              │
└─────────────────────────────────────────────────────────────┘
                           │
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                      数据存储层                               │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐            │
│  │ PostgreSQL │  │  MinIO/S3  │  │   Redis    │            │
│  │ - 业务数据  │  │  - 文件存储 │  │  - 缓存     │            │
│  │ - 用户数据  │  │  - 媒体资源 │  │  - 队列     │            │
│  └────────────┘  └────────────┘  └────────────┘            │
└─────────────────────────────────────────────────────────────┘
                           │
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                      外部服务层                               │
│  ┌────────────────────────────────────────────────┐         │
│  │         普华开放 API 平台（ph8.co）             │         │
│  │        （OpenAI 兼容接口标准）                   │         │
│  └────────────────────────────────────────────────┘         │
│    ↓ 统一接口，多模态支持（详见 ph8.md）                    │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐            │
│  │  LLM 服务  │  │  图像服务  │  │  视频服务  │            │
│  │ - GPT-4    │  │ - DALL-E 3 │  │ - Kling    │            │
│  │ - Claude   │  │ - MidJourney│ │ - Gen-2   │            │
│  │ - GPT-3.5  │  │ - SD-XL    │  │ - Pika    │            │
│  └────────────┘  └────────────┘  └────────────┘            │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 前端架构设计

#### 2.2.1 项目目录结构

```
storyweave-web/
├── public/                    # 静态资源
│   ├── favicon.ico
│   └── logo.png
├── src/
│   ├── api/                   # API 接口层
│   │   ├── index.ts          # API 客户端配置
│   │   ├── auth.ts           # 认证相关 API
│   │   ├── project.ts        # 项目相关 API
│   │   ├── asset.ts          # 素材相关 API
│   │   ├── script.ts         # 剧本相关 API
│   │   ├── video.ts          # 视频相关 API
│   │   └── websocket.ts      # WebSocket 客户端
│   ├── assets/               # 静态资源
│   │   ├── images/
│   │   ├── styles/
│   │   │   ├── variables.scss # 全局变量
│   │   │   ├── mixins.scss   # 样式混入
│   │   │   └── global.scss   # 全局样式
│   │   └── fonts/
│   ├── components/           # 通用组件
│   │   ├── common/           # 基础组件
│   │   │   ├── Button/
│   │   │   ├── Input/
│   │   │   ├── Modal/
│   │   │   └── Loading/
│   │   ├── business/         # 业务组件
│   │   │   ├── ProjectCard/
│   │   │   ├── ScriptEditor/
│   │   │   ├── StoryboardCanvas/
│   │   │   └── VideoPlayer/
│   │   └── layout/           # 布局组件
│   │       ├── Header/
│   │       ├── Sidebar/
│   │       └── Footer/
│   ├── pages/                # 页面组件
│   │   ├── Login/           # 登录页
│   │   ├── Dashboard/       # 工作台
│   │   ├── Projects/        # 项目列表
│   │   ├── ProjectDetail/   # 项目详情
│   │   │   ├── Overview/    # 项目概览
│   │   │   ├── TextEditor/  # 原文编辑
│   │   │   ├── Outline/     # 大纲管理
│   │   │   ├── Assets/      # 素材管理
│   │   │   ├── Script/      # 剧本管理
│   │   │   └── Video/       # 视频生成
│   │   ├── Settings/        # 系统设置
│   │   ├── Tasks/           # 任务中心
│   │   └── NotFound/        # 404 页面
│   ├── store/               # Redux 状态管理
│   │   ├── index.ts         # Store 配置
│   │   ├── slices/          # 状态切片
│   │   │   ├── authSlice.ts
│   │   │   ├── projectSlice.ts
│   │   │   ├── assetSlice.ts
│   │   │   ├── scriptSlice.ts
│   │   │   └── videoSlice.ts
│   │   └── middleware/      # 中间件
│   │       └── logger.ts
│   ├── hooks/               # 自定义 Hooks
│   │   ├── useAuth.ts
│   │   ├── useProject.ts
│   │   ├── useWebSocket.ts
│   │   └── useAsync.ts
│   ├── utils/               # 工具函数
│   │   ├── request.ts       # HTTP 请求封装
│   │   ├── storage.ts       # 本地存储
│   │   ├── validate.ts      # 表单验证
│   │   ├── format.ts        # 格式化工具
│   │   └── parser.ts        # 文本解析
│   ├── types/               # TypeScript 类型定义
│   │   ├── api.d.ts         # API 类型
│   │   ├── models.d.ts      # 数据模型
│   │   └── components.d.ts  # 组件类型
│   ├── config/              # 配置文件
│   │   ├── constants.ts     # 常量定义
│   │   ├── routes.ts        # 路由配置
│   │   └── ai-providers.ts  # AI 厂商配置
│   ├── routes/              # 路由定义
│   │   └── index.tsx
│   ├── App.tsx              # 根组件
│   ├── main.tsx             # 入口文件
│   └── vite-env.d.ts        # Vite 类型声明
├── .env.development         # 开发环境配置
├── .env.production          # 生产环境配置
├── package.json
├── tsconfig.json
├── vite.config.ts
├── .eslintrc.js
└── .prettierrc
```

#### 2.2.2 状态管理架构

采用 **Redux Toolkit** 进行状态管理，使用 **Slice** 模式组织代码。

**核心 Slice 设计**：

```typescript
// 认证状态
authSlice: {
  user: User | null
  token: string | null
  isAuthenticated: boolean
  loading: boolean
}

// 项目状态
projectSlice: {
  projects: Project[]
  currentProject: Project | null
  loading: boolean
  error: string | null
}

// 素材状态
assetSlice: {
  assets: Asset[]
  selectedAssets: string[]
  generatingAssets: Record<string, boolean>
}

// 剧本状态
scriptSlice: {
  scripts: Script[]
  currentScript: Script | null
  storyboards: Storyboard[]
}

// 视频状态
videoSlice: {
  configs: VideoConfig[]
  results: VideoResult[]
  polling: boolean
  selectedResults: Record<string, string>
}
```

#### 2.2.3 路由设计

采用 **React Router v6** 嵌套路由设计：

```typescript
路由结构：
/
├── /login                    # 登录页（公开）
├── /register                 # 注册页（公开）
└── /app                      # 应用主体（需认证）
    ├── /dashboard            # 工作台首页
    ├── /projects             # 项目列表
    │   └── /:projectId       # 项目详情
    │       ├── /overview     # 概览
    │       ├── /text         # 原文管理
    │       ├── /outline      # 大纲管理
    │       ├── /assets       # 素材管理
    │       ├── /script       # 剧本管理
    │       └── /video        # 视频生成
    ├── /tasks                # 任务中心
    ├── /settings             # 系统设置
    │   ├── /profile          # 个人信息
    │   ├── /ai-config        # AI 配置
    │   └── /billing          # 账单管理
    └── /help                 # 帮助中心
```

### 2.3 后端架构设计

#### 2.3.1 FastAPI 项目结构

```
storyweave-backend/
├── app/                     # 应用根目录
│   ├── main.py             # FastAPI 应用入口
│   ├── config.py           # 配置管理（Settings）
│   ├── dependencies.py     # 全局依赖注入
│   ├── core/               # 核心模块
│   │   ├── __init__.py
│   │   ├── security.py     # 安全相关（JWT、密码哈希）
│   │   ├── database.py     # 数据库连接和会话管理
│   │   └── config.py       # 配置类
│   ├── api/                # API 路由
│   │   ├── __init__.py
│   │   ├── api_v1/         # API v1 版本
│   │   │   ├── __init__.py
│   │   │   ├── api.py      # 路由聚合
│   │   │   ├── endpoints/  # 端点
│   │   │   │   ├── __init__.py
│   │   │   │   ├── auth.py       # 认证
│   │   │   │   ├── users.py      # 用户
│   │   │   │   ├── projects.py   # 项目
│   │   │   │   ├── assets.py     # 素材
│   │   │   │   ├── scripts.py    # 剧本
│   │   │   │   ├── storyboards.py # 分镜
│   │   │   │   ├── videos.py     # 视频
│   │   │   │   └── tasks.py      # 任务
│   │   │   └── deps.py     # 依赖注入
│   │   └── websocket/      # WebSocket 端点
│   │       ├── __init__.py
│   │       └── storyboard.py # 分镜生成 WebSocket
│   ├── models/             # SQLAlchemy 模型
│   │   ├── __init__.py
│   │   ├── user.py         # 用户模型
│   │   ├── project.py      # 项目模型
│   │   ├── asset.py        # 素材模型
│   │   ├── script.py       # 剧本模型
│   │   ├── storyboard.py   # 分镜模型
│   │   ├── video.py        # 视频模型
│   │   └── task.py         # 任务模型
│   ├── schemas/            # Pydantic 数据模式
│   │   ├── __init__.py
│   │   ├── user.py         # 用户请求/响应模式
│   │   ├── project.py      # 项目请求/响应模式
│   │   ├── asset.py        # 素材请求/响应模式
│   │   ├── script.py       # 剧本请求/响应模式
│   │   ├── storyboard.py   # 分镜请求/响应模式
│   │   ├── video.py        # 视频请求/响应模式
│   │   ├── task.py         # 任务请求/响应模式
│   │   └── common.py       # 通用模式（分页、响应等）
│   ├── services/           # 业务逻辑层
│   │   ├── __init__.py
│   │   ├── auth_service.py # 认证服务
│   │   ├── user_service.py # 用户服务
│   │   ├── project_service.py # 项目服务
│   │   ├── asset_service.py   # 素材服务（异步）
│   │   ├── script_service.py  # 剧本服务
│   │   ├── storyboard_service.py # 分镜服务
│   │   ├── video_service.py   # 视频服务（异步）
│   │   └── ai/             # AI 服务
│   │       ├── __init__.py
│   │       ├── llm.py      # 大语言模型服务
│   │       ├── image.py    # 图像生成服务
│   │       ├── video.py    # 视频生成服务
│   │       └── agents.py   # AI Agent 实现
│   ├── utils/              # 工具函数
│   │   ├── __init__.py
│   │   ├── storage.py      # 文件存储工具
│   │   ├── parser.py       # 文本解析工具
│   │   └── tasks.py        # 任务管理工具
│   └── middleware/         # 中间件
│       ├── __init__.py
│       ├── logging.py      # 日志中间件
│       └── cors.py         # CORS 中间件
├── tasks/                  # 后台任务
│   ├── __init__.py
│   ├── scheduler.py        # APScheduler 配置
│   └── workers.py          # 任务处理器（线程池）
├── tests/                  # 测试用例
│   ├── __init__.py
│   ├── conftest.py         # pytest 配置
│   ├── test_api/           # API 测试
│   └── test_services/      # 服务测试
├── alembic/                # 数据库迁移
│   ├── versions/           # 迁移版本
│   └── env.py              # Alembic 配置
├── requirements/           # 依赖管理
│   ├── base.txt            # 基础依赖
│   ├── dev.txt             # 开发依赖
│   └── prod.txt            # 生产依赖
├── docker/                 # Docker 配置
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── nginx.conf
├── .env.example            # 环境变量示例
├── alembic.ini             # Alembic 配置文件
├── pyproject.toml          # 项目配置（推荐）
└── README.md               # 项目说明
```

#### 2.3.2 数据模型设计

**核心模型关系**：

```
User (用户)
  ↓ 1:N
Project (项目)
  ↓ 1:N
Novel (原始文本)
  ↓ 1:N
Chapter (章节)
  ↓ 1:N
Outline (大纲)
  ↓ 1:N
Script (剧本)
  ↓ 1:N
  ├─→ Asset (素材)
  └─→ Storyboard (分镜)
       ↓ 1:N
       VideoConfig (视频配置)
         ↓ 1:N
         VideoResult (视频结果)
```

#### 2.3.3 API 设计规范

采用 **RESTful API** 设计，遵循以下规范：

**URL 设计**：
```
GET    /api/v1/projects/              # 获取项目列表
POST   /api/v1/projects/              # 创建项目
GET    /api/v1/projects/{id}/         # 获取项目详情
PUT    /api/v1/projects/{id}/         # 更新项目
DELETE /api/v1/projects/{id}/         # 删除项目

POST   /api/v1/assets/generate/       # 生成素材
GET    /api/v1/assets/{id}/status/    # 查询生成状态

POST   /api/v1/videos/generate/       # 生成视频
GET    /api/v1/videos/{id}/results/   # 获取视频结果
```

**响应格式**：
```json
成功响应：
{
  "code": 200,
  "message": "success",
  "data": { ... }
}

错误响应：
{
  "code": 400,
  "message": "参数错误",
  "errors": {
    "field": ["错误信息"]
  }
}
```

#### 2.3.4 异步任务设计

采用 **FastAPI 原生异步方案**：

**技术方案**：
1. **FastAPI 异步路由**：使用 `async def` 处理 I/O 密集型任务（原生支持）
2. **asyncio + 线程池**：结合 asyncio 和 ThreadPoolExecutor 处理混合任务
3. **APScheduler**：处理定时任务（清理过期文件、数据统计）
4. **Redis**：存储任务状态和结果，支持异步操作（aioredis）

**FastAPI 异步优势**：
- 原生异步支持，性能优秀
- 基于 Starlette 和 uvicorn，高并发能力强
- 代码简洁，async/await 语法直观
- 无需额外的 Worker 进程

**任务类型**：
- 素材生成任务（图像生成）- 异步路由 + 线程池
- 视频生成任务（视频合成）- 异步路由 + 线程池
- 批量处理任务（批量导出）- 后台线程处理
- 定时任务（清理过期文件）- APScheduler 定时调度

**任务状态**：
```python
from enum import Enum

class TaskStatus(str, Enum):
    PENDING = "pending"       # 等待执行
    PROCESSING = "processing" # 处理中
    COMPLETED = "completed"   # 已完成
    FAILED = "failed"         # 失败
    CANCELLED = "cancelled"   # 已取消
```

**实现方案**：
```python
from fastapi import FastAPI, BackgroundTasks
from typing import Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor

app = FastAPI()
executor = ThreadPoolExecutor(max_workers=10)

# 异步路由 - 立即返回任务 ID
@app.post("/api/v1/videos/generate/")
async def generate_video(
    config: VideoConfig,
    background_tasks: BackgroundTasks
):
    task_id = create_task_id()
    
    # 方式1: 使用 BackgroundTasks（适合轻量任务）
    background_tasks.add_task(
        process_video_generation, task_id, config
    )
    
    # 方式2: 使用线程池（适合 CPU 密集型）
    loop = asyncio.get_event_loop()
    loop.run_in_executor(
        executor,
        process_video_generation,
        task_id,
        config
    )
    
    return {"task_id": task_id, "status": "pending"}

# 异步查询任务状态
@app.get("/api/v1/tasks/{task_id}/status/")
async def get_task_status(task_id: str):
    # 使用 aioredis 异步查询 Redis
    status = await redis_client.get(f"task:{task_id}")
    return {"task_id": task_id, "status": status}

# 异步处理函数
async def process_video_generation(task_id: str, config: dict):
    try:
        await update_task_status(task_id, TaskStatus.PROCESSING)
        
        # 调用 AI 服务（异步）
        video_result = await ai_service.generate_video(config)
        
        await update_task_status(task_id, TaskStatus.COMPLETED)
        await store_result(task_id, video_result)
    except Exception as e:
        await update_task_status(task_id, TaskStatus.FAILED)
        await store_error(task_id, str(e))
```

**性能对比**：

| 方案 | 并发能力 | 内存占用 | 响应速度 | 复杂度 |
|------|---------|---------|---------|--------|
| Celery | 高 | 高 | 中 | 高 |
| FastAPI 异步 | 极高 | 低 | 极快 | 低 |

### 2.4 通信协议设计

#### HTTP API
- 请求格式：JSON
- 认证方式：JWT Token（Header: `Authorization: Bearer {token}`）
- 超时设置：30秒（普通请求）/ 10分钟（AI 生成请求）

#### WebSocket
- 协议：Socket.io
- 认证：连接时传递 token
- 心跳：每 30 秒发送 ping
- 重连：断线后自动重连，最多 10 次

**消息格式**：
```json
{
  "type": "event_type",
  "data": { ... },
  "timestamp": 1234567890
}
```

---

## 三、核心功能模块

### 3.1 用户认证与权限管理

#### 3.1.1 功能概述
提供安全可靠的用户认证机制和细粒度的权限控制系统。

#### 3.1.2 核心功能

**用户注册**
- 账号密码注册（用户名/邮箱 + 密码）
- 邮箱验证（可选）

**用户登录**
- 账号密码登录（用户名/邮箱 + 密码）
- JWT Token 认证

**权限管理**
- 基于角色的访问控制（RBAC）
- 角色：管理员、创作者、观众
- 权限粒度：项目级、功能级

#### 3.1.3 业务规则

- Token 有效期：7天（可刷新）
- 密码强度：至少 8 位，包含字母和数字
- 登录失败限制：5次失败后锁定账户 30 分钟
- 多端登录：支持，同一账号最多 3 个设备同时在线

---

### 3.2 项目管理模块

#### 3.2.1 功能概述
管理短剧创作项目的完整生命周期，提供项目创建、编辑、归档和删除等功能。

#### 3.2.2 核心功能

**项目创建**
- 基础信息：项目名称、描述、封面
- 项目类型：短剧、广告片、教学视频
- 风格设定：现实主义、动漫风、科幻风
- 目标参数：视频时长、分辨率、纵横比

**项目管理**
- 项目列表：支持筛选、排序、搜索
- 项目详情：查看项目进度和统计数据
- 项目归档：完成的项目可归档
- 项目删除：软删除，保留 30 天可恢复

**协作功能**
- 成员邀请：邀请团队成员协作
- 权限分配：所有者、编辑者、查看者
- 操作日志：记录所有编辑历史

#### 3.2.3 数据统计

项目维度统计：
- 创作进度（原文、大纲、剧本、分镜、视频）
- 素材数量（角色、场景、道具）
- 生成任务数量和成功率
- 耗时统计（各环节用时）

---

### 3.3 原文管理模块

#### 3.3.1 功能概述
导入和管理小说原文，智能解析章节结构，为后续创作提供内容基础。

#### 3.3.2 核心功能

**文本导入**
- 支持格式：Word (.docx)、TXT、Markdown
- 在线编辑器：直接在线输入文本
- 网络导入：从小说网站导入（爬虫）
- 智能去重：自动去除重复章节

**文本解析**
- 章节识别：自动识别章节标题和内容
- 卷册识别：支持卷/部/篇结构
- 数字转换：中文数字自动转为阿拉伯数字
- 内容清洗：去除广告、注释等无关内容

**文本编辑**
- 富文本编辑器：支持格式化文本
- 章节管理：增删改章节
- 批量操作：批量修改章节标题
- 版本管理：保存编辑历史

**智能分析**
- 字数统计：总字数、章节字数
- 人物识别：自动识别主要角色
- 场景识别：自动识别主要场景
- 情节分析：提取核心情节线

#### 3.3.3 解析规则

章节标题匹配规则（正则表达式）：
- `第[一二三四五六七八九十百千万\d]+章`
- `第[一二三四五六七八九十百千万\d]+节`
- `Chapter \d+`
- 自定义规则：用户可自定义匹配规则

---

### 3.4 大纲管理模块

#### 3.4.1 功能概述
创建和管理故事大纲，梳理故事结构，为剧本生成提供框架。

#### 3.4.2 核心功能

**大纲创建**
- 手动创建：自由编辑大纲
- AI 生成：基于原文自动生成大纲
- 模板应用：使用预设大纲模板

**大纲编辑**
- 层级结构：支持多级大纲
- 拖拽排序：可视化调整顺序
- 折叠展开：支持折叠查看
- 富文本：支持格式化内容

**事件线管理**
- 主线：主要情节线
- 支线：次要情节线
- 时间线：事件时间轴
- 人物线：角色发展线

**AI 辅助**
- 大纲生成：基于原文生成大纲
- 大纲扩展：补充大纲细节
- 大纲优化：优化大纲结构
- 冲突检测：检测情节冲突

#### 3.4.3 大纲模板

预设模板：
- 三幕式结构
- 英雄之旅
- 五幕式结构
- 自定义模板

---

### 3.5 素材管理模块

#### 3.5.1 功能概述
管理项目所需的各类视觉素材，支持 AI 生成和手动上传，建立统一的素材库。

#### 3.5.2 核心功能

**素材分类**
- 角色素材：人物角色外观图
- 场景素材：背景场景图
- 道具素材：关键道具图
- 特效素材：特效元素图

**素材来源**
- AI 生成：基于描述生成图片
- 本地上传：上传本地图片
- 在线搜索：搜索免版权图片
- 素材库：从公共素材库选择

**AI 生成流程**
1. 填写素材信息（名称、类型、描述）
2. 上传参考图（可选）
3. AI 生成提示词（可手动编辑）
4. 选择图像模型（MidJourney / DALL-E / Stable Diffusion）
5. 生成多张候选图（通常 4-6 张）
6. 预览并选择最佳结果
7. 保存到素材库

**素材管理**
- 素材预览：网格视图/列表视图
- 素材筛选：按类型、标签、创建时间
- 素材标签：自定义标签分类
- 素材编辑：裁剪、调色、添加滤镜
- 素材删除：软删除，可恢复

**批量操作**
- 批量生成：一次生成多个素材
- 批量上传：拖拽上传多个文件
- 批量标签：批量添加标签
- 批量导出：打包下载素材

#### 3.5.3 生成参数

图像生成参数：
- 分辨率：512x512 / 1024x1024 / 2048x2048
- 风格：写实风、动漫风、水彩风、油画风
- 光照：自然光、工作室光、柔光、硬光
- 视角：正面、侧面、俯视、仰视
- 数量：1-10 张

---

### 3.6 剧本管理模块

#### 3.6.1 功能概述
创建和编辑剧本内容，支持 AI 辅助生成，提供专业的剧本编辑器。

#### 3.6.2 核心功能

**剧本创建**
- 手动创建：使用剧本编辑器编写
- AI 生成：基于大纲和原文生成
- 分集创建：创建多集剧本

**剧本编辑**
- 专业编辑器：场景、人物、对话、旁白
- 格式化：自动格式化剧本格式
- 角色高亮：不同角色不同颜色
- 实时字数：统计剧本字数和时长

**剧本元素**
- 场景：场景描述（内景/外景、时间）
- 人物：角色登场和动作
- 对话：角色对话和语气
- 旁白：画外音和说明

**AI 辅助**
- 剧本生成：基于大纲生成完整剧本
- 对话优化：优化对话流畅度
- 场景扩展：扩展场景描述
- 冲突增强：增强戏剧冲突

**剧本导出**
- 导出格式：PDF、Word、Final Draft
- 导出样式：自定义导出样式

#### 3.6.3 剧本格式

标准剧本格式：
```
场景一：咖啡馆 - 内景 - 白天

小明独自坐在靠窗的位置，手里拿着一本书。窗外阳光明媚。

小红走进咖啡馆，四处张望。

小红
（兴奋地）
小明！好久不见！

小明抬起头，露出惊喜的表情。

小明
（微笑）
小红！快坐！
```

---

### 3.7 分镜生成模块

#### 3.7.1 功能概述
基于剧本内容，通过 AI Agent 协作自动生成分镜图，支持实时预览和交互式编辑。

#### 3.7.2 核心功能

**AI Agent 工作流**

采用多 Agent 协作模式：

1. **导演 Agent (Director)**
   - 职责：统筹全局，分配任务
   - 输入：完整剧本
   - 输出：工作计划和任务分配

2. **片段师 Agent (Segmenter)**
   - 职责：将剧本切分为场景片段
   - 输入：剧本内容
   - 输出：场景片段列表

3. **分镜师 Agent (Shot Designer)**
   - 职责：为每个片段设计分镜
   - 输入：场景片段描述
   - 输出：分镜描述（机位、构图、运镜）

4. **画师 Agent (Artist)**
   - 职责：根据分镜描述生成图像
   - 输入：分镜描述 + 角色素材
   - 输出：分镜图片

**实时生成流程**

通过 WebSocket 实时通信：

```
用户提交剧本
  ↓
建立 WebSocket 连接
  ↓
Director 分析剧本
  ↓ [实时推送思考过程]
Segmenter 切分场景
  ↓ [实时推送切分结果]
Shot Designer 设计分镜
  ↓ [实时推送分镜描述]
Artist 生成图像
  ↓ [实时推送生成进度]
前端接收并展示
```

**分镜编辑**
- 可视化画布：拖拽调整分镜顺序
- 分镜预览：查看单个分镜详情
- 手动编辑：修改分镜描述
- 重新生成：对不满意的分镜重新生成
- 批量操作：批量调整、删除

**分镜参数**
- 机位：特写、近景、中景、全景、远景
- 构图：三分法、对称构图、黄金分割
- 运镜：推拉摇移跟、固定镜头
- 景别：仰拍、俯拍、平拍

#### 3.7.3 生成策略

**图像生成策略**
- 宫格生成：一次生成多个分镜（2x2 或 3x3 网格）
- 自动切分：前端自动切分宫格图为单张分镜
- 风格一致性：使用相同的风格参数保证一致性
- 角色一致性：使用 LoRA 模型保证角色外观一致

**错误处理**
- 生成失败：自动重试 3 次
- 超时处理：超过 2 分钟自动重试
- 降级方案：使用低分辨率快速生成

---

### 3.8 视频配置模块

#### 3.8.1 功能概述
配置视频生成参数，支持多家 AI 视频服务商，提供灵活的配置选项。

#### 3.8.2 支持的 AI 厂商

| 厂商 | 特点 | 图片数量 | 时长 | 分辨率 |
|------|------|---------|------|--------|
| RunwayML | 质量高，速度快 | 1-4 张 | 5-15 秒 | 最高 4K |
| Pika Labs | 性价比高 | 1-9 张 | 3-10 秒 | 最高 1080p |
| 火山引擎 | 国内服务，稳定 | 1-2 张 | 4-12 秒 | 可自定义 |
| Stable Video | 开源，可控性强 | 1 张 | 2-5 秒 | 可自定义 |

#### 3.8.3 视频生成模式

**单图模式 (Image to Video)**
- 输入：1 张图片
- 输出：图片动起来的视频
- 适用：静态场景动态化

**首尾帧模式 (Start-End)**
- 输入：2 张图片（起始帧 + 结束帧）
- 输出：从起始过渡到结束的视频
- 适用：明确的动作变化

**多图模式 (Multi-Image)**
- 输入：3-9 张图片
- 输出：图片序列连接成的视频
- 适用：复杂的动作序列

**文本模式 (Text to Video)**
- 输入：文本描述
- 输出：根据描述生成的视频
- 适用：完全从零创作

#### 3.8.4 配置流程

```
1. 选择分镜图
   ↓
2. 选择 AI 厂商
   ↓
3. 选择生成模式
   ↓
4. 设置视频参数
   - 分辨率：720p / 1080p / 4K
   - 帧率：24fps / 30fps / 60fps
   - 时长：3-15 秒
   - 纵横比：16:9 / 9:16 / 1:1
   ↓
5. 编写/生成视频提示词
   ↓
6. 预览配置
   ↓
7. 保存配置
```

#### 3.8.5 提示词优化

**AI 提示词生成**
- 输入：分镜描述 + 图片
- 输出：优化的视频生成提示词
- 优化方向：
  - 动作描述更具体
  - 镜头运动更流畅
  - 风格描述更准确
  - 适配不同厂商的提示词风格

---

### 3.9 视频生成模块

#### 3.9.1 功能概述
基于视频配置调用 AI 服务生成视频，支持批量生成、实时监控和结果管理。

#### 3.9.2 核心功能

**视频生成**
- 单个生成：生成单个视频配置
- 批量生成：同时生成多个配置
- 队列管理：任务队列自动调度
- 优先级：VIP 用户优先生成

**状态监控**
- 实时状态：等待中、生成中、已完成、失败
- 进度追踪：百分比进度条
- 预计时间：预估剩余时间
- 失败重试：失败自动重试 3 次

**结果管理**
- 多版本：每个配置可生成多个版本
- 结果预览：视频播放器预览
- 结果对比：并排对比多个版本
- 选择最佳：标记最满意的版本
- 结果下载：单个/批量下载

**视频编辑**
- 简单剪辑：裁剪视频片段
- 添加音效：添加背景音乐和音效
- 添加字幕：自动或手动添加字幕
- 视频拼接：将多个片段拼接成完整视频

#### 3.9.3 生成流程

```
用户提交生成请求
  ↓
后端创建任务记录（保存到 Redis）
  ↓
异步视图返回 task_id
  ↓
后台线程池处理任务
  ├─ 调用 AI 服务商 API
  ├─ 轮询生成状态（每 5 秒）
  └─ 下载视频文件
  ↓
存储到对象存储 (OSS)
  ↓
更新数据库状态
  ↓
WebSocket 推送完成通知
  ↓
前端更新 UI
```

#### 3.9.4 任务调度策略

**队列优先级**
- 高优先级：VIP 用户
- 中优先级：付费用户
- 低优先级：免费用户

**并发控制**
- 单用户并发：最多 3 个任务
- 系统并发：根据服务器资源动态调整
- 限流：防止恶意调用

**失败重试**
- 重试策略：指数退避（1s、2s、4s）
- 最大重试：3 次
- 失败通知：邮件/站内信通知用户

---

### 3.10 任务中心模块

#### 3.10.1 功能概述
统一管理所有异步任务，提供任务查询、监控和管理功能。

#### 3.10.2 核心功能

**任务列表**
- 任务类型：素材生成、视频生成、批量处理
- 任务状态：等待、进行中、成功、失败、取消
- 任务筛选：按类型、状态、项目筛选
- 任务搜索：按任务名称搜索

**任务详情**
- 基本信息：任务名称、类型、创建时间
- 执行信息：开始时间、结束时间、耗时
- 参数信息：生成参数详情
- 结果信息：生成结果预览
- 错误信息：失败原因和堆栈

**任务操作**
- 取消任务：取消等待中或进行中的任务
- 重试任务：重新执行失败的任务
- 删除任务：删除任务记录
- 批量操作：批量取消、重试、删除

**任务统计**
- 今日任务：今日任务数量和成功率
- 任务趋势：任务数量趋势图
- 耗时分析：各类任务平均耗时
- 失败分析：失败原因分布

---

### 3.11 系统设置模块

#### 3.11.1 功能概述
配置系统参数、AI 服务和用户偏好。

#### 3.11.2 核心功能

**个人设置**
- 基本信息：昵称、头像、邮箱
- 账户安全：修改密码
- 通知设置：邮件通知、站内通知
- 偏好设置：语言、主题

**AI 服务配置**

**配置方式**：极简配置，用户只需填写 API 密钥

**API 密钥配置**
- 服务商：PH8 大模型 API（系统预配置，用户无需关心）
- 用户配置：只需填写 API Key
- 功能：
  - 填写 API 密钥
  - 测试连接验证
  - 保存并加密存储
  - 查看密钥状态
- 状态显示：
  - ✅ 未配置：提示用户配置密钥
  - ✅ 验证中：正在测试连接
  - ✅ 有效：显示最后验证时间
  - ✅ 无效：显示错误原因
  - ✅ 已过期：提示充值或更新

**支持的 AI 功能**
- 语言模型：GPT-4、GPT-3.5、Claude 等
- 图像生成：DALL-E 3、MidJourney、Stable Diffusion 等
- 视频生成：RunwayML、Pika Labs、火山引擎等

**获取 API 密钥**
- 访问：https://ph8.co
- 注册账号并获取 API Key
- 复制密钥粘贴到系统设置中
- 点击测试连接验证

**提示词模板管理**
- 预设模板：系统预设模板
- 自定义模板：用户自定义模板
- 模板变量：支持动态变量
- 模板导入导出

**数据管理**
- 存储空间：查看存储使用情况
- 数据清理：清理过期文件
- 数据导出：导出项目数据
- 数据备份：备份到云端

**账单管理**
- 套餐信息：当前套餐和到期时间
- 使用统计：各项资源使用情况
- 充值记录：充值历史记录
- 发票管理：申请和下载发票

---

### 3.12 AI 服务配置系统

#### 3.12.1 设计理念

**简化配置，降低门槛**：系统使用 PH8 大模型 API 中转商（OpenAI 兼容接口），用户只需配置 API 密钥即可使用全球主流 AI 模型。

**核心优势**：
- **极简配置**：用户只需填写一个 API Key
- **全模型支持**：一个 API 支持 GPT-4、Claude、DALL-E 等全球模型
- **统一标准**：完全兼容 OpenAI API 格式
- **成本优化**：中转商价格比官方更优惠

#### 3.12.2 配置方式

**系统级配置（环境变量）**

由管理员在部署时配置，用户无需关心：

```bash
# .env 配置示例

# LLM 服务配置（普华开放 API 平台）
AI_LLM_BASE_URL=https://ph8.co/openai/v1
AI_LLM_DEFAULT_MODEL=gpt-4-turbo
AI_LLM_TEMPERATURE=0.7
AI_LLM_MAX_TOKENS=2048

# 图像生成服务配置
AI_IMAGE_BASE_URL=https://ph8.co/openai/v1
AI_IMAGE_DEFAULT_MODEL=dall-e-3
AI_IMAGE_SIZE=1024x1024
AI_IMAGE_RESPONSE_FORMAT=url
AI_IMAGE_COUNT=4

# 视频生成服务配置
AI_VIDEO_BASE_URL=https://ph8.co/openai/v1
AI_VIDEO_DEFAULT_MODEL=kling-v1
AI_VIDEO_SECONDS=8
```

**用户级配置（界面配置）**

用户只需在设置页面配置 API 密钥：

**用户模型扩展 (User)**
```
新增字段：
- ai_api_key: API 密钥（加密存储）
- ai_api_key_status: 密钥状态（未配置/有效/无效/已过期）
- ai_api_key_verified_at: 最后验证时间
```

#### 3.12.3 API 调用实现

**统一服务层**：

```python
# app/services/ai/base.py
from openai import AsyncOpenAI
from app.core.config import settings
from app.models.user import User

class BaseAIService:
    @staticmethod
    async def get_client(user: User, service_type: str = "llm") -> AsyncOpenAI:
        """获取 AI 客户端"""
        if not user.ai_api_key:
            raise ValueError("用户未配置 API 密钥")
        
        # 根据服务类型选择 Base URL
        base_url = {
            "llm": settings.AI_LLM_BASE_URL,
            "image": settings.AI_IMAGE_BASE_URL,
        }.get(service_type, settings.AI_LLM_BASE_URL)
        
        return AsyncOpenAI(
            base_url=base_url,
            api_key=user.ai_api_key,  # 用户的 API Key
            timeout=60.0
        )

# app/services/ai/llm.py
class LLMService(BaseAIService):
    async def generate_text(
        self,
        user: User,
        messages: List[dict],
        model: str = None,
        **kwargs
    ) -> str:
        """调用语言模型生成文本"""
        client = await self.get_client(user, "llm")
        
        # 使用环境变量配置的默认模型
        model = model or settings.AI_LLM_DEFAULT_MODEL
        
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=kwargs.get("temperature", settings.AI_LLM_TEMPERATURE),
            max_tokens=kwargs.get("max_tokens", settings.AI_LLM_MAX_TOKENS),
        )
        
        return response.choices[0].message.content

# app/services/ai/image.py  
class ImageService(BaseAIService):
    async def generate_image(
        self,
        user: User,
        prompt: str,
        n: int = None,
        **kwargs
    ) -> List[str]:
        """调用图像模型生成图片"""
        client = await self.get_client(user, "image")
        
        response = await client.images.generate(
            model=settings.AI_IMAGE_DEFAULT_MODEL,
            prompt=prompt,
            n=n or settings.AI_IMAGE_COUNT,
            size=kwargs.get("size", settings.AI_IMAGE_SIZE),
            quality=kwargs.get("quality", settings.AI_IMAGE_QUALITY),
        )
        
        return [img.url for img in response.data]
```

#### 3.12.4 用户界面设计

**API 密钥配置页面**

位置：系统设置 → AI 配置

**界面内容**：

```
┌─────────────────────────────────────────┐
│  AI 服务配置                             │
├─────────────────────────────────────────┤
│                                          │
│  API 密钥 *                              │
│  ┌─────────────────────────────────┐    │
│  │ ph8-xxxxxxxxxxxxxxxxxxxxx       │    │
│  └─────────────────────────────────┘    │
│                                          │
│  [测试连接]  [保存]                      │
│                                          │
│  状态: ● 已验证 (2026-02-01 10:30)      │
│                                          │
├─────────────────────────────────────────┤
│  服务说明                                │
├─────────────────────────────────────────┤
│  • 服务商: 普华开放 API 平台             │
│  • 支持模型: GPT-4, Claude, DALL-E 等    │
│  • 获取密钥: https://ph8.co              │
│                                          │
│  已支持的模型:                           │
│  - LLM: gpt-4, gpt-4-turbo, claude-3    │
│  - 图像: dall-e-3, midjourney, sd-xl    │
│  - 视频: kling-v1, gen-2, pika-1.0     │
│                                          │
└─────────────────────────────────────────┘
```

**功能按钮**：

1. **测试连接**
   - 发送测试请求到 PH8 API
   - 验证 API Key 有效性
   - 显示连接状态和延迟
   - 更新验证时间戳

2. **保存**
   - 加密存储 API Key
   - 更新密钥状态
   - 显示保存成功提示

**提示信息**：

```
首次使用提示:
"您需要配置普华开放 API 平台密钥才能使用 AI 功能。
访问 https://ph8.co 注册并获取 API 密钥。"

密钥无效提示:
"API 密钥验证失败，请检查密钥是否正确。
错误信息: Invalid API Key"

密钥过期提示:
"您的 API 密钥余额不足或已过期，请充值后继续使用。"
```

#### 3.12.5 密钥验证流程

**验证接口**：

```python
# app/api/v1/endpoints/user.py
@router.post("/settings/ai/verify")
async def verify_ai_api_key(
    api_key: str,
    current_user: User = Depends(get_current_user)
):
    """验证 AI API 密钥"""
    try:
        # 临时创建客户端测试
        client = AsyncOpenAI(
            base_url=settings.AI_LLM_BASE_URL,
            api_key=api_key,
            timeout=10.0
        )
        
        # 发送测试请求
        response = await client.chat.completions.create(
            model=settings.AI_LLM_DEFAULT_MODEL,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=5
        )
        
        return {
            "valid": True,
            "message": "API 密钥验证成功",
            "model": response.model,
            "latency": "50ms"  # 计算实际延迟
        }
    
    except Exception as e:
        return {
            "valid": False,
            "message": f"API 密钥验证失败: {str(e)}"
        }

@router.post("/settings/ai/save")
async def save_ai_api_key(
    api_key: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """保存 AI API 密钥"""
    # 先验证
    verify_result = await verify_ai_api_key(api_key, current_user)
    
    if not verify_result["valid"]:
        raise HTTPException(400, detail=verify_result["message"])
    
    # 加密存储
    current_user.ai_api_key = encrypt_api_key(api_key)
    current_user.ai_api_key_status = "valid"
    current_user.ai_api_key_verified_at = datetime.utcnow()
    
    await db.commit()
    
    return {"message": "API 密钥保存成功"}
```

#### 3.12.6 PH8 API 说明

**服务地址**: https://ph8.co/openai/v1

**支持模型**: GPT-4、GPT-3.5、Claude、DALL-E 3、MidJourney、Kling 等

**接口文档**: [ph8.md](./ph8.md)

---

## 四、用户操作流程

### 4.1 完整创作流程图

```
┌─────────────────────┐
│  注册/登录           │
│  - 账号密码注册      │
│  - 账号密码登录      │
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│  创建项目            │
│  - 填写项目信息      │
│  - 选择风格类型      │
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│  导入原始文本        │
│  - 上传 Word/TXT    │
│  - 在线编辑          │
│  - 自动解析章节      │
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│  创建大纲            │
│  - 手动创建          │
│  - AI 生成大纲       │
│  - 编辑事件线        │
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│  生成素材            │
│  - 添加角色          │
│  - AI 生成角色图     │
│  - 添加场景/道具     │
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│  编写剧本            │
│  - 手动编写          │
│  - AI 生成剧本       │
│  - 优化对话          │
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│  生成分镜            │
│  - WebSocket 连接   │
│  - AI Agent 协作    │
│  - 实时查看进度      │
│  - 编辑分镜          │
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│  配置视频            │
│  - 选择分镜图        │
│  - 选择 AI 厂商      │
│  - 设置视频参数      │
│  - 生成提示词        │
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│  生成视频            │
│  - 提交生成任务      │
│  - 实时监控进度      │
│  - 查看生成结果      │
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│  编辑视频            │
│  - 选择最佳版本      │
│  - 简单剪辑          │
│  - 添加字幕音效      │
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│  导出视频            │
│  - 单个下载          │
│  - 批量下载          │
│  - 发布到平台        │
└─────────────────────┘
```

### 4.2 典型使用场景

#### 场景一：个人创作者

**目标**：5000 字小说改编为 3 分钟短剧

**耗时**：约 2.5 小时（AI 生成 1.5h，人工调整 1h）

**效率**：传统方式 2 周 → 现在 2.5 小时（50 倍提升）

#### 场景二：团队协作

**目标**：批量产出 10 集系列短剧

**角色**：项目经理、剧本编写、素材管理

**耗时**：3 周

**效率**：传统方式 3-6 个月 → 现在 3 周（4-8 倍提升）

---

## 五、数据架构设计

### 5.1 核心数据模型

#### 用户模型 (User)
```
字段：
- id: UUID
- username: 用户名（唯一）
- email: 邮箱（唯一）
- password: 密码（加密存储）
- avatar: 头像 URL
- role: 角色（admin/creator/viewer）
- plan: 订阅套餐（free/pro/enterprise）
- ai_api_key: AI API 密钥（加密存储，用于调用普华 API）
- ai_api_key_status: 密钥状态（未配置/有效/无效/已过期）
- ai_api_key_verified_at: 最后验证时间
- is_active: 账户是否激活
- last_login: 最后登录时间
- created_at: 创建时间
- updated_at: 更新时间
```

#### 项目模型 (Project)
```
字段：
- id: UUID
- name: 项目名称
- description: 项目描述
- cover: 封面图片 URL
- type: 项目类型（drama/ad/tutorial）
- style: 风格（realistic/anime/scifi）
- owner_id: 创建者 ID（外键）
- status: 状态（active/archived/deleted）
- created_at: 创建时间
- updated_at: 更新时间

关系：
- owner: User（多对一）
- members: User（多对多，通过 ProjectMember）
- novels: Novel（一对多）
- outlines: Outline（一对多）
- assets: Asset（一对多）
- scripts: Script（一对多）
```

#### 原文模型 (Novel)
```
字段：
- id: UUID
- project_id: 项目 ID（外键）
- title: 标题
- content: 全文内容（TextField）
- word_count: 字数
- chapters: JSON（章节列表）
- created_at: 创建时间
- updated_at: 更新时间

关系：
- project: Project（多对一）
```

#### 大纲模型 (Outline)
```
字段：
- id: UUID
- project_id: 项目 ID（外键）
- title: 大纲标题
- content: 大纲内容（JSON）
- structure: 结构类型（three_act/hero_journey）
- created_at: 创建时间
- updated_at: 更新时间

关系：
- project: Project（多对一）
- scripts: Script（一对多）
```

#### 素材模型 (Asset)
```
字段：
- id: UUID
- project_id: 项目 ID（外键）
- name: 素材名称
- type: 素材类型（character/scene/prop）
- description: 描述
- file_path: 文件路径
- file_url: 文件 URL
- prompt: 生成提示词
- metadata: JSON（分辨率、风格等）
- tags: JSON（标签数组）
- created_at: 创建时间
- updated_at: 更新时间

关系：
- project: Project（多对一）
```

#### 剧本模型 (Script)
```
字段：
- id: UUID
- project_id: 项目 ID（外键）
- outline_id: 大纲 ID（外键）
- title: 剧本标题
- episode: 集数
- content: 剧本内容（JSON）
- word_count: 字数
- duration_estimate: 预估时长（秒）
- status: 状态（draft/completed）
- created_at: 创建时间
- updated_at: 更新时间

关系：
- project: Project（多对一）
- outline: Outline（多对一）
- storyboards: Storyboard（一对多）
```

#### 分镜模型 (Storyboard)
```
字段：
- id: UUID
- script_id: 剧本 ID（外键）
- sequence: 序号
- scene_description: 场景描述
- shot_type: 镜头类型（closeup/medium/wide）
- camera_movement: 运镜方式
- image_path: 分镜图片路径
- image_url: 分镜图片 URL
- prompt: 生成提示词
- duration: 时长（秒）
- created_at: 创建时间
- updated_at: 更新时间

关系：
- script: Script（多对一）
- video_configs: VideoConfig（一对多）
```

#### 视频配置模型 (VideoConfig)
```
字段：
- id: UUID
- storyboard_id: 分镜 ID（外键）
- provider: AI 厂商（runwayml/pika/volcengine）
- mode: 生成模式（single/startEnd/multi）
- images: JSON（图片路径数组）
- resolution: 分辨率
- fps: 帧率
- duration: 时长（秒）
- aspect_ratio: 纵横比
- prompt: 视频提示词
- parameters: JSON（其他参数）
- selected_result_id: 选中的结果 ID
- created_at: 创建时间
- updated_at: 更新时间

关系：
- storyboard: Storyboard（多对一）
- results: VideoResult（一对多）
```

#### 视频结果模型 (VideoResult)
```
字段：
- id: UUID
- config_id: 配置 ID（外键）
- task_id: 任务 ID
- status: 状态（pending/processing/completed/failed）
- video_path: 视频文件路径
- video_url: 视频 URL
- thumbnail_url: 缩略图 URL
- duration: 实际时长（秒）
- file_size: 文件大小（字节）
- error_message: 错误信息
- created_at: 创建时间
- updated_at: 更新时间

关系：
- config: VideoConfig（多对一）
```

#### 任务模型 (Task)
```
字段：
- id: UUID
- user_id: 用户 ID（外键）
- project_id: 项目 ID（外键）
- type: 任务类型（asset_gen/video_gen/batch）
- status: 状态（pending/processing/completed/failed/cancelled）
- priority: 优先级（1-10）
- progress: 进度（0-100）
- params: JSON（任务参数）
- result: JSON（任务结果）
- error: 错误信息
- started_at: 开始时间
- completed_at: 完成时间
- created_at: 创建时间
- updated_at: 更新时间

关系：
- user: User（多对一）
- project: Project（多对一）
```

### 5.2 数据关系图

```
User ────┬─── Project ────┬─── Novel
         │                ├─── Outline ─── Script ─── Storyboard ─── VideoConfig ─── VideoResult
         │                └─── Asset
         │
         └─── Task
```

### 5.3 数据存储策略

#### 结构化数据（PostgreSQL）
- 用户信息
- 项目元数据
- 模型关系
- 任务状态

#### 非结构化数据（MinIO/S3）
- 图片文件
- 视频文件
- 文档文件
- 备份文件

#### 缓存数据（Redis）
- 用户 Session
- API 频率限制
- 任务队列
- 热点数据

---

## 六、产品方案

### 6.1 技术栈选型

#### 前端技术栈

- **框架**: React 18.x + TypeScript
- **构建工具**: Vite
- **状态管理**: Redux Toolkit
- **路由**: React Router v6
- **UI 组件**: Ant Design
- **HTTP 客户端**: Axios
- **实时通信**: Socket.io-client
- **表单处理**: React Hook Form
- **样式方案**: CSS Modules + Sass

#### 后端技术栈

- **框架**: FastAPI 0.109+
- **语言**: Python 3.11+
- **数据库**: PostgreSQL 15+ (SQLAlchemy 2.x)
- **缓存**: Redis 7+
- **数据验证**: Pydantic 2.x
- **任务处理**: asyncio + ThreadPoolExecutor
- **定时任务**: APScheduler
- **认证**: JWT (python-jose)
- **WebSocket**: FastAPI 原生 WebSocket

#### AI 服务

- **服务商**: 普华开放 API 平台 (ph8.co)
- **接口标准**: OpenAI 兼容接口
- **SDK**: openai Python SDK
- **配置方式**: 环境变量 + 用户 API Key
- **支持模型**: GPT-4、Claude、DALL-E 3、MidJourney、Kling 等

#### 认证方案

- **方式**: 账号密码 (用户名/邮箱 + 密码)
- **加密**: bcrypt/argon2
- **Token**: JWT
- **有效期**: 7 天（可刷新）
- **限制**: 5 次失败锁定 30 分钟

#### 任务队列方案

- **MVP 阶段**: FastAPI BackgroundTasks + ThreadPoolExecutor
- **升级路径**: ARQ → Celery/云服务

### 6.2 架构方案

#### 前后端分离架构

**通信方式**：
- HTTP REST API：常规请求
- WebSocket：实时通信（分镜生成、视频生成）
- Server-Sent Events：服务器推送

#### AI 服务调用架构

```
┌─────────────────────────────────────────────┐
│              业务层                          │
│  （剧本生成、素材生成、视频生成）             │
└────────────────┬────────────────────────────┘
                 ↓
┌─────────────────────────────────────────────┐
│          AI Service 统一服务层               │
│    （LLM / Image / Video Service）          │
└────────────────┬────────────────────────────┘
                 ↓
┌─────────────────────────────────────────────┐
│        普华开放 API 平台中转服务             │
│      （OpenAI 兼容接口）                     │
│   https://ph8.co/openai/v1                  │
└────────────────┬────────────────────────────┘
                 ↓
┌─────────────────────────────────────────────┐
│              全球 AI 模型                    │
│  GPT-4 │ Claude │ DALL-E │ MidJourney       │
└─────────────────────────────────────────────┘
```

**实现代码**：

```python
# app/services/ai/base.py
from openai import AsyncOpenAI
from app.core.config import settings
from app.models.user import User

class BaseAIService:
    @staticmethod
    async def get_client(user: User, service_type: str = "llm") -> AsyncOpenAI:
        """获取 AI 客户端"""
        if not user.ai_api_key:
            raise ValueError("用户未配置 API 密钥")
        
        # 根据服务类型选择 Base URL
        base_url = {
            "llm": settings.AI_LLM_BASE_URL,
            "image": settings.AI_IMAGE_BASE_URL,
        }.get(service_type, settings.AI_LLM_BASE_URL)
        
        return AsyncOpenAI(
            base_url=base_url,
            api_key=user.ai_api_key,  # 使用用户的 API Key
            timeout=60.0
        )

# app/services/ai/llm.py
class LLMService(BaseAIService):
    async def generate_text(
        self,
        user: User,
        messages: List[dict],
        **kwargs
    ) -> str:
        """调用语言模型生成文本"""
        client = await self.get_client(user, "llm")
        
        response = await client.chat.completions.create(
            model=settings.AI_LLM_DEFAULT_MODEL,
            messages=messages,
            temperature=settings.AI_LLM_TEMPERATURE,
            **kwargs
        )
        
        return response.choices[0].message.content

# app/services/ai/image.py  
class ImageService(BaseAIService):
    async def generate_image(
        self,
        user: User,
        prompt: str,
        **kwargs
    ) -> List[str]:
        """调用图像模型生成图片"""
        client = await self.get_client(user, "image")
        
        response = await client.images.generate(
            model=settings.AI_IMAGE_DEFAULT_MODEL,
            prompt=prompt,
            n=settings.AI_IMAGE_COUNT,
            **kwargs
        )
        
        return [img.url for img in response.data]
```

**环境变量配置**：

```bash
# .env
AI_LLM_BASE_URL=https://ph8.co/openai/v1
AI_LLM_DEFAULT_MODEL=gpt-4-turbo
AI_IMAGE_BASE_URL=https://ph8.co/openai/v1
AI_IMAGE_DEFAULT_MODEL=dall-e-3
AI_VIDEO_BASE_URL=https://ph8.co/openai/v1
AI_VIDEO_DEFAULT_MODEL=kling-v1
```

**接口文档**：[ph8.md](./ph8.md)

### 6.3 性能优化

**前端优化**：
- 代码分割（React.lazy）、组件懒加载
- 图片懒加载、WebP 格式、CDN 加速
- React.memo、虚拟滚动、防抖节流
- Web Worker 处理复杂计算

**后端优化**：
- 数据库索引、查询优化、连接池、读写分离
- Redis 缓存热点数据、API 响应缓存
- FastAPI 原生异步、BackgroundTasks、线程池

### 6.4 安全方案

**认证授权**: JWT Token、RBAC 权限控制、API 签名

**数据安全**: HTTPS、密码加密存储、ORM 防注入、XSS/CSRF 防护

**业务安全**: 频率限制、IP 黑名单、操作日志、数据备份

### 6.5 监控与运维

**监控**: Sentry、Prometheus、Grafana、ELK

**部署**: Docker、CI/CD (GitHub Actions)、Nginx、Let's Encrypt

---

## 七、商业价值

### 7.1 市场分析

#### 市场规模

**短视频市场**
- 全球短视频用户：30 亿+
- 中国短视频用户：9 亿+
- 市场规模：3000 亿+ 人民币

**短剧市场**
- 中国微短剧市场：500 亿+ 人民币
- 年增长率：100%+
- 日均播放量：10 亿+ 次

#### 目标市场

**一级市场**：个人创作者和小型工作室
- 市场规模：100 万+
- 付费意愿：中等
- 客单价：199-999 元/月

**二级市场**：MCN 机构和影视公司
- 市场规模：5 万+
- 付费意愿：高
- 客单价：5000-50000 元/月

**三级市场**：品牌营销团队
- 市场规模：10 万+
- 付费意愿：高
- 客单价：10000-100000 元/年

### 7.2 竞争分析

#### 竞争对手

| 产品 | 定位 | 优势 | 劣势 |
|------|------|------|------|
| Toonflow | 全流程 AI 短剧创作 | 开源、可自部署 | 需要技术能力 |
| 剪映 | 视频编辑工具 | 用户量大、易用 | 不支持 AI 生成 |
| Runway | AI 视频生成 | 技术领先 | 价格高、学习门槛高 |
| Pika | AI 视频生成 | 效果好、速度快 | 仅视频生成，无全流程 |

#### 竞争策略

**差异化**：全流程 AI 短剧创作、极简配置、成本降低 70%

**技术特色**：多 Agent 协作、WebSocket 实时反馈、统一 AI 接口

### 7.3 盈利模式

#### 订阅制（主要）

**免费版**
- 1 个项目
- 每月 10 次 AI 生成
- 720p 视频
- 社区支持

**专业版** - ¥199/月
- 10 个项目
- 每月 100 次 AI 生成
- 1080p 视频
- 邮件支持

**企业版** - ¥999/月
- 无限项目
- 无限 AI 生成
- 4K 视频
- 团队协作
- 专属客服

#### 按量付费（补充）
- AI 生成次数包：¥99/100次
- 视频生成时长包：¥199/60分钟
- 存储空间包：¥99/100GB

#### 增值服务
- 定制化开发
- 企业培训
- API 接入
- 私有化部署

### 7.4 价值主张

**用户价值**：降本 90%、提效 10 倍、AI 提质

**平台价值**：降低门槛、数据积累、生态构建

---

## 八、产品路线图

### 8.1 MVP（3 个月）

用户认证、项目管理、原文管理、素材生成、剧本生成、视频生成、任务监控

**目标**：100 种子用户

### 8.2 V1.0（6 个月）

大纲管理、分镜生成（AI Agent）、多厂商视频、视频编辑、团队协作、数据统计

**目标**：5000 注册用户、500 付费用户

### 8.3 V2.0（12 个月）

高级视频编辑、语音生成、字幕生成、音效库、模板市场、作品社区、移动端 H5

**目标**：20,000 注册用户、3,000 付费用户

### 8.4 V3.0（18 个月）

实时渲染、3D 场景、角色动画、多语言、API 平台、插件市场、桌面客户端

**目标**：100,000 注册用户、15,000 付费用户、实现盈利

---

## 附录

### A. 快速开始指南

#### A.1 管理员部署流程

**1. 环境准备**
```bash
# 安装依赖
- PostgreSQL 15+
- Redis 7+
- MinIO/S3（对象存储）
- Python 3.11+
- Node.js 18+
```

**2. 配置环境变量**
```bash
# 复制环境变量模板
cp .env.example .env

# 编辑配置（必须修改的项）
nano .env

必须配置：
- SECRET_KEY（生成随机字符串）
- JWT_SECRET_KEY（生成随机字符串）
- DATABASE_URL（数据库连接）
- STORAGE_*（对象存储配置）

无需修改（普华 API 配置）：
- AI_LLM_BASE_URL=https://ph8.co/openai/v1
- AI_IMAGE_BASE_URL=https://ph8.co/openai/v1
- AI_VIDEO_BASE_URL=https://ph8.co/openai/v1
```

**3. 启动服务**
```bash
# 后端
cd backend
pip install -r requirements.txt
alembic upgrade head
uvicorn app.main:app --host 0.0.0.0 --port 8000

# 前端
cd frontend
npm install
npm run dev
```

**4. 验证部署**
- 访问：http://localhost:3000
- 注册管理员账号
- 进入设置 → AI 配置
- 配置 PH8 API Key

#### A.2 用户使用流程

**1. 注册**: 用户名/邮箱 + 密码

**2. 配置 API Key**: 访问 https://ph8.co → 获取密钥 → 系统设置填写 → 测试连接

**3. 创作流程**: 新建项目 → 导入原文 → AI 生成大纲(30s) → 生成素材(10-15min) → AI 生成剧本(2-5min) → AI 生成分镜(5-10min) → 配置视频 → 批量生成视频(1-3h) → 编辑导出

**耗时**: 约 2-3 小时

**4. 成本**: 单个项目（3分钟短剧）约 ¥70-120

#### A.3 常见问题

**Q1: API Key 验证失败？**
检查密钥格式、PH8 账号余额、密钥有效期、网络连接

**Q2: AI 生成速度慢？**
检查网络 (ping ph8.co)、更换 GPT-3.5、简化提示词

**Q3: 更换 AI 模型？**
修改环境变量 `AI_LLM_DEFAULT_MODEL`，重启服务

**Q4: 使用 OpenAI Key？**
修改 `AI_LLM_BASE_URL=https://api.openai.com/v1`（不建议，价格贵 70%）

### B. 术语表

| 术语 | 解释 |
|------|------|
| AI Agent | 具有自主决策能力的 AI 智能体 |
| 分镜 | 将剧本转化为视觉画面的设计图 |
| 提示词 | 用于引导 AI 生成内容的文本描述 |
| Token | AI 模型处理文本的基本单位，约等于一个汉字或 0.75 个英文单词 |
| LoRA | 低秩适应，一种 AI 模型微调技术 |
| WebSocket | 双向实时通信协议 |
| APScheduler | Python 定时任务调度库 |
| ThreadPoolExecutor | Python 线程池，用于并发处理任务 |
| OpenAI 兼容接口 | 遵循 OpenAI API 规范的接口标准，普华 API 完全兼容此标准 |
| 普华开放 API 平台 | PH8 提供的多模态 AI 服务，支持文本、图像、视频生成，完全兼容 OpenAI SDK |
| API Key | 用于访问普华 API 的密钥，用户在 https://ph8.co 注册获取 |

### B. 参考资料

**技术文档**
- React 官方文档：https://react.dev
- FastAPI 官方文档：https://fastapi.tiangolo.com
- Redux Toolkit 文档：https://redux-toolkit.js.org
- SQLAlchemy 文档：https://docs.sqlalchemy.org
- Pydantic 文档：https://docs.pydantic.dev
- OpenAI API 文档：https://platform.openai.com/docs/api-reference

**AI 服务**
- 普华开放 API 平台：https://ph8.co
- PH8 接口文档：[ph8.md](./ph8.md)
- OpenAI 官方文档：https://platform.openai.com/docs

**行业报告**
- 中国短视频行业报告 2024
- AI 创作工具市场分析报告

### C. 环境变量配置示例

#### C.1 完整配置文件

创建 `.env` 文件（后端项目根目录）：

```bash
# ===========================================
# 应用基础配置
# ===========================================
APP_NAME=StoryWeave
APP_ENV=production
DEBUG=false
SECRET_KEY=your-secret-key-here

# ===========================================
# 数据库配置
# ===========================================
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/storyweave
REDIS_URL=redis://localhost:6379/0

# ===========================================
# AI 服务配置（普华 PH8 开放 API 平台）
# ===========================================

# 语言模型配置
AI_LLM_BASE_URL=https://ph8.co/openai/v1
AI_LLM_DEFAULT_MODEL=gpt-4-turbo
AI_LLM_FALLBACK_MODEL=gpt-3.5-turbo
AI_LLM_TEMPERATURE=0.7
AI_LLM_MAX_TOKENS=2048
AI_LLM_TOP_P=1.0

# 图像生成配置
AI_IMAGE_BASE_URL=https://ph8.co/openai/v1
AI_IMAGE_DEFAULT_MODEL=dall-e-3
AI_IMAGE_SIZE=1024x1024
AI_IMAGE_QUALITY=hd
AI_IMAGE_COUNT=4
AI_IMAGE_RESPONSE_FORMAT=url

# 视频生成配置（PH8 统一接口）
AI_VIDEO_BASE_URL=https://ph8.co/openai/v1
AI_VIDEO_DEFAULT_MODEL=kling-v1
AI_VIDEO_SIZE=1280x720
AI_VIDEO_SECONDS=8
AI_VIDEO_RESOLUTION=1080p
AI_VIDEO_RATIO=16:9

# API 调用配置
AI_REQUEST_TIMEOUT=60
AI_MAX_RETRIES=3
AI_RETRY_DELAY=2

# ===========================================
# 对象存储配置（MinIO/S3）
# ===========================================
STORAGE_TYPE=minio
STORAGE_ENDPOINT=http://localhost:9000
STORAGE_ACCESS_KEY=minioadmin
STORAGE_SECRET_KEY=minioadmin
STORAGE_BUCKET=storyweave
STORAGE_PUBLIC_URL=http://localhost:9000/storyweave

# ===========================================
# 任务队列配置
# ===========================================
TASK_QUEUE_REDIS_URL=redis://localhost:6379/1
TASK_MAX_WORKERS=10
TASK_POLL_INTERVAL=1

# ===========================================
# 安全配置
# ===========================================
CORS_ORIGINS=["http://localhost:3000"]
JWT_SECRET_KEY=your-jwt-secret-key
JWT_ALGORITHM=HS256
JWT_EXPIRATION_MINUTES=1440

# ===========================================
# 日志配置
# ===========================================
LOG_LEVEL=INFO
LOG_FILE=logs/app.log
```

#### C.2 开发环境配置

创建 `.env.dev` 文件：

```bash
APP_ENV=development
DEBUG=true

# 开发环境数据库
DATABASE_URL=postgresql+asyncpg://dev:dev@localhost:5432/storyweave_dev
REDIS_URL=redis://localhost:6379/0

# PH8 API（开发环境也使用生产配置）
AI_LLM_BASE_URL=https://ph8.co/openai/v1
AI_LLM_DEFAULT_MODEL=gpt-3.5-turbo  # 开发用便宜的模型
AI_IMAGE_BASE_URL=https://ph8.co/openai/v1
AI_IMAGE_DEFAULT_MODEL=dall-e-3

# 本地存储
STORAGE_TYPE=local
STORAGE_PATH=./storage

# CORS 允许所有源（开发环境）
CORS_ORIGINS=["*"]

# 详细日志
LOG_LEVEL=DEBUG
```

#### C.3 配置说明

**必须修改的配置**：
- `SECRET_KEY` - 应用密钥，用于加密
- `JWT_SECRET_KEY` - JWT 签名密钥
- `DATABASE_URL` - 数据库连接地址
- `STORAGE_*` - 对象存储配置

**无需修改的配置**：
- `AI_LLM_BASE_URL` - PH8 API 地址（固定）
- `AI_IMAGE_BASE_URL` - PH8 API 地址（固定）
- `AI_*_DEFAULT_MODEL` - 默认模型（可根据需要调整）

**用户配置**：
- 用户的 API Key 存储在数据库 `User.ai_api_key` 字段
- 系统不存储全局 API Key，每个用户使用自己的密钥

### D. PH8 API 接口文档

#### D.1 完整接口文档

**普华开放 API 平台** 提供完全兼容 OpenAI SDK 的多模态 AI 服务，详细接口文档请参考：

📄 **接口文档**：[ph8.md](./ph8.md)

#### D.2 核心信息速查

**服务地址**：
- 主地址：`https://ph8.co/openai/v1`
- 备用地址：`https://ph8.co/v1`

**支持的服务**：
- ✅ 文本生成（LLM）- `/v1/chat/completions`
- ✅ 图像生成（文生图、图生图）- `/v1/images/generations`
- ✅ 视频生成（文生视频、图生视频）- `/v1/videos`
- ✅ 文本嵌入（Embedding）- `/v1/embeddings`
- ✅ OCR 识别 - `/v1/chat/completions`

**快速开始**：

```python
from openai import OpenAI

# 使用普华 API
client = OpenAI(
    base_url="https://ph8.co/openai/v1",
    api_key="your-api-key"
)

# 文本生成
response = client.chat.completions.create(
    model="gpt-4-turbo",
    messages=[{"role": "user", "content": "你好"}]
)

# 图像生成
image = client.images.generate(
    model="dall-e-3",
    prompt="一只可爱的猫咪",
    size="1024x1024"
)

# 视频生成
video = client.videos.create(
    model="kling-v1",
    prompt="一辆跑车在公路上疾驰",
    seconds="8"
)
```

#### D.3 系统集成配置

**环境变量配置**（`.env` 文件）：

```bash
# PH8 API 配置
AI_LLM_BASE_URL=https://ph8.co/openai/v1
AI_IMAGE_BASE_URL=https://ph8.co/openai/v1
AI_VIDEO_BASE_URL=https://ph8.co/openai/v1

# 默认模型
AI_LLM_DEFAULT_MODEL=gpt-4-turbo
AI_IMAGE_DEFAULT_MODEL=dall-e-3
AI_VIDEO_DEFAULT_MODEL=kling-v1
```

**用户配置**：
- 用户只需在系统设置中配置自己的 API Key
- API Key 加密存储在 `User.ai_api_key` 字段
- 系统自动使用用户的密钥调用 PH8 API

#### D.4 获取 API Key

1. 访问官网：https://ph8.co
2. 注册并登录账号
3. 进入控制台获取 API 密钥
4. 在 StoryWeave 系统设置中配置密钥

#### D.5 更多信息

完整的接口参数说明、错误处理、最佳实践等详细信息，请参考项目根目录的 [ph8.md](./ph8.md) 文档。

该文档包含：
- ✅ 所有接口的详细参数说明
- ✅ 完整的代码示例（Python）
- ✅ 错误处理和重试机制
- ✅ 流式输出示例
- ✅ 最佳实践建议
- ✅ 常见问题解答


---

**版权声明**

本文档为 StoryWeave 产品设计文档，版权所有。未经授权，不得擅自复制、传播或用于商业用途。
