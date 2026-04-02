# 🏠 AI-HomeFlow: 智能家居协同设计系统

**AI-HomeFlow** 是一款基于多智能体（Multi-Agent）技术的智能家居选购与视觉预览系统。它通过独创的“双列表（Show-List & User-List）动态交互范式”，解决了 AI 在长链路装修决策中容易产生的“上下文幻觉”和“需求变更难”等痛点，为用户提供从预算控制到意向图生成的全流程体验。

---

## 🚀 核心架构：双列表动态交互逻辑

本项目的核心竞争力在于其对 LLM 上下文的精细化管理，确保 Agent 在多轮对话中依然保持极高的逻辑准确度。

### 1. 动态内存管理机制

- **Show-List (候选池)**：由 Agent 根据当前搜索指令生成的临时推荐列表。每轮交互后自动清空，防止陈旧信息干扰。
- **User-List (决策池)**：存放用户最终确认的家具。作为持久化上下文，随对话流传递，确保后续推荐的风格与预算的一致性。

### 2. 多级过滤漏斗

1. **硬性规则过滤**：基于价格区间、尺寸边界、家具类型进行 SQL/API 级别检索。
2. **语义重排 (RAG)**：利用 OpenAI/HuggingFace Embedding 对家具描述进行向量匹配，精准捕捉用户如“温馨”、“工业感”等模糊偏好。

---

## 📊 数据 Schema (模拟验证用)

系统采用以下标准字段进行逻辑验证与向量检索：


| **字段**        | **类型** | **说明**                              |
| ------------- | ------ | ----------------------------------- |
| `id`          | String | 家具唯一 ID                             |
| `category`    | Enum   | 家具类型 (sofa, table, bed, etc.)       |
| `price`       | Float  | 单价                                  |
| `dimensions`  | Object | 物理尺寸 `{"w": 200, "d": 90, "h": 85}` |
| `description` | Text   | 详细描述（用于向量 Embedding 检索的关键字段）        |
| `style_tags`  | List   | 风格标签 (Nordic, Minimalist, etc.)     |
| `image_url`   | String | 家具静态图片资源地址                          |


---

## 📝 运行流程 (Workflow)

1. **Init**: 用户上传房型图（可选），输入风格偏好等；决策池金额以已选家具单价之和为准。
2. **Loop**:
  - 用户和 Agent 描述需求，Agent 进行查找推荐。
  - Agent 填充 `Show-List`。
  - 用户勾选家具进入 `User-List`。
  - 界面展示决策池总价 `sum(User-List.price)`，清空 `Show-List`，进入下一轮（如：选完沙发选茶几）。
3. **Finish**: 用户确认 `User-List` 完整清单。
4. **Visualize**: 触发视觉 Agent，生成基于房型结构的样板间意向图。

---

## 🖥️ Streamlit 原型（Agent + 双列表）

依赖安装（含 `streamlit`、`langchain-openai`）：

```bash
cd backend
pip install -r requirements.txt
```

环境变量（可选，亦可在应用侧栏填写 API Key）：

- `OPENAI_API_KEY`：OpenAI 或兼容服务的密钥。
- `OPENAI_BASE_URL`：可选；侧栏「Base URL」会覆盖并传入客户端。

在**仓库根目录**启动：

```bash
streamlit run frontend/app.py
```

说明：原型通过 `backend/agent/runner.py` 加载 `backend/agent/tools/SKILL.md` 作为系统提示，并直接调用家具检索工具；首次对话会触发 GTE/Chroma 预热（与 FastAPI 同源逻辑）。**不必同时启动** `backend/main.py` 的 Uvicorn 即可试用对话与双列表。

---

## ⚠️ 免责声明

> 本系统生成的图片仅作为**布局意向参考**，不代表 1:1 的物理还原。家具尺寸、光影及摆放比例以最终采购清单及实际测量为准。

