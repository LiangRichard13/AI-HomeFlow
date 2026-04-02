"""AI-HomeFlow Agent 运行配置。"""

from __future__ import annotations

import os
import dotenv

dotenv.load_dotenv()

DEFAULT_MAX_TOOL_ROUNDS = 8

# Chat / Agent 配置
DEFAULT_CHAT_API_KEY = (os.getenv("CHAT_API_KEY") or "").strip()
DEFAULT_CHAT_BASE_URL = "https://api.siliconflow.cn/v1"
DEFAULT_CHAT_MODEL = "Pro/MiniMaxAI/MiniMax-M2.5"

# 图生图配置
DEFAULT_ARK_IMAGE_API_KEY = (os.getenv("ARK_API_KEY") or "").strip()
DEFAULT_ARK_IMAGE_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
DEFAULT_ARK_IMAGE_MODEL = "doubao-seedream-5-0-260128"
SUPPORTED_ARK_IMAGE_FORMATS = {
    ".jpg": "jpeg",
    ".jpeg": "jpeg",
    ".png": "png",
    ".webp": "webp",
    ".bmp": "bmp",
    ".tif": "tiff",
    ".tiff": "tiff",
    ".gif": "gif",
}
DEFAULT_ARK_IMAGE_PROMPT = """请你根据房间照片，将提供的家具添加进去：
渲染技术词
● 照片级渲染
● 高细节
● 8k 分辨率
视角与风格
保持房间照片视角
灯光与氛围
● 柔和的暖光
● 自然日光从窗口射入
● 温馨、舒适、整洁
装修风格
● 如果提供的房间照片中有装修细节则保持不变
● 如果房间中尚未装修则提供装修细节，其风格和家具相协调
添加其他家具
● 如果提供的房间照片中有其他家具则保持不变
● 如果房间照片中尚未提供任何家具则加入其他家具和提供的家具相协调，让房间看起来完整（比如提供了客厅照片和沙发照片，如果缺少电视、灯具等需要进行补充）
● 如果房间中有重复家具，则使用提供的家具进行替换（比如提供了客厅照片和沙发照片，且客厅中已有沙发，则使用沙发照片中的沙发替换该客厅中的沙发）"""