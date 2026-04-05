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
# DEFAULT_CHAT_BASE_URL = "https://api.deepseek.com"
# DEFAULT_CHAT_MODEL = "deepseek-chat"

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
是否添加其他家具
● 如果提供的房间照片中有其他家具则保持不变
● 如果房间照片中尚未提供任何家具则加入其他家具和提供的家具相协调，让房间看起来完整（比如提供了客厅照片和沙发照片，如果缺少电视、灯具等需要进行补充）
● 如果房间中有重复家具，则使用提供的家具进行替换（比如提供了客厅照片和沙发照片，且客厅中已有沙发，则使用沙发照片中的沙发替换该客厅中的沙发）
如果没有提供房间照片，则根据提供的家具图片生成房间照片、
● 该房间只在没有提供房间照片的情况下生成，在提供房间照片的情况下不生成
● 该房间的装修应该和提供的家具图片相协调
"""

# 房间风格理解
DEFAULT_ARK_VISION_MODEL = "doubao-seed-2-0-mini-260215"
DEFAULT_ARK_VISION_MAX_TOKENS = 400
DEFAULT_ARK_ROOM_UNDERSTANDING_PROMPT = """你是室内空间分析助手。请基于用户上传的房间照片，输出适合家具推荐系统使用的中文分析。

要求：
1. 先判断这是不是一个室内房间；若不是，也要如实说明。
2. 简要描述整体风格倾向，例如现代、北欧、原木、奶油、工业、轻奢、中古、极简等；若混合风格也说明。
3. 提取会影响家具搭配的关键信息：主色调、材质感觉、采光/氛围、空间大小感、已有装修元素或明显限制。
4. 不要编造尺寸；没有把握就说“无法判断”。
5. 输出 4-6 条项目符号，最后补一行“推荐提示：...”，总结适合优先考虑的家具方向。
6. 语气客观简洁，便于被主 Agent 直接引用为上下文。
"""
