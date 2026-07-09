"""DocQA Agent —— 基于 LLM 的项目文档智能问答系统"""

import logging
import os
import sys

# ================================================================
# 网络修复：必须在导入 sentence_transformers / huggingface_hub 之前执行
# ================================================================

# 1. 清除本机代理（127.0.0.1:7897），否则 HuggingFace 下载会 SSL 断开
for _key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
    _val = os.environ.get(_key, "")
    if "127.0.0.1" in _val and _key in os.environ:
        del os.environ[_key]

# 2. 使用 HuggingFace 国内镜像，不走代理也能快
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# ================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)

logger = logging.getLogger("docqa")
