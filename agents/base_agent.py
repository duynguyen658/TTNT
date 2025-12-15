"""
Base Agent class cho tất cả các agents trong hệ thống
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import config

# Import LLM clients
try:
    from openai import OpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from groq import Groq

    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

try:
    import base64
    from io import BytesIO

    import requests
    from PIL import Image

    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


class BaseAgent(ABC):
    """Lớp cơ sở cho tất cả các agents"""

    def __init__(self, agent_id: str, agent_config: Dict[str, Any]):
        self.agent_id = agent_id
        self.name = agent_config.get("name", agent_id)
        self.model = agent_config.get("model", "")
        self.model_type = agent_config.get("model_type", "text")
        self.temperature = agent_config.get("temperature", 0.5)

        # Provider được định nghĩa trong config cho từng agent
        self.provider = agent_config.get("provider", "local").lower()

        # Khởi tạo client dựa trên provider của agent
        self.client = None
        self.hf_api_key = None

        if self.provider == "huggingface" and HF_AVAILABLE:
            # Sử dụng HuggingFace Inference API (cloud)
            if config.HF_API_KEY:
                self.hf_api_key = config.HF_API_KEY
                print(f"✅ {agent_id}: Sử dụng HuggingFace Inference API (cloud)")
                print(f"   Model: {self.model}")
            else:
                print(f"⚠️  {agent_id}: HF_API_KEY not set, HuggingFace features will be disabled")
                print(f"   Lấy API key tại: https://huggingface.co/settings/tokens")
        elif self.provider == "groq" and GROQ_AVAILABLE:
            if config.GROQ_API_KEY:
                self.client = Groq(api_key=config.GROQ_API_KEY)
                print(f"✅ {agent_id}: Sử dụng Groq API (cloud)")
                print(f"   Model: {self.model}")
            else:
                print(f"⚠️  {agent_id}: GROQ_API_KEY not set, LLM features will be disabled")
        elif self.provider == "local":
            # Local model (YOLO, etc.)
            print(f"✅ {agent_id}: Sử dụng local model")
            print(f"   Model: {self.model}")
        else:
            if self.provider == "huggingface" and not HF_AVAILABLE:
                print(
                    f"⚠️  {agent_id}: requests library not installed. Install with: pip install requests pillow"
                )
            elif self.provider == "groq" and not GROQ_AVAILABLE:
                print(f"⚠️  {agent_id}: Groq library not installed. Install with: pip install groq")

    @abstractmethod
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Xử lý input và trả về kết quả
        Args:
            input_data: Dữ liệu đầu vào từ agent trước hoặc từ người dùng
        Returns:
            Dict chứa kết quả xử lý
        """
        pass

    async def call_hf_model(self, prompt: str, image: Optional[Any] = None) -> str:
        """Gọi HuggingFace Inference API (cloud) để generate text"""
        if not self.hf_api_key:
            return "HuggingFace API key chưa được set"

        try:
            # Chuẩn bị headers
            headers = {
                "Authorization": f"Bearer {self.hf_api_key}",
                "Content-Type": "application/json",
            }

            if self.model_type == "vision" and image is not None:
                # Vision model với image
                # Convert image to base64
                if isinstance(image, str):
                    # image_path
                    with open(image, "rb") as f:
                        image_bytes = f.read()
                elif hasattr(image, "save"):
                    # PIL Image
                    buffered = BytesIO()
                    image.save(buffered, format="PNG")
                    image_bytes = buffered.getvalue()
                else:
                    image_bytes = image

                image_b64 = base64.b64encode(image_bytes).decode("utf-8")

                # Gọi API với image và text
                payload = {
                    "inputs": {"text": prompt, "image": image_b64},
                    "parameters": {
                        "max_new_tokens": 8192,
                        "temperature": self.temperature,
                        "return_full_text": False,
                    },
                }
            else:
                # Text model
                payload = {
                    "inputs": prompt,
                    "parameters": {
                        "max_new_tokens": 8192,
                        "temperature": self.temperature,
                        "return_full_text": False,
                    },
                }

            # Thử các endpoint theo thứ tự ưu tiên
            endpoints = [
                f"https://inference-api.huggingface.co/models/{self.model}",
                f"https://api-inference.huggingface.co/models/{self.model}",
                f"https://hf-inference.co/models/{self.model}",
            ]

            last_error = None
            for api_url in endpoints:
                try:
                    # Gọi API
                    response = requests.post(api_url, headers=headers, json=payload, timeout=60)
                    response.raise_for_status()
                    break  # Thành công, thoát khỏi loop
                except requests.exceptions.HTTPError as e:
                    last_error = e
                    if e.response:
                        try:
                            error_text = (
                                e.response.text[:500]
                                if hasattr(e.response, "text")
                                else str(e.response)
                            )
                            print(
                                f"⚠️  {self.agent_id}: Endpoint {api_url} trả về {e.response.status_code}"
                            )
                            print(f"   Response: {error_text}")
                        except:
                            pass

                    if e.response and e.response.status_code == 404:
                        # 404 - thử endpoint tiếp theo
                        print(
                            f"⚠️  {self.agent_id}: Endpoint {api_url} trả về 404, thử endpoint khác..."
                        )
                        continue
                    else:
                        # Lỗi khác (401, 403, 500, etc.) - không retry
                        raise
                except requests.exceptions.RequestException as e:
                    last_error = e
                    # Network error - thử endpoint tiếp theo
                    print(f"⚠️  {self.agent_id}: Lỗi network với {api_url}, thử endpoint khác...")
                    continue
            else:
                # Tất cả endpoints đều fail
                if last_error:
                    raise last_error

            result = response.json()

            # Parse response
            if isinstance(result, list) and len(result) > 0:
                if "generated_text" in result[0]:
                    return result[0]["generated_text"].strip()
                elif "text" in result[0]:
                    return result[0]["text"].strip()
            elif isinstance(result, dict):
                if "generated_text" in result:
                    return result["generated_text"].strip()
                elif "text" in result:
                    return result["text"].strip()

            # Fallback: return raw result
            return str(result)

        except requests.exceptions.RequestException as e:
            print(f"❌ Lỗi khi gọi HuggingFace API: {e}")
            if hasattr(e.response, "text"):
                print(f"   Response: {e.response.text}")
            return f"Lỗi API: {str(e)}"
        except Exception as e:
            print(f"❌ Lỗi khi gọi HuggingFace model: {e}")
            import traceback

            traceback.print_exc()
            return f"Lỗi: {str(e)}"

    def get_status(self) -> Dict[str, Any]:
        """Trả về trạng thái của agent"""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "model": self.model,
            "model_type": self.model_type,
            "provider": self.provider,
            "status": "ready",
        }
