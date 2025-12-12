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


class BaseAgent(ABC):
    """Lớp cơ sở cho tất cả các agents"""

    def __init__(self, agent_id: str, agent_config: Dict[str, Any]):
        self.agent_id = agent_id
        self.name = agent_config.get("name", agent_id)
        self.model = agent_config.get("model", "gpt-4")
        self.temperature = agent_config.get("temperature", 0.5)

        # Khởi tạo client dựa trên provider
        self.client = None
        self.provider = config.LLM_PROVIDER.lower()

        if self.provider == "groq" and GROQ_AVAILABLE:
            if config.GROQ_API_KEY:
                self.client = Groq(api_key=config.GROQ_API_KEY)
            else:
                print(f"⚠️  {agent_id}: GROQ_API_KEY not set, LLM features will be disabled")
        elif self.provider == "openai" and OPENAI_AVAILABLE:
            if config.OPENAI_API_KEY:
                self.client = OpenAI(api_key=config.OPENAI_API_KEY)
            else:
                print(f"⚠️  {agent_id}: OPENAI_API_KEY not set, LLM features will be disabled")
        else:
            if self.provider == "groq" and not GROQ_AVAILABLE:
                print(f"⚠️  {agent_id}: Groq library not installed. Install with: pip install groq")
            elif self.provider == "openai" and not OPENAI_AVAILABLE:
                print(
                    f"⚠️  {agent_id}: OpenAI library not installed. Install with: pip install openai"
                )

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

    def get_status(self) -> Dict[str, Any]:
        """Trả về trạng thái của agent"""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "model": self.model,
            "status": "ready",
        }
