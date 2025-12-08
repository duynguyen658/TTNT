"""
Base Agent class cho tất cả các agents trong hệ thống
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import config
from openai import OpenAI


class BaseAgent(ABC):
    """Lớp cơ sở cho tất cả các agents"""

    def __init__(self, agent_id: str, agent_config: Dict[str, Any]):
        self.agent_id = agent_id
        self.name = agent_config.get("name", agent_id)
        self.model = agent_config.get("model", "gpt-4")
        self.temperature = agent_config.get("temperature", 0.5)
        self.client = OpenAI(api_key=config.OPENAI_API_KEY) if config.OPENAI_API_KEY else None

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
