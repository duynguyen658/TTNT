"""
Hệ thống điều phối các agents
"""

import asyncio
from typing import Any, Dict, List, Optional

from agents import (
    DatasetDiagnosisAgent,
    FinalSynthesisAgent,
    ImageDiagnosisAgent,
    SocialMediaSearchAgent,
    UserInformationCollector,
)


class AgentOrchestrator:
    """Điều phối việc thực thi các agents"""

    def __init__(self):
        self.agents = {
            "agent1": UserInformationCollector(),
            "agent2": ImageDiagnosisAgent(),
            "agent3": DatasetDiagnosisAgent(),
            "agent4": SocialMediaSearchAgent(),
            "agent5": FinalSynthesisAgent(),
        }
        self.execution_log = []

    async def execute(self, user_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Thực thi pipeline các agents
        Args:
            user_input: {
                "user_query": str,
                "user_context": dict (optional),
                "image_path": str (optional),
                "image_data": str (optional, base64),
                "dataset_path": str (optional),
                "dataset_data": dict/DataFrame (optional),
                "search_query": str (optional)
            }
        Returns:
            Dict chứa kết quả cuối cùng từ tất cả agents
        """
        try:
            # Bước 1: Agent 1 - Thu thập thông tin
            self._log("Bắt đầu với Agent 1: Thu thập thông tin người dùng")
            agent1_input = {
                "user_query": user_input.get("user_query", ""),
                "user_context": user_input.get("user_context", {}),
            }
            agent1_result = await self.agents["agent1"].process(agent1_input)
            self._log(f"Agent 1 hoàn thành: {agent1_result.get('status')}")

            # Chuẩn bị dữ liệu cho các agent tiếp theo
            processed_info = agent1_result.get("output", {})
            next_agents = agent1_result.get("next_agents", [])

            # Tạo context chung
            shared_context = {
                "user_query": user_input.get("user_query", ""),
                "original_context": user_input.get("user_context", {}),
                "processed_info": processed_info,
            }

            # Bước 2-4: Chạy các agent song song (nếu có thể)
            agent_results = {"agent1_output": agent1_result.get("output", {})}

            tasks = []

            # Agent 2 - Chẩn đoán hình ảnh
            if "agent2" in next_agents and processed_info.get("requires_image"):
                self._log("Khởi động Agent 2: Chẩn đoán hình ảnh")
                agent2_input = {
                    "image_path": user_input.get("image_path"),
                    "image_data": user_input.get("image_data"),
                    "user_query": user_input.get("user_query", ""),
                    "context": shared_context,
                }
                tasks.append(("agent2", self.agents["agent2"].process(agent2_input)))

            # Agent 3 - Chẩn đoán dataset
            if "agent3" in next_agents and processed_info.get("requires_dataset"):
                self._log("Khởi động Agent 3: Chẩn đoán dataset")
                agent3_input = {
                    "dataset_path": user_input.get("dataset_path"),
                    "dataset_data": user_input.get("dataset_data"),
                    "user_query": user_input.get("user_query", ""),
                    "context": shared_context,
                }
                tasks.append(("agent3", self.agents["agent3"].process(agent3_input)))

            # Agent 4 - Tìm kiếm mạng xã hội
            if "agent4" in next_agents and processed_info.get("requires_social_media"):
                self._log("Khởi động Agent 4: Tìm kiếm mạng xã hội")
                agent4_input = {
                    "search_query": user_input.get(
                        "search_query", user_input.get("user_query", "")
                    ),
                    "keywords": processed_info.get("extracted_keywords", []),
                    "user_query": user_input.get("user_query", ""),
                    "context": shared_context,
                }
                tasks.append(("agent4", self.agents["agent4"].process(agent4_input)))

            # Chạy các agent song song
            if tasks:
                self._log(f"Chạy {len(tasks)} agent(s) song song")
                results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)

                for (agent_id, _), result in zip(tasks, results):
                    if isinstance(result, Exception):
                        self._log(f"Lỗi ở {agent_id}: {result}")
                        agent_results[f"{agent_id}_output"] = {
                            "error": str(result),
                            "status": "error",
                        }
                    else:
                        self._log(f"{agent_id} hoàn thành: {result.get('status')}")
                        agent_results[f"{agent_id}_output"] = result.get("output", {})

            # Bước 5: Agent 5 - Tổng hợp cuối cùng
            self._log("Khởi động Agent 5: Tổng hợp và tư vấn cuối cùng")
            agent5_input = {
                **agent_results,
                "user_query": user_input.get("user_query", ""),
                "original_context": user_input.get("user_context", {}),
            }
            agent5_result = await self.agents["agent5"].process(agent5_input)
            self._log(f"Agent 5 hoàn thành: {agent5_result.get('status')}")

            # Tổng hợp kết quả cuối cùng
            final_result = {
                "status": "completed",
                "user_query": user_input.get("user_query", ""),
                "agent_results": {
                    "agent1": agent1_result,
                    "agent2": agent_results.get("agent2_output"),
                    "agent3": agent_results.get("agent3_output"),
                    "agent4": agent_results.get("agent4_output"),
                    "agent5": agent5_result,
                },
                "final_advice": agent5_result.get("output", {}),
                "execution_log": self.execution_log,
            }

            self._log("Hoàn thành tất cả agents")
            return final_result

        except Exception as e:
            self._log(f"Lỗi trong orchestrator: {e}")
            return {"status": "error", "error": str(e), "execution_log": self.execution_log}

    def _log(self, message: str):
        """Ghi log"""
        log_entry = f"[{len(self.execution_log) + 1}] {message}"
        self.execution_log.append(log_entry)
        print(log_entry)

    def get_agent_status(self) -> Dict[str, Any]:
        """Lấy trạng thái của tất cả agents"""
        return {agent_id: agent.get_status() for agent_id, agent in self.agents.items()}

    def reset(self):
        """Reset orchestrator"""
        self.execution_log = []
