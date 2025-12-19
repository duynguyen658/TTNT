import asyncio
from typing import Any, Dict, List, Optional

from agents import (
    DiagnosisValidatorAgent,
    FinalSynthesisAgent,
    ImageDiagnosisAgent,
    KnowledgeExperienceAgent,
    UserInformationCollector,
)


class AgentOrchestrator:
    """Điều phối việc thực thi các agents"""

    def __init__(self):
        self.agents = {
            "agent1": UserInformationCollector(),
            "agent2": ImageDiagnosisAgent(),
            "agent3": DiagnosisValidatorAgent(),
            "agent4": KnowledgeExperienceAgent(),
            "agent5": FinalSynthesisAgent(),
        }
        self.execution_log = []

    async def execute(self, user_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Thực thi pipeline các agents theo flow mới:
        Agent 1 → Agent 2 → Agent 3 → Agent 4 → Agent 5

        Args:
            user_input: {
                "user_query": str,
                "user_context": dict (optional),
                "image_path": str (optional),
                "image_data": str (optional, base64),
            }
        Returns:
            Dict chứa kết quả cuối cùng từ tất cả agents
        """
        try:
            # Tạo context chung
            shared_context = {
                "user_query": user_input.get("user_query", ""),
                "original_context": user_input.get("user_context", {}),
            }

            # Bước 1: Agent 1 - Thu thập & Phân tích Yêu cầu
            self._log("🟢 Bắt đầu Agent 1: Thu thập & Phân tích Yêu cầu")
            agent1_input = {
                "user_query": user_input.get("user_query", ""),
                "user_context": user_input.get("user_context", {}),
            }
            agent1_result = await self.agents["agent1"].process(agent1_input)
            self._log(f"✅ Agent 1 hoàn thành: {agent1_result.get('status')}")
            agent_results = {"agent1_output": agent1_result.get("output", {})}

            # Bước 2: Agent 2 - Chẩn đoán Bệnh từ Hình ảnh (nếu có hình ảnh)
            agent2_result = None
            processed_info = agent1_result.get("output", {})
            if (
                processed_info.get("requires_image")
                or user_input.get("image_path")
                or user_input.get("image_data")
            ):
                self._log("🟢 Khởi động Agent 2: Chẩn đoán Bệnh từ Hình ảnh (Vision/YOLO)")
                agent2_input = {
                    "image_path": user_input.get("image_path"),
                    "image_data": user_input.get("image_data"),
                    "user_query": user_input.get("user_query", ""),
                    "context": {**shared_context, "processed_info": processed_info},
                }
                agent2_result = await self.agents["agent2"].process(agent2_input)
                self._log(f"✅ Agent 2 hoàn thành: {agent2_result.get('status')}")
                agent_results["agent2_output"] = agent2_result.get("output", {})
            else:
                self._log("⏭️  Bỏ qua Agent 2: Không có hình ảnh")
                agent_results["agent2_output"] = {}

            # Bước 3: Agent 3 - Thẩm định Chẩn đoán & Xác định Tác nhân
            self._log("🟡 Khởi động Agent 3: Thẩm định Chẩn đoán & Xác định Tác nhân gây bệnh")
            agent3_input = {
                "agent2_output": agent_results.get("agent2_output", {}),
                "user_query": user_input.get("user_query", ""),
                "context": {**shared_context, "processed_info": processed_info},
            }
            agent3_result = await self.agents["agent3"].process(agent3_input)
            self._log(f"✅ Agent 3 hoàn thành: {agent3_result.get('status')}")
            agent_results["agent3_output"] = agent3_result.get("output", {})

            # Bước 4: Agent 4 - Kiến thức & Kinh nghiệm Thực tế
            self._log("🔵 Khởi động Agent 4: Kiến thức & Kinh nghiệm Thực tế")
            agent4_input = {
                "agent2_output": agent_results.get("agent2_output", {}),
                "agent3_output": agent_results.get("agent3_output", {}),
                "user_query": user_input.get("user_query", ""),
                "context": {**shared_context, "processed_info": processed_info},
            }
            agent4_result = await self.agents["agent4"].process(agent4_input)
            self._log(f"✅ Agent 4 hoàn thành: {agent4_result.get('status')}")
            agent_results["agent4_output"] = agent4_result.get("output", {})

            # Bước 5: Agent 5 - Tổng hợp & Tư vấn Điều trị
            self._log("🟢 Khởi động Agent 5: Tổng hợp & Tư vấn Điều trị")
            agent5_input = {
                **agent_results,
                "user_query": user_input.get("user_query", ""),
                "original_context": user_input.get("user_context", {}),
            }
            agent5_result = await self.agents["agent5"].process(agent5_input)
            self._log(f"✅ Agent 5 hoàn thành: {agent5_result.get('status')}")

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

            self._log("🎉 Hoàn thành tất cả agents")
            return final_result

        except Exception as e:
            self._log(f"❌ Lỗi trong orchestrator: {e}")
            import traceback

            error_trace = traceback.format_exc()
            print(f"\n❌ Orchestrator Exception: {e}")
            print(error_trace)
            return {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__,
                "execution_log": self.execution_log,
            }

    def _log(self, message: str):
        """Ghi log"""
        log_entry = f"[{len(self.execution_log) + 1}] {message}"
        self.execution_log.append(log_entry)
        # Print với format đẹp hơn
        print(f"  📌 {log_entry}")

    def get_agent_status(self) -> Dict[str, Any]:
        """Lấy trạng thái của tất cả agents"""
        return {agent_id: agent.get_status() for agent_id, agent in self.agents.items()}

    def reset(self):
        """Reset orchestrator"""
        self.execution_log = []
