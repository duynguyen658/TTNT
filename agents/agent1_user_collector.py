from typing import Any, Dict

import config

from agents.base_agent import BaseAgent


class UserInformationCollector(BaseAgent):
    def __init__(self):
        super().__init__("agent1", config.AGENT_CONFIG["agent1"])

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        user_query = input_data.get("user_query", "")
        user_context = input_data.get("user_context", {})

        # Tiền xử lý thông tin
        processed_info = {
            "original_query": user_query,
            "user_context": user_context,
            "extracted_keywords": self._extract_keywords(user_query),
            "query_type": self._classify_query(user_query),
            "processed_query": self._preprocess_query(user_query),
            "requires_image": self._check_image_requirement(user_query),
            "requires_dataset": self._check_dataset_requirement(user_query),
            "requires_social_media": self._check_social_media_requirement(user_query),
        }

        # Sử dụng LLM để làm giàu thông tin nếu có
        if self.client:
            enriched_info = await self._enrich_with_llm(user_query, user_context)
            processed_info.update(enriched_info)

        return {
            "agent_id": self.agent_id,
            "status": "completed",
            "output": processed_info,
            "next_agents": self._determine_next_agents(processed_info),
        }

    def _extract_keywords(self, query: str) -> list:
        """Trích xuất từ khóa về bệnh cây trồng từ câu hỏi"""
        # Từ khóa liên quan đến bệnh cây trồng
        plant_disease_keywords = [
            "bệnh",
            "cây",
            "lá",
            "thân",
            "rễ",
            "quả",
            "hoa",
            "nấm",
            "sâu",
            "côn trùng",
            "vàng",
            "héo",
            "thối",
            "đốm",
            "cháy",
            "rụng",
            "khô",
            "phấn",
            "mốc",
            "lúa",
            "ngô",
            "cà chua",
            "ớt",
            "dưa",
            "bầu",
            "bí",
            "cà tím",
            "rau",
        ]

        stop_words = ["là", "của", "và", "hoặc", "từ", "với", "cho", "về", "có", "được", "bị"]
        words = query.lower().split()
        keywords = [w for w in words if w not in stop_words and len(w) > 2]

        # Ưu tiên từ khóa liên quan đến bệnh cây
        prioritized_keywords = [
            w for w in keywords if any(kw in w for kw in plant_disease_keywords)
        ]
        other_keywords = [w for w in keywords if w not in prioritized_keywords]

        return (prioritized_keywords + other_keywords)[:15]  # Tăng lên 15 từ khóa

    def _classify_query(self, query: str) -> str:
        """Phân loại loại câu hỏi về bệnh cây trồng"""
        query_lower = query.lower()

        if any(
            word in query_lower
            for word in ["hình ảnh", "ảnh", "hình", "xem", "phân tích ảnh", "nhận dạng"]
        ):
            return "image_analysis"
        elif any(
            word in query_lower
            for word in ["dữ liệu", "dataset", "thống kê", "phân tích dữ liệu", "dữ liệu bệnh"]
        ):
            return "dataset_analysis"
        elif any(
            word in query_lower
            for word in ["mạng xã hội", "facebook", "twitter", "social", "diễn đàn", "cộng đồng"]
        ):
            return "social_media"
        elif any(
            word in query_lower
            for word in ["tư vấn", "khuyên", "gợi ý", "nên", "điều trị", "chữa", "phòng"]
        ):
            return "consultation"
        elif any(word in query_lower for word in ["bệnh", "chẩn đoán", "nhận dạng"]):
            return "disease_diagnosis"
        else:
            return "general"

    def _preprocess_query(self, query: str) -> str:
        """Tiền xử lý câu hỏi"""
        # Loại bỏ ký tự đặc biệt, chuẩn hóa khoảng trắng
        processed = " ".join(query.split())
        return processed.strip()

    def _check_image_requirement(self, query: str) -> bool:
        """Kiểm tra xem có cần phân tích hình ảnh bệnh cây không"""
        image_keywords = [
            "hình ảnh",
            "ảnh",
            "hình",
            "xem",
            "phân tích ảnh",
            "chẩn đoán ảnh",
            "nhận dạng",
            "ảnh cây",
            "hình cây",
            "ảnh lá",
            "hình lá",
        ]
        # Nếu có từ "bệnh" hoặc "cây" kèm theo từ khóa ảnh, ưu tiên phân tích ảnh
        has_disease_context = any(word in query.lower() for word in ["bệnh", "cây", "lá", "thân"])
        has_image_keyword = any(keyword in query.lower() for keyword in image_keywords)
        return has_image_keyword or (has_disease_context and "ảnh" in query.lower())

    def _check_dataset_requirement(self, query: str) -> bool:
        """Kiểm tra xem có cần phân tích dataset bệnh cây không"""
        dataset_keywords = [
            "dữ liệu",
            "dataset",
            "thống kê",
            "phân tích dữ liệu",
            "dữ liệu bệnh",
            "dataset bệnh",
            "thống kê bệnh",
        ]
        # Luôn khuyến nghị phân tích dataset nếu có dataset_path trong input
        return any(keyword in query.lower() for keyword in dataset_keywords)

    def _check_social_media_requirement(self, query: str) -> bool:
        """Kiểm tra xem có cần tìm kiếm thông tin bệnh cây trên mạng xã hội không"""
        social_keywords = [
            "mạng xã hội",
            "facebook",
            "twitter",
            "social media",
            "cộng đồng",
            "diễn đàn",
            "nhóm",
            "kinh nghiệm",
            "người dùng",
            "nông dân",
        ]
        # Nếu câu hỏi về bệnh cây, nên tìm kiếm thêm trên mạng xã hội
        has_disease_query = any(
            word in query.lower() for word in ["bệnh", "cây", "tư vấn", "kinh nghiệm"]
        )
        return any(keyword in query.lower() for keyword in social_keywords) or has_disease_query

    async def _enrich_with_llm(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Sử dụng LLM để làm giàu thông tin"""
        if not self.client:
            return {}

        try:
            prompt = f"""
            Phân tích câu hỏi về bệnh cây trồng và trả về thông tin có cấu trúc:
            Câu hỏi: {query}
            Ngữ cảnh: {context}

            Hãy trả về:
            1. Mục đích chính của câu hỏi (nhận dạng bệnh, tư vấn điều trị, phòng ngừa, v.v.)
            2. Loại cây trồng được đề cập (nếu có)
            3. Triệu chứng bệnh được mô tả (nếu có)
            4. Thông tin quan trọng cần thu thập thêm
            5. Loại tư vấn phù hợp (chẩn đoán, điều trị, phòng ngừa)
            """

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "Bạn là một chuyên gia nông nghiệp AI, chuyên thu thập và phân tích thông tin về bệnh cây trồng từ người dùng.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
            )

            llm_output = response.choices[0].message.content

            return {"llm_analysis": llm_output, "enriched": True}
        except Exception as e:
            print(f"Error in LLM enrichment: {e}")
            return {"enriched": False, "error": str(e)}

    def _determine_next_agents(self, processed_info: Dict[str, Any]) -> list:
        """Xác định các agent tiếp theo cần gọi"""
        next_agents = []

        if processed_info.get("requires_image"):
            next_agents.append("agent2")
        if processed_info.get("requires_dataset"):
            next_agents.append("agent3")
        if processed_info.get("requires_social_media"):
            next_agents.append("agent4")

        # Luôn luôn cần agent5 để tổng hợp
        if "agent5" not in next_agents:
            next_agents.append("agent5")

        return next_agents
