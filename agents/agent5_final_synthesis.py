"""
Agent 5: Tổng hợp & tư vấn điều trị bệnh cây trồng cuối cùng
"""

from typing import Any, Dict, List

import config

from agents.base_agent import BaseAgent


class FinalSynthesisAgent(BaseAgent):
    """Agent tổng hợp kết quả từ tất cả các agent và đưa ra tư vấn điều trị bệnh cây trồng cuối cùng"""

    def __init__(self):
        super().__init__("agent5", config.AGENT_CONFIG["agent5"])

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Tổng hợp kết quả từ tất cả các agent và đưa ra tư vấn điều trị bệnh cây trồng cuối cùng
        """
        # Thu thập kết quả từ các agent trước
        agent1_result = input_data.get("agent1_output", {})
        agent2_result = input_data.get("agent2_output", {})
        agent3_result = input_data.get("agent3_output", {})
        agent4_result = input_data.get("agent4_output", {})

        user_query = input_data.get("user_query", "")
        original_context = input_data.get("original_context", {})

        # Tổng hợp tất cả thông tin
        synthesis = await self._synthesize_all_results(
            agent1_result, agent2_result, agent3_result, agent4_result, user_query, original_context
        )

        # Tạo tư vấn cuối cùng
        final_advice = await self._generate_final_advice(synthesis, user_query)

        return {
            "agent_id": self.agent_id,
            "status": "completed",
            "output": {
                "synthesis": synthesis,
                "final_advice": final_advice,
                "confidence_score": self._calculate_confidence(synthesis),
                "recommendations": final_advice.get("recommendations", []),
                "next_steps": final_advice.get("next_steps", []),
            },
        }

    async def _synthesize_all_results(
        self,
        agent1: Dict,
        agent2: Dict,
        agent3: Dict,
        agent4: Dict,
        user_query: str,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Tổng hợp kết quả từ tất cả các agent"""
        synthesis = {
            "user_query": user_query,
            "context": context,
            "agent_results": {
                "information_collection": agent1,
                "image_diagnosis": agent2,
                "dataset_diagnosis": agent3,
                "social_media_search": agent4,
            },
            "key_findings": [],
            "conflicts": [],
            "consensus": {},
        }

        # Thu thập key findings từ mỗi agent
        if agent1:
            findings = agent1.get("output", {}).get("extracted_keywords", [])
            synthesis["key_findings"].extend([("information", f) for f in findings])

        if agent2:
            diagnosis = agent2.get("output", {}).get("diagnosis", "")
            if diagnosis:
                synthesis["key_findings"].append(("image", diagnosis[:200]))

        if agent3:
            diagnosis = agent3.get("output", {}).get("llm_diagnosis", {}).get("diagnosis", "")
            if diagnosis:
                synthesis["key_findings"].append(("dataset", diagnosis[:200]))

        if agent4:
            summary = agent4.get("output", {}).get("summary", {}).get("summary", "")
            if summary:
                synthesis["key_findings"].append(("social_media", summary[:200]))

        # Phát hiện conflicts (nếu có)
        synthesis["conflicts"] = self._detect_conflicts(agent2, agent3, agent4)

        # Tìm consensus
        synthesis["consensus"] = self._find_consensus(agent2, agent3, agent4)

        return synthesis

    def _detect_conflicts(self, agent2: Dict, agent3: Dict, agent4: Dict) -> List[str]:
        """Phát hiện xung đột giữa các kết quả"""
        conflicts = []

        # Logic đơn giản để phát hiện conflicts
        # Trong thực tế, có thể sử dụng semantic similarity

        agent2_diag = str(agent2.get("output", {}).get("diagnosis", "")).lower()
        agent3_diag = str(
            agent3.get("output", {}).get("llm_diagnosis", {}).get("diagnosis", "")
        ).lower()

        # Kiểm tra xung đột cơ bản (ví dụ: positive vs negative)
        positive_words = ["tốt", "tích cực", "khỏe", "bình thường", "good", "positive"]
        negative_words = ["xấu", "tiêu cực", "bệnh", "bất thường", "bad", "negative"]

        agent2_positive = any(word in agent2_diag for word in positive_words)
        agent2_negative = any(word in agent2_diag for word in negative_words)
        agent3_positive = any(word in agent3_diag for word in positive_words)
        agent3_negative = any(word in agent3_diag for word in negative_words)

        if (agent2_positive and agent3_negative) or (agent2_negative and agent3_positive):
            conflicts.append("Xung đột giữa chẩn đoán hình ảnh và chẩn đoán dataset")

        return conflicts

    def _find_consensus(self, agent2: Dict, agent3: Dict, agent4: Dict) -> Dict[str, Any]:
        """Tìm điểm đồng thuận giữa các kết quả"""
        consensus = {"points": [], "confidence": 0.5}

        # Thu thập các điểm chung
        all_findings = []

        if agent2:
            findings = agent2.get("output", {}).get("findings", [])
            all_findings.extend(findings)

        if agent3:
            insights = agent3.get("output", {}).get("llm_diagnosis", {}).get("insights", [])
            all_findings.extend(insights)

        if agent4:
            key_points = agent4.get("output", {}).get("summary", {}).get("key_points", [])
            all_findings.extend(key_points)

        consensus["points"] = all_findings[:5]

        # Tính confidence dựa trên số lượng findings và conflicts
        if len(all_findings) > 3:
            consensus["confidence"] = 0.7
        if len(all_findings) > 5:
            consensus["confidence"] = 0.8

        return consensus

    async def _generate_final_advice(
        self, synthesis: Dict[str, Any], user_query: str
    ) -> Dict[str, Any]:
        """Tạo tư vấn cuối cùng sử dụng LLM"""
        if not self.client:
            return self._simple_advice(synthesis)

        try:
            # Tạo prompt tổng hợp
            synthesis_text = f"""
            Tổng hợp kết quả từ các agent:

            1. Thu thập thông tin: {synthesis.get('agent_results', {}).get('information_collection', {})}
            2. Chẩn đoán hình ảnh: {synthesis.get('agent_results', {}).get('image_diagnosis', {}).get('output', {}).get('diagnosis', 'N/A')}
            3. Chẩn đoán dataset: {synthesis.get('agent_results', {}).get('dataset_diagnosis', {}).get('output', {}).get('llm_diagnosis', {}).get('diagnosis', 'N/A')}
            4. Tìm kiếm mạng xã hội: {synthesis.get('agent_results', {}).get('social_media_search', {}).get('output', {}).get('summary', {}).get('summary', 'N/A')}

            Các phát hiện chính: {synthesis.get('key_findings', [])}
            Xung đột: {synthesis.get('conflicts', [])}
            Đồng thuận: {synthesis.get('consensus', {})}
            """

            prompt = f"""
            Bạn là chuyên gia nông nghiệp hàng đầu. Dựa trên tổng hợp kết quả từ nhiều nguồn (hình ảnh, dataset, mạng xã hội), hãy đưa ra tư vấn điều trị bệnh cây trồng cuối cùng cho người dùng:

            Câu hỏi của người dùng: {user_query}

            {synthesis_text}

            Hãy cung cấp một báo cáo tư vấn đầy đủ và chi tiết:

            1. TÓM TẮT TỔNG QUAN:
               - Tình trạng bệnh cây được xác định
               - Mức độ nghiêm trọng
               - Nguyên nhân có thể

            2. CHẨN ĐOÁN TỔNG HỢP:
               - Tên bệnh (nếu xác định được)
               - Triệu chứng chính
               - So sánh kết quả từ các nguồn khác nhau
               - Độ tin cậy của chẩn đoán

            3. TƯ VẤN ĐIỀU TRỊ CỤ THỂ:
               - Biện pháp điều trị ngay lập tức
               - Thuốc/phân bón phù hợp (nếu biết)
               - Liều lượng và cách sử dụng
               - Thời gian điều trị dự kiến

            4. BIỆN PHÁP PHÒNG NGỪA:
               - Cách phòng ngừa bệnh tái phát
               - Chăm sóc cây trồng đúng cách
               - Điều kiện môi trường phù hợp

            5. CÁC BƯỚC TIẾP THEO:
               - Hành động ngay lập tức (nếu cần)
               - Theo dõi và đánh giá
               - Khi nào cần tham khảo chuyên gia

            6. LƯU Ý QUAN TRỌNG:
               - Cảnh báo về mức độ nghiêm trọng (nếu có)
               - Xung đột giữa các nguồn thông tin (nếu có)
               - Khuyến nghị tham khảo thêm (nếu cần)

            Hãy trình bày một cách rõ ràng, dễ hiểu, có cấu trúc và thực tế, phù hợp với nông dân Việt Nam.
            """

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "Bạn là một chuyên gia nông nghiệp và bệnh học thực vật hàng đầu với nhiều năm kinh nghiệm. Bạn có khả năng tổng hợp thông tin từ nhiều nguồn (hình ảnh, dữ liệu, kinh nghiệm cộng đồng) để đưa ra tư vấn điều trị bệnh cây trồng chính xác, thực tế và hữu ích.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                max_tokens=2500,
            )

            advice_text = response.choices[0].message.content

            # Trích xuất structured advice
            structured_advice = self._extract_structured_advice(advice_text)

            return {
                "full_advice": advice_text,
                "summary": structured_advice.get("summary", ""),
                "recommendations": structured_advice.get("recommendations", []),
                "next_steps": structured_advice.get("next_steps", []),
                "warnings": structured_advice.get("warnings", []),
            }

        except Exception as e:
            print(f"Error in generating final advice: {e}")
            return self._simple_advice(synthesis)

    def _simple_advice(self, synthesis: Dict[str, Any]) -> Dict[str, Any]:
        """Tạo tư vấn đơn giản không dùng LLM"""
        return {
            "summary": f"Dựa trên {len(synthesis.get('key_findings', []))} phát hiện từ các nguồn khác nhau",
            "recommendations": [
                "Xem xét kỹ các kết quả từ các agent",
                "Tham khảo ý kiến chuyên gia nếu cần",
            ],
            "next_steps": ["Xem lại chi tiết từng agent", "Thực hiện các khuyến nghị"],
            "warnings": synthesis.get("conflicts", []),
        }

    def _extract_structured_advice(self, advice_text: str) -> Dict[str, Any]:
        """Trích xuất tư vấn có cấu trúc từ text"""
        recommendations = []
        next_steps = []
        warnings = []

        lines = advice_text.split("\n")
        current_section = None

        for line in lines:
            line_lower = line.lower()
            if "khuyến nghị" in line_lower or "recommend" in line_lower:
                current_section = "recommendations"
            elif "bước tiếp theo" in line_lower or "next step" in line_lower:
                current_section = "next_steps"
            elif "lưu ý" in line_lower or "warning" in line_lower or "xung đột" in line_lower:
                current_section = "warnings"
            elif line.strip() and line.strip()[0] in ["-", "•", "1", "2", "3", "4", "5"]:
                item = line.strip().lstrip("-•1234567890. ").strip()
                if item and len(item) > 10:
                    if current_section == "recommendations":
                        recommendations.append(item)
                    elif current_section == "next_steps":
                        next_steps.append(item)
                    elif current_section == "warnings":
                        warnings.append(item)

        return {
            "summary": advice_text[:300],
            "recommendations": recommendations[:5],
            "next_steps": next_steps[:5],
            "warnings": warnings[:3],
        }

    def _calculate_confidence(self, synthesis: Dict[str, Any]) -> float:
        """Tính toán độ tin cậy của kết quả tổng hợp"""
        confidence = 0.5

        # Tăng confidence nếu có nhiều findings
        findings_count = len(synthesis.get("key_findings", []))
        if findings_count > 3:
            confidence += 0.1
        if findings_count > 5:
            confidence += 0.1

        # Giảm confidence nếu có conflicts
        conflicts_count = len(synthesis.get("conflicts", []))
        confidence -= conflicts_count * 0.1

        # Tăng confidence nếu có consensus
        consensus_confidence = synthesis.get("consensus", {}).get("confidence", 0.5)
        confidence = (confidence + consensus_confidence) / 2

        return min(max(confidence, 0.0), 1.0)
