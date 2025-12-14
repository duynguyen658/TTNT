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

        Flow mới:
        - Agent 1: Thu thập & Phân tích Yêu cầu
        - Agent 2: Chẩn đoán Bệnh từ Hình ảnh
        - Agent 3: Thẩm định Chẩn đoán & Xác định Tác nhân
        - Agent 4: Kiến thức & Kinh nghiệm Thực tế
        - Agent 5: Tổng hợp & Tư vấn Điều trị (này)
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
                "diagnosis": final_advice.get("diagnosis", ""),
                "treatment_plan": final_advice.get("treatment_plan", ""),
                "prevention_measures": final_advice.get("prevention_measures", []),
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
        """Tổng hợp kết quả từ tất cả các agent theo flow mới"""
        synthesis = {
            "user_query": user_query,
            "context": context,
            "agent_results": {
                "information_collection": agent1,  # Agent 1: Thu thập & Phân tích Yêu cầu
                "image_diagnosis": agent2,  # Agent 2: Chẩn đoán Bệnh từ Hình ảnh
                "diagnosis_validation": agent3,  # Agent 3: Thẩm định Chẩn đoán & Xác định Tác nhân
                "knowledge_experience": agent4,  # Agent 4: Kiến thức & Kinh nghiệm Thực tế
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
            confidence = agent2.get("output", {}).get("confidence", 0.0)
            disease_name = agent2.get("output", {}).get("disease_name", "")
            if diagnosis:
                synthesis["key_findings"].append(
                    (
                        "image_diagnosis",
                        f"{disease_name}: {diagnosis[:200]} (độ tin cậy: {confidence:.2%})",
                    )
                )

        if agent3:
            validation = agent3.get("output", {}).get("validation", {})
            pathogen = agent3.get("output", {}).get("pathogen_analysis", {})
            validated_confidence = validation.get("final_confidence", 0.0)
            pathogen_type = pathogen.get("pathogen_type", "unknown")
            if validation:
                synthesis["key_findings"].append(
                    (
                        "validation",
                        f"Độ tin cậy sau thẩm định: {validated_confidence:.2%}, Tác nhân: {pathogen_type}",
                    )
                )

        if agent4:
            knowledge = agent4.get("output", {}).get("knowledge_points", [])
            experience = agent4.get("output", {}).get("practical_experience", [])
            if knowledge or experience:
                synthesis["key_findings"].append(
                    (
                        "knowledge",
                        f"Kiến thức và kinh nghiệm: {len(knowledge)} điểm, {len(experience)} kinh nghiệm",
                    )
                )

        # Phát hiện conflicts (nếu có)
        synthesis["conflicts"] = self._detect_conflicts(agent2, agent3, agent4)

        # Tìm consensus
        synthesis["consensus"] = self._find_consensus(agent2, agent3, agent4)

        return synthesis

    def _detect_conflicts(self, agent2: Dict, agent3: Dict, agent4: Dict) -> List[str]:
        """Phát hiện xung đột giữa các kết quả"""
        conflicts = []

        # Lấy thông tin từ Agent 2 (chẩn đoán ban đầu)
        agent2_confidence = agent2.get("output", {}).get("confidence", 0.0)

        # Lấy thông tin từ Agent 3 (thẩm định)
        agent3_validation = agent3.get("output", {}).get("validation", {})
        agent3_confidence = agent3_validation.get("final_confidence", 0.0)
        agent3_warnings = agent3_validation.get("warnings", [])

        # Kiểm tra xung đột về độ tin cậy
        confidence_diff = abs(agent3_confidence - agent2_confidence)
        if confidence_diff > 0.3:
            conflicts.append(
                f"Độ tin cậy thay đổi đáng kể: {agent2_confidence:.2%} → {agent3_confidence:.2%}"
            )

        # Kiểm tra cảnh báo từ Agent 3
        if agent3_warnings:
            conflicts.extend(agent3_warnings[:2])  # Lấy 2 cảnh báo đầu tiên

        return conflicts

    def _find_consensus(self, agent2: Dict, agent3: Dict, agent4: Dict) -> Dict[str, Any]:
        """Tìm điểm đồng thuận giữa các kết quả"""
        consensus = {"points": [], "confidence": 0.5}

        # Thu thập các điểm chung từ các agent
        all_findings = []

        # Từ Agent 2 (chẩn đoán hình ảnh)
        if agent2:
            findings = agent2.get("output", {}).get("findings", [])
            all_findings.extend(findings)
            disease_name = agent2.get("output", {}).get("disease_name", "")
            if disease_name:
                all_findings.append(f"Chẩn đoán: {disease_name}")

        # Từ Agent 3 (thẩm định)
        if agent3:
            pathogen = agent3.get("output", {}).get("pathogen_analysis", {})
            pathogen_type = pathogen.get("pathogen_type", "")
            if pathogen_type:
                all_findings.append(f"Tác nhân: {pathogen_type}")

        # Từ Agent 4 (kiến thức và kinh nghiệm)
        if agent4:
            knowledge_points = agent4.get("output", {}).get("knowledge_points", [])
            prevention_tips = agent4.get("output", {}).get("prevention_tips", [])
            all_findings.extend(knowledge_points[:3])
            all_findings.extend(prevention_tips[:2])

        consensus["points"] = all_findings[:8]

        # Tính confidence dựa trên số lượng findings và độ tin cậy từ Agent 3
        agent3_confidence = (
            agent3.get("output", {}).get("validation", {}).get("final_confidence", 0.5)
        )
        if len(all_findings) > 3:
            consensus["confidence"] = max(agent3_confidence, 0.6)
        if len(all_findings) > 5:
            consensus["confidence"] = max(agent3_confidence, 0.7)

        return consensus

    async def _generate_final_advice(
        self, synthesis: Dict[str, Any], user_query: str
    ) -> Dict[str, Any]:
        """Tạo tư vấn cuối cùng sử dụng LLM"""
        if not self.client:
            return self._simple_advice(synthesis)

        try:
            # Tạo prompt tổng hợp
            # Lấy thông tin từ các agent
            agent1_info = synthesis.get("agent_results", {}).get("information_collection", {})
            agent2_info = (
                synthesis.get("agent_results", {}).get("image_diagnosis", {}).get("output", {})
            )
            agent3_info = (
                synthesis.get("agent_results", {}).get("diagnosis_validation", {}).get("output", {})
            )
            agent4_info = (
                synthesis.get("agent_results", {}).get("knowledge_experience", {}).get("output", {})
            )

            synthesis_text = f"""
            Tổng hợp kết quả từ các agent:

            1. Agent 1 - Thu thập & Phân tích Yêu cầu:
               - Câu hỏi đã được phân tích: {user_query}
               - Từ khóa: {agent1_info.get('extracted_keywords', [])[:5]}

            2. Agent 2 - Chẩn đoán Bệnh từ Hình ảnh:
               - Chẩn đoán ban đầu: {agent2_info.get('diagnosis', 'N/A')[:300]}
               - Tên bệnh: {agent2_info.get('disease_name', 'N/A')}
               - Độ tin cậy ban đầu: {agent2_info.get('confidence', 0.0):.2%}

            3. Agent 3 - Thẩm định Chẩn đoán & Xác định Tác nhân:
               - Độ tin cậy sau thẩm định: {agent3_info.get('validation', {}).get('final_confidence', 0.0):.2%}
               - Tác nhân gây bệnh: {agent3_info.get('pathogen_analysis', {}).get('pathogen_type', 'N/A')}
               - Cảnh báo: {agent3_info.get('validation', {}).get('warnings', [])[:2]}

            4. Agent 4 - Kiến thức & Kinh nghiệm Thực tế:
               - Kiến thức nông học: {len(agent4_info.get('knowledge_points', []))} điểm
               - Kinh nghiệm thực tế: {len(agent4_info.get('practical_experience', []))} điểm
               - Biện pháp phòng ngừa: {len(agent4_info.get('prevention_tips', []))} điểm

            Các phát hiện chính: {synthesis.get('key_findings', [])}
            Xung đột: {synthesis.get('conflicts', [])}
            Đồng thuận: {synthesis.get('consensus', {})}
            """

            prompt = f"""
            Bạn là chuyên gia nông nghiệp hàng đầu. Dựa trên tổng hợp kết quả từ hệ thống Multi-Agent (5 agents), hãy đưa ra tư vấn điều trị bệnh cây trồng cuối cùng cho người dùng:

            Câu hỏi của người dùng: {user_query}

            {synthesis_text}

            Hãy cung cấp một báo cáo tư vấn đầy đủ và chi tiết, dễ hiểu cho nông dân:

            1. CHẨN ĐOÁN CUỐI CÙNG:
               - Tên bệnh đã được xác định (từ Agent 2 và Agent 3)
               - Tác nhân gây bệnh (nấm/vi khuẩn/virus/dinh dưỡng/môi trường)
               - Triệu chứng chính
               - Độ tin cậy của chẩn đoán (từ Agent 3)
               - Mức độ nghiêm trọng

            2. TƯ VẤN ĐIỀU TRỊ CỤ THỂ:
               - Biện pháp điều trị ngay lập tức (dựa trên tác nhân gây bệnh)
               - Thuốc/phân bón phù hợp (nếu biết)
               - Liều lượng và cách sử dụng chi tiết
               - Thời gian điều trị dự kiến
               - Dấu hiệu cho thấy điều trị có hiệu quả

            3. BIỆN PHÁP PHÒNG NGỪA:
               - Cách phòng ngừa bệnh tái phát (từ kiến thức và kinh nghiệm thực tế)
               - Chăm sóc cây trồng đúng cách
               - Điều kiện môi trường phù hợp
               - Lịch trình chăm sóc và theo dõi

            4. CÁC BƯỚC TIẾP THEO:
               - Hành động ngay lập tức (nếu cần)
               - Theo dõi và đánh giá tiến độ
               - Khi nào cần tham khảo chuyên gia
               - Các dấu hiệu cảnh báo cần chú ý

            5. LƯU Ý QUAN TRỌNG:
               - Cảnh báo về mức độ nghiêm trọng (nếu có)
               - Các cảnh báo từ Agent 3 về nguy cơ nhầm lẫn
               - Những điều cần tránh khi điều trị
               - Khuyến nghị tham khảo thêm (nếu cần)

            Hãy trình bày một cách rõ ràng, dễ hiểu, có cấu trúc và thực tế, phù hợp với nông dân Việt Nam. Sử dụng ngôn ngữ đơn giản, tránh thuật ngữ khoa học phức tạp.
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
                "diagnosis": structured_advice.get("diagnosis", ""),
                "treatment_plan": structured_advice.get("treatment_plan", ""),
                "prevention_measures": structured_advice.get("prevention_measures", []),
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
        diagnosis = ""
        treatment_plan = ""
        prevention_measures = []

        lines = advice_text.split("\n")
        current_section = None

        for line in lines:
            line_lower = line.lower()
            # Xác định section
            if "chẩn đoán" in line_lower or "diagnosis" in line_lower:
                current_section = "diagnosis"
            elif "điều trị" in line_lower or "treatment" in line_lower:
                current_section = "treatment"
            elif "phòng ngừa" in line_lower or "prevention" in line_lower:
                current_section = "prevention"
            elif "khuyến nghị" in line_lower or "recommend" in line_lower:
                current_section = "recommendations"
            elif "bước tiếp theo" in line_lower or "next step" in line_lower:
                current_section = "next_steps"
            elif "lưu ý" in line_lower or "warning" in line_lower or "xung đột" in line_lower:
                current_section = "warnings"

            # Trích xuất nội dung
            if line.strip() and line.strip()[0] in ["-", "•", "1", "2", "3", "4", "5", "*"]:
                item = line.strip().lstrip("-•1234567890.* ").strip()
                if item and len(item) > 10:
                    if current_section == "diagnosis":
                        if not diagnosis:
                            diagnosis = item
                        else:
                            diagnosis += " " + item
                    elif current_section == "treatment":
                        if not treatment_plan:
                            treatment_plan = item
                        else:
                            treatment_plan += " " + item
                    elif current_section == "prevention":
                        prevention_measures.append(item)
                    elif current_section == "recommendations":
                        recommendations.append(item)
                    elif current_section == "next_steps":
                        next_steps.append(item)
                    elif current_section == "warnings":
                        warnings.append(item)
            elif line.strip() and len(line.strip()) > 20:
                # Nếu không có bullet, thêm vào section hiện tại
                if current_section == "diagnosis" and not diagnosis:
                    diagnosis = line.strip()
                elif current_section == "treatment" and not treatment_plan:
                    treatment_plan = line.strip()

        return {
            "summary": advice_text[:300],
            "diagnosis": diagnosis[:500] if diagnosis else advice_text[:200],
            "treatment_plan": treatment_plan[:500] if treatment_plan else "",
            "prevention_measures": prevention_measures[:8],
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
