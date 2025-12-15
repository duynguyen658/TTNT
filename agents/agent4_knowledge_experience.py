from typing import Any, Dict, List

import config
from agents.base_agent import BaseAgent


class KnowledgeExperienceAgent(BaseAgent):
    """Agent bổ sung kiến thức nông học và kinh nghiệm thực tế"""

    def __init__(self):
        super().__init__("agent4", config.AGENT_CONFIG["agent4"])

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Bổ sung kiến thức nông học và kinh nghiệm thực tế
        """
        agent2_result = input_data.get("agent2_output", {})
        agent3_result = input_data.get("agent3_output", {})
        user_query = input_data.get("user_query", "")
        context = input_data.get("context", {})

        # Lấy thông tin từ Agent 2 và Agent 3
        # Lưu ý: agent2_result và agent3_result đã là output trực tiếp từ orchestrator
        diagnosis = agent2_result.get("diagnosis", "")
        disease_name = agent2_result.get("disease_name", "")
        pathogen_info = agent3_result.get("pathogen_analysis", {})
        validation_info = agent3_result.get("validation", {})

        # Bổ sung kiến thức và kinh nghiệm
        knowledge_result = await self._provide_knowledge_and_experience(
            diagnosis, disease_name, pathogen_info, validation_info, user_query, context
        )

        return {
            "agent_id": self.agent_id,
            "status": "completed",
            "output": knowledge_result,
            "next_agents": ["agent5"],  # Chuyển đến Agent 5
        }

    async def _provide_knowledge_and_experience(
        self,
        diagnosis: str,
        disease_name: str,
        pathogen_info: Dict[str, Any],
        validation_info: Dict[str, Any],
        user_query: str,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Bổ sung kiến thức nông học và kinh nghiệm thực tế"""
        if not self.client:
            return self._simple_knowledge(diagnosis, disease_name, pathogen_info)

        try:
            pathogen_type = pathogen_info.get("pathogen_type", "unknown")
            pathogen_name = pathogen_info.get("pathogen_name", disease_name)

            prompt = f"""
            Bạn là chuyên gia nông nghiệp với nhiều năm kinh nghiệm thực tế. Hãy bổ sung kiến thức nông học và kinh nghiệm thực tế cho trường hợp bệnh cây trồng sau:

            QUAN TRỌNG: Bạn PHẢI chỉ sử dụng tiếng Việt trong mọi phản hồi. Không được sử dụng tiếng Anh hoặc bất kỳ ngôn ngữ nào khác.

            Câu hỏi của người dùng: {user_query}
            Ngữ cảnh: {context}

            Thông tin chẩn đoán:
            - Tên bệnh: {disease_name}
            - Mô tả: {diagnosis[:500]}
            - Tác nhân gây bệnh: {pathogen_type} ({pathogen_name})
            - Mức độ nghiêm trọng: {validation_info.get('severity', 'moderate')}

            Hãy cung cấp:

            1. KIẾN THỨC NÔNG HỌC:
               - Nguyên nhân sâu xa của bệnh (điều kiện phát triển, chu kỳ bệnh)
               - Đặc điểm sinh học của tác nhân gây bệnh
               - Các yếu tố môi trường ảnh hưởng (nhiệt độ, độ ẩm, pH đất, ánh sáng)
               - Mối quan hệ giữa cây trồng và tác nhân gây bệnh
               - Chu kỳ phát triển của bệnh trong năm

            2. KINH NGHIỆM THỰC TẾ TỪ NÔNG DÂN:
               - Các dấu hiệu sớm mà nông dân thường quan sát được
               - Thời điểm bệnh thường xuất hiện (mùa, giai đoạn sinh trưởng)
               - Các giống cây dễ bị bệnh hoặc kháng bệnh
               - Kinh nghiệm phòng ngừa từ thực tế
               - Các biện pháp dân gian hoặc truyền thống (nếu có)

            3. LƯU Ý QUAN TRỌNG:
               - Những điều cần tránh khi xử lý bệnh này
               - Thời điểm tốt nhất để can thiệp
               - Các yếu tố có thể làm bệnh nặng thêm
               - Dấu hiệu cho thấy cần can thiệp ngay
               - Cảnh báo về các biện pháp không hiệu quả

            4. KINH NGHIỆM PHÒNG NGỪA:
               - Các biện pháp phòng ngừa hiệu quả từ thực tế
               - Lịch trình chăm sóc phù hợp
               - Cách theo dõi và phát hiện sớm
               - Quản lý môi trường để giảm nguy cơ
               - Kinh nghiệm về giống cây và cách trồng

            5. HỖ TRỢ CHO TƯ VẤN ĐIỀU TRỊ:
               - Gợi ý về cách tiếp cận điều trị dựa trên kinh nghiệm
               - Các phương pháp điều trị đã được chứng minh hiệu quả
               - Thời gian điều trị thực tế thường mất bao lâu
               - Các dấu hiệu cho thấy điều trị có hiệu quả
               - Khi nào cần thay đổi phương pháp điều trị

            Hãy trình bày bằng ngôn ngữ dễ hiểu, thực tế, phù hợp với nông dân Việt Nam.
            """

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "Bạn là một chuyên gia nông nghiệp với nhiều năm kinh nghiệm thực tế trong việc chẩn đoán, phòng ngừa và điều trị bệnh cây trồng. Bạn có kiến thức sâu về nông học và am hiểu các kinh nghiệm thực tế từ nông dân. Bạn có khả năng đưa ra lời khuyên thực tế, dễ hiểu và phù hợp với điều kiện Việt Nam. QUAN TRỌNG: Bạn PHẢI chỉ sử dụng tiếng Việt trong mọi phản hồi. Không được sử dụng tiếng Anh hoặc bất kỳ ngôn ngữ nào khác.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
            )

            knowledge_text = response.choices[0].message.content

            # Trích xuất thông tin có cấu trúc
            structured_knowledge = self._extract_knowledge_info(
                knowledge_text, pathogen_type, disease_name
            )

            return structured_knowledge

        except Exception as e:
            print(f"Error in providing knowledge and experience: {e}")
            return self._simple_knowledge(diagnosis, disease_name, pathogen_info)

    def _simple_knowledge(
        self, diagnosis: str, disease_name: str, pathogen_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Bổ sung kiến thức đơn giản không dùng LLM"""
        pathogen_type = pathogen_info.get("pathogen_type", "unknown")

        # Kiến thức cơ bản dựa trên loại tác nhân
        knowledge_points = []
        prevention_tips = []
        important_notes = []

        if pathogen_type == "fungi":
            knowledge_points.append("Bệnh do nấm thường phát triển trong điều kiện ẩm ướt")
            prevention_tips.append("Đảm bảo thoát nước tốt và không tưới quá nhiều")
            important_notes.append("Tránh tưới nước vào lá vào buổi tối")

        elif pathogen_type == "bacteria":
            knowledge_points.append("Bệnh do vi khuẩn có thể lây lan nhanh qua nước và dụng cụ")
            prevention_tips.append("Vệ sinh dụng cụ và tay trước khi chăm sóc cây")
            important_notes.append("Cách ly cây bị bệnh để tránh lây lan")

        elif pathogen_type == "virus":
            knowledge_points.append("Bệnh do virus thường lây qua côn trùng hoặc dụng cụ")
            prevention_tips.append("Kiểm soát côn trùng và vệ sinh dụng cụ")
            important_notes.append("Cây bị virus thường khó chữa, nên phòng ngừa là chính")

        elif pathogen_type == "nutrition":
            knowledge_points.append("Thiếu dinh dưỡng thường do đất hoặc pH không phù hợp")
            prevention_tips.append("Kiểm tra đất và bón phân đầy đủ, cân đối")
            important_notes.append("Bổ sung dinh dưỡng từ từ, tránh sốc cho cây")

        elif pathogen_type == "environment":
            knowledge_points.append(
                "Bệnh do môi trường thường do stress từ nhiệt độ, độ ẩm, ánh sáng"
            )
            prevention_tips.append("Đảm bảo điều kiện môi trường phù hợp với loại cây")
            important_notes.append("Theo dõi và điều chỉnh môi trường kịp thời")

        return {
            "knowledge_points": knowledge_points,
            "prevention_tips": prevention_tips,
            "important_notes": important_notes,
            "practical_experience": [],
            "treatment_support": [],
            "full_knowledge": f"Kiến thức cơ bản về {disease_name}",
        }

    def _extract_knowledge_info(
        self, knowledge_text: str, pathogen_type: str, disease_name: str
    ) -> Dict[str, Any]:
        """Trích xuất thông tin kiến thức có cấu trúc"""
        knowledge_points = []
        practical_experience = []
        important_notes = []
        prevention_tips = []
        treatment_support = []

        lines = knowledge_text.split("\n")
        current_section = None

        for line in lines:
            line_lower = line.lower()
            # Xác định section
            if "kiến thức" in line_lower or "knowledge" in line_lower:
                current_section = "knowledge"
            elif "kinh nghiệm" in line_lower or "experience" in line_lower:
                current_section = "experience"
            elif "lưu ý" in line_lower or "note" in line_lower or "quan trọng" in line_lower:
                current_section = "notes"
            elif "phòng ngừa" in line_lower or "prevention" in line_lower:
                current_section = "prevention"
            elif "điều trị" in line_lower or "treatment" in line_lower or "tư vấn" in line_lower:
                current_section = "treatment"

            # Trích xuất nội dung
            if line.strip() and line.strip()[0] in ["-", "•", "1", "2", "3", "4", "5", "*"]:
                item = line.strip().lstrip("-•1234567890.* ").strip()
                if item and len(item) > 15:
                    if current_section == "knowledge":
                        knowledge_points.append(item)
                    elif current_section == "experience":
                        practical_experience.append(item)
                    elif current_section == "notes":
                        important_notes.append(item)
                    elif current_section == "prevention":
                        prevention_tips.append(item)
                    elif current_section == "treatment":
                        treatment_support.append(item)

        # Nếu không tìm thấy section, thêm tất cả vào knowledge_points
        if not knowledge_points and not practical_experience:
            sentences = knowledge_text.split(".")
            for sentence in sentences:
                if len(sentence.strip()) > 30:
                    knowledge_points.append(sentence.strip())

        return {
            "knowledge_points": knowledge_points[:8],
            "practical_experience": practical_experience[:8],
            "important_notes": important_notes[:5],
            "prevention_tips": prevention_tips[:8],
            "treatment_support": treatment_support[:5],
            "full_knowledge": knowledge_text,
        }
