"""
Agent 3: Thẩm định Chẩn đoán & Xác định Tác nhân gây bệnh
"""

from typing import Any, Dict, Optional

import config
from agents.base_agent import BaseAgent


class DiagnosisValidatorAgent(BaseAgent):
    """Agent thẩm định chẩn đoán và xác định tác nhân gây bệnh"""

    def __init__(self):
        super().__init__("agent3", config.AGENT_CONFIG["agent3"])

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Thẩm định chẩn đoán từ Agent 2 và xác định tác nhân gây bệnh
        """
        agent2_result = input_data.get("agent2_output", {})
        user_query = input_data.get("user_query", "")
        context = input_data.get("context", {})

        # Lấy thông tin chẩn đoán từ Agent 2
        # Lưu ý: agent2_result đã là output trực tiếp từ orchestrator (không cần .get("output", {}))
        diagnosis = agent2_result.get("diagnosis", "")
        confidence = agent2_result.get("confidence", 0.0)
        disease_name = agent2_result.get("disease_name", "")
        findings = agent2_result.get("findings", [])

        # Debug logging
        print(f"🔍 [Agent 3 Debug] agent2_result keys: {list(agent2_result.keys())}")
        print(f"🔍 [Agent 3 Debug] disease_name: {disease_name}")
        print(f"🔍 [Agent 3 Debug] confidence: {confidence}")
        print(f"🔍 [Agent 3 Debug] diagnosis length: {len(diagnosis) if diagnosis else 0}")

        # Thẩm định chẩn đoán
        validation_result = await self._validate_diagnosis(
            diagnosis, confidence, disease_name, findings, user_query, context
        )

        # Xác định tác nhân gây bệnh
        pathogen_analysis = await self._identify_pathogen(
            diagnosis, disease_name, findings, user_query, context
        )

        return {
            "agent_id": self.agent_id,
            "status": "completed",
            "output": {
                "validation": validation_result,
                "pathogen_analysis": pathogen_analysis,
                "original_diagnosis": diagnosis,
                "original_confidence": confidence,
                "validated_confidence": validation_result.get("final_confidence", confidence),
            },
            "next_agents": ["agent4"],  # Chuyển đến Agent 4
        }

    async def _validate_diagnosis(
        self,
        diagnosis: str,
        confidence: float,
        disease_name: str,
        findings: list,
        user_query: str,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Thẩm định độ tin cậy của chẩn đoán"""
        if not self.client:
            return self._simple_validation(diagnosis, confidence, disease_name, findings)

        try:
            prompt = f"""
            Bạn là chuyên gia bệnh học thực vật. Hãy thẩm định chẩn đoán bệnh cây trồng sau:

            Câu hỏi của người dùng: {user_query}
            Ngữ cảnh: {context}

            Chẩn đoán từ Agent 2 (Vision/YOLO):
            - Tên bệnh: {disease_name}
            - Mô tả: {diagnosis[:500]}
            - Độ tin cậy ban đầu: {confidence:.2%}
            - Các phát hiện: {', '.join(findings[:5])}

            Hãy đánh giá:

            1. ĐỘ TIN CẬY CỦA CHẨN ĐOÁN:
               - Đánh giá độ chính xác của chẩn đoán dựa trên:
                 + Mức độ rõ ràng của triệu chứng
                 + Sự phù hợp giữa triệu chứng và tên bệnh
                 + Độ tin cậy từ model (YOLO/Vision)
                 + Tính nhất quán của các phát hiện
               - Đưa ra điểm số từ 0.0 đến 1.0
               - Giải thích lý do

            2. CẢNH BÁO NGUY CƠ NHẦM LẪN:
               - Các bệnh có triệu chứng tương tự có thể gây nhầm lẫn
               - Điều kiện có thể dẫn đến chẩn đoán sai
               - Các yếu tố cần xem xét thêm để xác nhận

            3. ĐÁNH GIÁ MỨC ĐỘ NGHIÊM TRỌNG:
               - Mức độ nghiêm trọng của bệnh (nhẹ/trung bình/nặng)
               - Tốc độ lây lan (nếu có)
               - Ảnh hưởng đến năng suất
               - Mức độ cấp thiết cần xử lý

            4. KHUYẾN NGHỊ XÁC NHẬN:
               - Có cần thêm thông tin để xác nhận không?
               - Các triệu chứng bổ sung cần quan sát
               - Có nên tham khảo chuyên gia không?
            """

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "Bạn là một chuyên gia bệnh học thực vật với nhiều năm kinh nghiệm trong việc thẩm định và xác nhận chẩn đoán bệnh cây trồng. Bạn có khả năng đánh giá độ tin cậy của chẩn đoán và cảnh báo về các nguy cơ nhầm lẫn. QUAN TRỌNG: Bạn PHẢI chỉ sử dụng tiếng Việt trong mọi phản hồi. Không được sử dụng tiếng Anh hoặc bất kỳ ngôn ngữ nào khác.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
            )

            validation_text = response.choices[0].message.content

            # Trích xuất thông tin có cấu trúc
            structured_validation = self._extract_validation_info(validation_text, confidence)

            return structured_validation

        except Exception as e:
            print(f"Error in diagnosis validation: {e}")
            return self._simple_validation(diagnosis, confidence, disease_name, findings)

    async def _identify_pathogen(
        self,
        diagnosis: str,
        disease_name: str,
        findings: list,
        user_query: str,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Xác định tác nhân gây bệnh (nấm, vi khuẩn, virus, dinh dưỡng, môi trường)"""
        if not self.client:
            return self._simple_pathogen_identification(diagnosis, disease_name, findings)

        try:
            prompt = f"""
            Bạn là chuyên gia bệnh học thực vật. Hãy xác định tác nhân gây bệnh dựa trên chẩn đoán sau:

            QUAN TRỌNG: Bạn PHẢI chỉ sử dụng tiếng Việt trong mọi phản hồi. Không được sử dụng tiếng Anh hoặc bất kỳ ngôn ngữ nào khác.

            QUAN TRỌNG: Bạn PHẢI chỉ sử dụng tiếng Việt trong mọi phản hồi. Không được sử dụng tiếng Anh hoặc bất kỳ ngôn ngữ nào khác.

            Câu hỏi của người dùng: {user_query}
            Ngữ cảnh: {context}

            Chẩn đoán:
            - Tên bệnh: {disease_name}
            - Mô tả: {diagnosis[:500]}
            - Các phát hiện: {', '.join(findings[:5])}

            Hãy xác định:

            1. TÁC NHÂN GÂY BỆNH CHÍNH:
               - Nấm (Fungi): Nếu bệnh do nấm gây ra
               - Vi khuẩn (Bacteria): Nếu bệnh do vi khuẩn
               - Virus: Nếu bệnh do virus
               - Thiếu dinh dưỡng: Nếu do thiếu chất dinh dưỡng
               - Yếu tố môi trường: Nếu do điều kiện môi trường (nhiệt độ, độ ẩm, ánh sáng, v.v.)
               - Sâu bệnh/côn trùng: Nếu do sâu bệnh hoặc côn trùng
               - Kết hợp nhiều yếu tố: Nếu do nhiều tác nhân

            2. ĐẶC ĐIỂM CỦA TÁC NHÂN:
               - Tên khoa học (nếu biết)
               - Điều kiện phát triển
               - Cách thức lây lan
               - Thời gian ủ bệnh

            3. DẤU HIỆU NHẬN BIẾT:
               - Triệu chứng đặc trưng của tác nhân này
               - Cách phân biệt với các tác nhân khác
               - Các dấu hiệu quan trọng cần chú ý

            4. ĐỘ TIN CẬY XÁC ĐỊNH:
               - Mức độ chắc chắn về tác nhân (0.0 - 1.0)
               - Lý do xác định tác nhân này
               - Các tác nhân khác có thể (nếu không chắc chắn)
            """

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "Bạn là một chuyên gia bệnh học thực vật với kiến thức sâu về các tác nhân gây bệnh cây trồng (nấm, vi khuẩn, virus, dinh dưỡng, môi trường). Bạn có khả năng xác định chính xác tác nhân gây bệnh dựa trên triệu chứng và chẩn đoán.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
            )

            pathogen_text = response.choices[0].message.content

            # Trích xuất thông tin có cấu trúc
            structured_pathogen = self._extract_pathogen_info(
                pathogen_text, diagnosis, disease_name
            )

            return structured_pathogen

        except Exception as e:
            print(f"Error in pathogen identification: {e}")
            return self._simple_pathogen_identification(diagnosis, disease_name, findings)

    def _simple_validation(
        self, diagnosis: str, confidence: float, disease_name: str, findings: list
    ) -> Dict[str, Any]:
        """Thẩm định đơn giản không dùng LLM"""
        # Đánh giá cơ bản dựa trên confidence và số lượng findings
        final_confidence = confidence
        if len(findings) > 3:
            final_confidence = min(confidence + 0.1, 1.0)
        if len(findings) < 2:
            final_confidence = max(confidence - 0.1, 0.0)

        warnings = []
        if confidence < 0.5:
            warnings.append("Độ tin cậy chẩn đoán thấp, cần xem xét thêm")
        if not disease_name or disease_name == "Unknown":
            warnings.append("Chưa xác định được tên bệnh cụ thể")

        return {
            "final_confidence": final_confidence,
            "confidence_change": final_confidence - confidence,
            "warnings": warnings,
            "severity": "moderate",
            "needs_confirmation": confidence < 0.6,
        }

    def _simple_pathogen_identification(
        self, diagnosis: str, disease_name: str, findings: list
    ) -> Dict[str, Any]:
        """Xác định tác nhân đơn giản không dùng LLM"""
        # Phân tích cơ bản dựa trên từ khóa
        diagnosis_lower = diagnosis.lower()
        disease_lower = disease_name.lower()

        pathogen_type = "unknown"
        confidence = 0.5

        # Từ khóa cho nấm
        if any(
            word in diagnosis_lower or word in disease_lower
            for word in ["nấm", "mốc", "phấn", "fungus", "mildew"]
        ):
            pathogen_type = "fungi"
            confidence = 0.7

        # Từ khóa cho vi khuẩn
        elif any(
            word in diagnosis_lower or word in disease_lower
            for word in ["vi khuẩn", "bacteria", "bacterial", "thối nhũn"]
        ):
            pathogen_type = "bacteria"
            confidence = 0.7

        # Từ khóa cho virus
        elif any(
            word in diagnosis_lower or word in disease_lower
            for word in ["virus", "vi rút", "mosaic"]
        ):
            pathogen_type = "virus"
            confidence = 0.7

        # Từ khóa cho dinh dưỡng
        elif any(
            word in diagnosis_lower or word in disease_lower
            for word in ["thiếu", "dinh dưỡng", "nutrient", "deficiency", "vàng lá"]
        ):
            pathogen_type = "nutrition"
            confidence = 0.6

        # Từ khóa cho môi trường
        elif any(
            word in diagnosis_lower or word in disease_lower
            for word in ["nhiệt độ", "độ ẩm", "ánh sáng", "temperature", "humidity", "stress"]
        ):
            pathogen_type = "environment"
            confidence = 0.6

        return {
            "pathogen_type": pathogen_type,
            "pathogen_name": disease_name,
            "confidence": confidence,
            "characteristics": [],
            "identification_signs": findings[:3],
        }

    def _extract_validation_info(
        self, validation_text: str, original_confidence: float
    ) -> Dict[str, Any]:
        """Trích xuất thông tin thẩm định có cấu trúc"""
        final_confidence = original_confidence
        warnings = []
        severity = "moderate"
        needs_confirmation = False

        text_lower = validation_text.lower()

        # Tìm độ tin cậy mới
        if "độ tin cậy" in text_lower or "confidence" in text_lower:
            # Tìm số từ 0.0 đến 1.0
            import re

            confidence_matches = re.findall(r"0?\.\d+|1\.0|\d+%", validation_text)
            if confidence_matches:
                try:
                    conf_str = confidence_matches[0].replace("%", "")
                    final_confidence = float(conf_str)
                    if final_confidence > 1.0:
                        final_confidence = final_confidence / 100.0
                except:
                    pass

        # Tìm cảnh báo
        if "cảnh báo" in text_lower or "warning" in text_lower or "nhầm lẫn" in text_lower:
            sentences = validation_text.split(".")
            for sentence in sentences:
                if any(
                    word in sentence.lower()
                    for word in ["cảnh báo", "warning", "nhầm lẫn", "nguy cơ"]
                ):
                    if len(sentence.strip()) > 20:
                        warnings.append(sentence.strip())

        # Xác định mức độ nghiêm trọng
        if any(word in text_lower for word in ["nặng", "nghiêm trọng", "severe", "critical"]):
            severity = "severe"
        elif any(word in text_lower for word in ["nhẹ", "mild", "light"]):
            severity = "mild"

        # Kiểm tra cần xác nhận
        if any(
            word in text_lower
            for word in ["cần xác nhận", "cần thêm", "needs confirmation", "verify"]
        ):
            needs_confirmation = True

        return {
            "validation_text": validation_text,
            "final_confidence": min(max(final_confidence, 0.0), 1.0),
            "confidence_change": final_confidence - original_confidence,
            "warnings": warnings[:3],
            "severity": severity,
            "needs_confirmation": needs_confirmation,
            "recommendations": self._extract_recommendations(validation_text),
        }

    def _extract_pathogen_info(
        self, pathogen_text: str, diagnosis: str, disease_name: str
    ) -> Dict[str, Any]:
        """Trích xuất thông tin tác nhân có cấu trúc"""
        pathogen_type = "unknown"
        pathogen_name = disease_name
        confidence = 0.5
        characteristics = []
        identification_signs = []

        text_lower = pathogen_text.lower()

        # Xác định loại tác nhân
        if "nấm" in text_lower or "fungi" in text_lower or "fungus" in text_lower:
            pathogen_type = "fungi"
        elif "vi khuẩn" in text_lower or "bacteria" in text_lower or "bacterial" in text_lower:
            pathogen_type = "bacteria"
        elif "virus" in text_lower or "vi rút" in text_lower:
            pathogen_type = "virus"
        elif "dinh dưỡng" in text_lower or "nutrition" in text_lower or "thiếu" in text_lower:
            pathogen_type = "nutrition"
        elif "môi trường" in text_lower or "environment" in text_lower:
            pathogen_type = "environment"
        elif "sâu" in text_lower or "côn trùng" in text_lower or "insect" in text_lower:
            pathogen_type = "insect"

        # Trích xuất đặc điểm
        sentences = pathogen_text.split(".")
        for sentence in sentences:
            if any(
                word in sentence.lower()
                for word in ["đặc điểm", "characteristic", "điều kiện", "condition", "phát triển"]
            ):
                if len(sentence.strip()) > 20:
                    characteristics.append(sentence.strip())

        # Trích xuất dấu hiệu nhận biết
        for sentence in sentences:
            if any(
                word in sentence.lower()
                for word in ["dấu hiệu", "sign", "triệu chứng", "symptom", "nhận biết", "identify"]
            ):
                if len(sentence.strip()) > 20:
                    identification_signs.append(sentence.strip())

        # Tìm độ tin cậy
        import re

        confidence_matches = re.findall(r"0?\.\d+|1\.0|\d+%", pathogen_text)
        if confidence_matches:
            try:
                conf_str = confidence_matches[0].replace("%", "")
                confidence = float(conf_str)
                if confidence > 1.0:
                    confidence = confidence / 100.0
            except:
                pass

        return {
            "pathogen_type": pathogen_type,
            "pathogen_name": pathogen_name,
            "confidence": min(max(confidence, 0.0), 1.0),
            "characteristics": characteristics[:5],
            "identification_signs": identification_signs[:5],
            "full_analysis": pathogen_text,
        }

    def _extract_recommendations(self, text: str) -> list:
        """Trích xuất khuyến nghị từ text"""
        recommendations = []
        sentences = text.split(".")
        for sentence in sentences:
            if any(
                word in sentence.lower()
                for word in ["nên", "khuyến nghị", "recommend", "cần", "nên làm", "should"]
            ):
                if len(sentence.strip()) > 20:
                    recommendations.append(sentence.strip())
        return recommendations[:3]
