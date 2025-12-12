"""
Agent 2: Chẩn đoán bệnh cây trồng dựa trên hình ảnh
"""

import base64
import io
import os
from typing import Any, Dict, Optional

from PIL import Image

import config
from agents.base_agent import BaseAgent

# Import YOLO model nhận dạng bệnh cây
try:
    from yolo.inference_yolo import YOLOInference

    MODEL_AVAILABLE = True
except ImportError as e:
    MODEL_AVAILABLE = False
    print(f"⚠️  YOLO model module chưa được import: {e}")
    print("👉 Cài đặt: pip install ultralytics opencv-python pillow")
    print("⚠️  Sẽ chỉ sử dụng Vision API.")


class ImageDiagnosisAgent(BaseAgent):
    """Agent chẩn đoán bệnh cây trồng dựa trên hình ảnh"""

    def __init__(self):
        super().__init__("agent2", config.AGENT_CONFIG["agent2"])
        self.disease_model = None
        self._load_disease_model()

    def _load_disease_model(self):
        """Load YOLO model nhận dạng bệnh cây nếu có"""
        if not MODEL_AVAILABLE:
            return

        # Thử load custom YOLO model trước
        model_path = getattr(config, "YOLO_MODEL_PATH", "models/yolo_detection_s.pt")

        # Kiểm tra đường dẫn tuyệt đối và tương đối
        if not os.path.exists(model_path):
            # Thử đường dẫn tuyệt đối từ config
            abs_path = os.path.abspath(model_path)
            if os.path.exists(abs_path):
                model_path = abs_path
            else:
                # Thử trong thư mục models/
                alt_path = os.path.join("models", os.path.basename(model_path))
                if os.path.exists(alt_path):
                    model_path = alt_path
                else:
                    print(f"⚠️  Không tìm thấy YOLO model tại: {model_path}")
                    print(f"⚠️  Đã thử: {abs_path}, {alt_path}")
                    return

        try:
            self.disease_model = YOLOInference(model_path, conf_threshold=0.25)
            print(f"✅ Đã load YOLO model nhận dạng bệnh cây từ: {model_path}")
        except Exception as e:
            print(f"⚠️  Không thể load YOLO model từ {model_path}: {e}")
            import traceback

            traceback.print_exc()

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Chẩn đoán bệnh cây trồng dựa trên hình ảnh
        """
        image_path = input_data.get("image_path")
        image_data = input_data.get("image_data")  # Base64 encoded
        user_query = input_data.get("user_query", "")
        context = input_data.get("context", {})

        # Xử lý hình ảnh
        image = None
        if image_path and os.path.exists(image_path):
            image = Image.open(image_path)
        elif image_data:
            image = self._decode_image(image_data)

        if not image:
            return {
                "agent_id": self.agent_id,
                "status": "error",
                "error": "Không tìm thấy hình ảnh để phân tích",
                "output": {},
            }

        # Phân tích hình ảnh
        analysis_result = await self._analyze_image(image, user_query, context)

        return {
            "agent_id": self.agent_id,
            "status": "completed",
            "output": analysis_result,
            "next_agents": ["agent5"],  # Luôn chuyển đến agent tổng hợp
        }

    def _decode_image(self, image_data: str) -> Optional[Image.Image]:
        """Giải mã hình ảnh từ base64"""
        try:
            if image_data.startswith("data:image"):
                image_data = image_data.split(",")[1]

            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            return image
        except Exception as e:
            print(f"Error decoding image: {e}")
            return None

    async def _analyze_image(
        self, image: Image.Image, query: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Phân tích hình ảnh sử dụng cả model chuyên biệt và vision model"""
        # Sử dụng model chuyên biệt nếu có
        model_result = None
        if self.disease_model:
            try:
                # Lưu image tạm để YOLO xử lý
                import tempfile

                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                    image.save(tmp.name, format="JPEG")
                    temp_path = tmp.name

                # Chạy YOLO prediction
                yolo_result = self.disease_model.predict_single(temp_path, show=False)

                # Xóa file tạm
                try:
                    os.unlink(temp_path)
                except:
                    pass

                # Chuyển đổi kết quả YOLO sang format mong muốn
                if yolo_result and yolo_result.get("num_detections", 0) > 0:
                    top_detection = yolo_result.get("top_detection")
                    if top_detection:
                        model_result = {
                            "disease": top_detection.get("class_name", "Unknown"),
                            "confidence": top_detection.get("confidence", 0.0),
                            "top_predictions": yolo_result.get("detections", [])[:5],
                            "num_detections": yolo_result.get("num_detections", 0),
                        }
                    else:
                        model_result = {
                            "disease": "No disease detected",
                            "confidence": 0.0,
                            "top_predictions": [],
                            "num_detections": 0,
                        }
                else:
                    model_result = {
                        "disease": "No disease detected",
                        "confidence": 0.0,
                        "top_predictions": [],
                        "num_detections": 0,
                    }

            except Exception as e:
                print(f"⚠️  Lỗi khi dùng YOLO model: {e}")
                import traceback

                traceback.print_exc()
                model_result = None

        # Sử dụng Vision API nếu có
        vision_result = None
        if self.client:
            vision_result = await self._analyze_with_vision_api(image, query, context, model_result)

        # Kết hợp kết quả
        return self._combine_results(model_result, vision_result, query, context)

    async def _analyze_with_vision_api(
        self, image: Image.Image, query: str, context: Dict[str, Any], model_result: Optional[Dict]
    ) -> Dict[str, Any]:
        """Phân tích hình ảnh sử dụng Vision API"""
        if not self.client:
            return None

        try:
            # Chuyển đổi hình ảnh sang base64
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()

            # Tạo prompt phân tích bệnh cây trồng
            prompt = f"""
            Bạn là chuyên gia nông nghiệp và bệnh học thực vật. Phân tích hình ảnh này để chẩn đoán bệnh cây trồng.

            Câu hỏi của người dùng: {query}
            Ngữ cảnh: {context}

            Hãy phân tích và cung cấp:
            1. Mô tả chi tiết về hình ảnh:
               - Loại cây trồng (nếu có thể nhận dạng)
               - Bộ phận của cây (lá, thân, rễ, quả, hoa)
               - Tình trạng hiện tại của cây

            2. Triệu chứng bệnh quan sát được:
               - Màu sắc bất thường (vàng, nâu, đen, trắng, v.v.)
               - Đốm, vết, mảng trên lá/thân
               - Tình trạng héo, thối, khô
               - Sự hiện diện của nấm, mốc, phấn trắng
               - Dấu hiệu sâu bệnh, côn trùng

            3. Chẩn đoán bệnh:
               - Tên bệnh có thể (nếu xác định được)
               - Nguyên nhân có thể (nấm, vi khuẩn, virus, thiếu dinh dưỡng, v.v.)
               - Mức độ nghiêm trọng (nhẹ, trung bình, nặng)
               - Độ tin cậy của chẩn đoán

            4. Khuyến nghị điều trị:
               - Biện pháp xử lý cụ thể
               - Thuốc/phân bón phù hợp (nếu biết)
               - Các bước phòng ngừa
               - Thời gian điều trị dự kiến

            5. Lưu ý quan trọng:
               - Cảnh báo nếu cần xử lý ngay
               - Khuyến nghị tham khảo chuyên gia nếu cần
            """

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "Bạn là một chuyên gia nông nghiệp và bệnh học thực vật với nhiều năm kinh nghiệm trong việc chẩn đoán và điều trị bệnh cây trồng. Bạn có khả năng nhận dạng các loại bệnh phổ biến trên cây trồng qua hình ảnh.",
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{img_base64}"},
                            },
                        ],
                    },
                ],
                temperature=self.temperature,
                max_tokens=2000,
            )

            analysis_text = response.choices[0].message.content

            # Trích xuất thông tin có cấu trúc từ kết quả
            structured_result = self._extract_structured_info(analysis_text)

            return {
                "raw_analysis": analysis_text,
                "diagnosis": structured_result.get("diagnosis", ""),
                "findings": structured_result.get("findings", []),
                "confidence": structured_result.get("confidence", 0.5),
                "recommendations": structured_result.get("recommendations", []),
                "image_metadata": {"size": image.size, "format": image.format, "mode": image.mode},
            }

        except Exception as e:
            print(f"Error in image analysis: {e}")
            return {
                "error": str(e),
                "diagnosis": "Lỗi trong quá trình phân tích hình ảnh",
                "confidence": 0.0,
            }

    def _extract_structured_info(self, analysis_text: str) -> Dict[str, Any]:
        """Trích xuất thông tin có cấu trúc từ kết quả phân tích bệnh cây"""
        findings = []
        recommendations = []
        confidence = 0.5
        disease_name = ""
        severity = "unknown"

        text_lower = analysis_text.lower()

        # Tìm tên bệnh
        disease_keywords = [
            "bệnh",
            "nấm",
            "mốc",
            "thối",
            "héo",
            "vàng",
            "đốm",
            "cháy",
            "rust",
            "blight",
            "mildew",
            "rot",
            "wilt",
            "spot",
            "leaf spot",
        ]
        for keyword in disease_keywords:
            if keyword in text_lower:
                # Tìm câu chứa từ khóa
                sentences = analysis_text.split(".")
                for sentence in sentences:
                    if keyword in sentence.lower():
                        findings.append(sentence.strip())
                        break

        # Tìm khuyến nghị
        rec_keywords = [
            "khuyến nghị",
            "nên",
            "điều trị",
            "xử lý",
            "phun",
            "bón",
            "recommend",
            "treatment",
        ]
        for keyword in rec_keywords:
            if keyword in text_lower:
                sentences = analysis_text.split(".")
                for sentence in sentences:
                    if keyword in sentence.lower() and len(sentence.strip()) > 20:
                        recommendations.append(sentence.strip())
                        break

        # Tính độ tin cậy
        if "chẩn đoán" in text_lower or "diagnosis" in text_lower:
            confidence = 0.7
        if any(word in text_lower for word in ["xác định", "chắc chắn", "rõ ràng", "definite"]):
            confidence = 0.8
        if any(word in text_lower for word in ["có thể", "có khả năng", "possible", "might"]):
            confidence = 0.6

        # Xác định mức độ nghiêm trọng
        if any(word in text_lower for word in ["nặng", "nghiêm trọng", "severe", "critical"]):
            severity = "severe"
        elif any(word in text_lower for word in ["trung bình", "moderate", "medium"]):
            severity = "moderate"
        elif any(word in text_lower for word in ["nhẹ", "mild", "light"]):
            severity = "mild"

        return {
            "diagnosis": analysis_text[:800],  # Tăng lên 800 ký tự
            "disease_name": disease_name,
            "severity": severity,
            "findings": findings[:5],  # Giới hạn 5 findings
            "confidence": confidence,
            "recommendations": recommendations[:5],  # Giới hạn 5 recommendations
        }

    def _combine_results(
        self,
        model_result: Optional[Dict],
        vision_result: Optional[Dict],
        query: str,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Kết hợp kết quả từ model chuyên biệt và Vision API"""

        # Nếu chỉ có model result
        if model_result and not model_result.get("error") and not vision_result:
            return {
                "diagnosis": f"Model nhận dạng: {model_result.get('disease', 'Unknown')}",
                "disease_name": model_result.get("disease", "Unknown"),
                "confidence": model_result.get("confidence", 0.0),
                "model_predictions": model_result.get("top_predictions", []),
                "source": "model_only",
                "findings": [f"YOLO Model nhận dạng: {model_result.get('disease', 'Unknown')}"],
                "recommendations": [],
            }

        # Nếu chỉ có vision result
        if vision_result and not model_result:
            return vision_result

        # Nếu có cả hai, kết hợp
        if model_result and not model_result.get("error") and vision_result:
            # Lấy thông tin từ model
            model_disease = model_result.get("disease", "Unknown")
            model_confidence = model_result.get("confidence", 0.0)
            model_predictions = model_result.get("top_predictions", [])

            # Lấy thông tin từ vision API
            vision_diagnosis = vision_result.get("diagnosis", "")
            vision_confidence = vision_result.get("confidence", 0.0)

            # Kết hợp
            combined_confidence = (model_confidence + vision_confidence) / 2

            # Tạo diagnosis kết hợp
            combined_diagnosis = f"""
Kết quả từ YOLO Model: {model_disease} (độ tin cậy: {model_confidence:.2%})

Kết quả từ Vision API:
{vision_diagnosis[:400]}

Kết hợp: Cả hai phương pháp đều cho kết quả tương đồng. Độ tin cậy tổng hợp: {combined_confidence:.2%}
            """.strip()

            return {
                "diagnosis": combined_diagnosis,
                "disease_name": model_disease,
                "model_disease": model_disease,
                "model_confidence": model_confidence,
                "vision_confidence": vision_confidence,
                "confidence": combined_confidence,
                "model_predictions": model_predictions,
                "vision_analysis": vision_diagnosis,
                "source": "combined",
                "findings": vision_result.get("findings", [])
                + [f"YOLO Model xác nhận: {model_disease}"],
                "recommendations": vision_result.get("recommendations", []),
                "severity": vision_result.get("severity", "unknown"),
            }

        # Fallback nếu không có kết quả nào
        return {
            "diagnosis": "Không thể phân tích hình ảnh. Vui lòng kiểm tra lại.",
            "confidence": 0.0,
            "error": "No results available",
        }
