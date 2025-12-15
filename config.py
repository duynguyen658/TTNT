"""
Configuration file cho Multi-Agent System
"""

import os
from typing import Any, Dict

# Groq API Key (cho Agent 1, 3, 4, 5)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# YOLO Model Path
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "models/yolo_detection_s.pt")

# Agent Configuration
AGENT_CONFIG: Dict[str, Dict[str, Any]] = {
    "agent1": {
        "name": "User Information Collector",
        "model": "llama-3.1-8b-instant",  # Groq model name
        "provider": "groq",  # Sử dụng Groq API
        "model_type": "text",
        "temperature": 0.3,
        "description": "Thu thập và xử lý thông tin từ người dùng",
    },
    "agent2": {
        "name": "Image Diagnosis Agent",
        "model": "yolo",  # YOLO model đã train sẵn
        "provider": "local",  # Local YOLO model
        "model_type": "yolo",
        "temperature": 0.2,
        "description": "Chẩn đoán bệnh cây trồng dựa trên hình ảnh",
    },
    "agent3": {
        "name": "Diagnosis Validator Agent",
        "model": "llama-3.3-70b-versatile",  # Groq model name
        "provider": "groq",  # Sử dụng Groq API
        "model_type": "text",
        "temperature": 0.3,
        "description": "Thẩm định chẩn đoán & xác định tác nhân gây bệnh",
    },
    "agent4": {
        "name": "Knowledge & Experience Agent",
        "model": "llama-3.1-8b-instant",  # Groq model name
        "provider": "groq",  # Sử dụng Groq API
        "model_type": "text",
        "temperature": 0.5,
        "description": "Bổ sung kiến thức nông học & kinh nghiệm thực tế",
    },
    "agent5": {
        "name": "Final Synthesis Agent",
        "model": "llama-3.3-70b-versatile",  # Groq model name
        "provider": "groq",  # Sử dụng Groq API
        "model_type": "text",
        "temperature": 0.4,
        "description": "Tổng hợp và đưa ra tư vấn cuối cùng",
    },
}
