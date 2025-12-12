"""
Configuration file cho Multi-Agent System
"""
import os
from typing import Any, Dict

# LLM Provider: "openai" hoặc "groq"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()

# OpenAI API Key (nếu dùng OpenAI)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Groq API Key (nếu dùng Groq)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# YOLO Model Path
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "models/yolo_detection_s.pt")

# Agent Configuration
# Models sẽ được map dựa trên LLM_PROVIDER
AGENT_CONFIG: Dict[str, Dict[str, Any]] = {
    "agent1": {
        "name": "User Information Collector",
        "model": "gpt-4" if LLM_PROVIDER == "openai" else "llama-3.3-70b-versatile",  # Groq model
        "temperature": 0.3,
        "description": "Thu thập và xử lý thông tin từ người dùng",
    },
    "agent2": {
        "name": "Image Diagnosis Agent",
        "model": "gpt-4-vision-preview"
        if LLM_PROVIDER == "openai"
        else "llama-3.1-70b-versatile",  # Groq không có vision, dùng text model
        "temperature": 0.2,
        "description": "Chẩn đoán bệnh cây trồng dựa trên hình ảnh",
    },
    "agent3": {
        "name": "Dataset Diagnosis Agent",
        "model": "gpt-4" if LLM_PROVIDER == "openai" else "llama-3.3-70b-versatile",
        "temperature": 0.3,
        "description": "Phân tích dataset về bệnh cây trồng",
    },
    "agent4": {
        "name": "Social Media Search Agent",
        "model": "gpt-4" if LLM_PROVIDER == "openai" else "llama-3.3-70b-versatile",
        "temperature": 0.5,
        "description": "Tìm kiếm thông tin từ mạng xã hội",
    },
    "agent5": {
        "name": "Final Synthesis Agent",
        "model": "gpt-4" if LLM_PROVIDER == "openai" else "llama-3.3-70b-versatile",
        "temperature": 0.4,
        "description": "Tổng hợp và đưa ra tư vấn cuối cùng",
    },
}

# Default values nếu không có trong env
DEFAULT_OPENAI_MODEL = "gpt-4"
DEFAULT_TEMPERATURE = 0.5
