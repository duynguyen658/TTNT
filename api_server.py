"""
FastAPI server để expose Python AI models cho Next.js frontend
"""

import asyncio
import base64
import json
import os
from typing import Any, Dict, Optional

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

# Load environment variables from .env file
load_dotenv()

# Import orchestrator after load_dotenv to ensure env vars are loaded
from orchestrator import AgentOrchestrator  # noqa: E402

app = FastAPI(title="Plant Disease AI API", version="1.0.0")

# CORS middleware để cho phép frontend gọi
# Cho phép tất cả origins trong development, hoặc set CORS_ORIGINS env var trong production
# Hỗ trợ wildcard: https://*.vercel.app, https://*.ngrok-free.app
cors_origins_env = os.getenv("CORS_ORIGINS", "")
if cors_origins_env:
    cors_origins = [origin.strip() for origin in cors_origins_env.split(",")]
else:
    cors_origins = ["*"]  # Cho phép tất cả trong development

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Khởi tạo orchestrator (lazy load để tránh lỗi khi import)
_orchestrator: Optional[AgentOrchestrator] = None


def get_orchestrator() -> AgentOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = AgentOrchestrator()
    return _orchestrator


@app.get("/")
async def root():
    return {"message": "Plant Disease AI API", "status": "running", "version": "1.0.0"}


@app.get("/health")
async def health():
    try:
        orchestrator = get_orchestrator()
        status = orchestrator.get_agent_status()
        return {"status": "healthy", "agents": status}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


class ChatRequest(BaseModel):
    user_query: str
    user_context: Optional[Dict[str, Any]] = None
    image_data: Optional[str] = None  # base64 encoded image


@app.post("/api/chat")
async def chat_endpoint(request: Request):
    """
    Endpoint chính để xử lý chat với AI

    Args:
        user_query: Câu hỏi của người dùng
        user_context: Context bổ sung (plant_type, location, etc.)
        image_data: Base64 encoded image (nếu có)
    """
    try:
        # Parse request body manually để debug
        try:
            body = await request.json()
            print("\n" + "=" * 60)
            print("🌱 BACKEND: Nhận được request từ frontend")
            print("=" * 60)
            print(f"📦 Raw request body: {json.dumps(body, indent=2, ensure_ascii=False)[:500]}")
        except Exception as e:
            print(f"❌ Error parsing request body: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid JSON: {str(e)}")

        # Validate với Pydantic
        try:
            chat_request = ChatRequest(**body)
        except Exception as e:
            print(f"❌ Pydantic validation error: {e}")
            print(
                f"   Received body keys: {list(body.keys()) if isinstance(body, dict) else 'Not a dict'}"
            )
            raise HTTPException(status_code=422, detail=f"Validation error: {str(e)}")

        # Sử dụng validated request
        request_data = chat_request
        print(f"📝 User query: {request_data.user_query}")
        print(f"📦 User context: {request_data.user_context}")
        print(f"🖼️  Có image_data: {bool(request_data.image_data)}")
        if request_data.image_data:
            print(f"   Image size: {len(request_data.image_data)} bytes (base64)")

        orchestrator = get_orchestrator()

        # Xử lý image nếu có - KHÔNG lưu ảnh gốc, chỉ truyền image_data
        # Agent 2 sẽ tự lưu ảnh đã xử lý (có bounding box)
        user_input: Dict[str, Any] = {
            "user_query": request_data.user_query,
            "user_context": request_data.user_context or {},
        }

        if request_data.image_data:
            # Chỉ truyền image_data, không lưu ảnh gốc
            user_input["image_data"] = request_data.image_data
            print("🖼️  Đã nhận image_data, sẽ được xử lý bởi Agent 2")

        # Reset orchestrator execution log trước mỗi request
        orchestrator.reset()

        print("\n🚀 Bắt đầu chạy orchestrator với 5 agents...")
        # Gọi orchestrator
        try:
            result = await orchestrator.execute(user_input)
            print("✅ Orchestrator hoàn thành!")
            print("=" * 60 + "\n")
        except Exception as e:
            print(f"❌ Lỗi trong orchestrator.execute: {e}")
            import traceback

            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Orchestrator error: {str(e)}")

        # Không xóa image vì đã lưu vào folder image để tham khảo
        # if image_path and os.path.exists(image_path):
        #     try:
        #         os.unlink(image_path)
        #     except:
        #         pass

        return JSONResponse(
            {
                "status": "success",
                "result": result,
            }
        )

    except HTTPException:
        # Re-raise HTTPException để FastAPI xử lý đúng status code (422, 400, etc.)
        raise
    except Exception as e:
        print(f"\n❌ ERROR in chat_endpoint: {e}")
        import traceback

        traceback.print_exc()
        print("=" * 60 + "\n")

        # Trả về error response với thông tin chi tiết
        return JSONResponse(
            {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__,
            },
            status_code=500,
        )


@app.post("/api/chat/stream")
async def chat_stream_endpoint(
    user_query: str,
    user_context: Optional[Dict[str, Any]] = None,
    image_data: Optional[str] = None,
):
    """
    Streaming endpoint để trả về kết quả theo thời gian thực
    """

    async def generate():
        try:
            orchestrator = get_orchestrator()

            # Xử lý image
            image_path = None
            if image_data:
                try:
                    # Tạo folder image nếu chưa có
                    image_folder = "image"
                    os.makedirs(image_folder, exist_ok=True)

                    image_bytes = base64.b64decode(image_data)
                    import time

                    timestamp = int(time.time() * 1000)  # milliseconds
                    image_filename = f"analyzed_{timestamp}.jpg"
                    image_path = os.path.join(image_folder, image_filename)

                    with open(image_path, "wb") as f:
                        f.write(image_bytes)
                except Exception as e:
                    yield f"data: {json.dumps({'error': f'Image processing error: {e}'})}\n\n"
                    return

            user_input: Dict[str, Any] = {
                "user_query": user_query,
                "user_context": user_context or {},
            }

            if image_path:
                user_input["image_path"] = image_path

            # Stream results
            result = await orchestrator.execute(user_input)

            # Send final result
            final_advice = result.get("final_advice", {})
            yield f"data: {json.dumps({'type': 'result', 'data': final_advice})}\n\n"

            # Cleanup
            if image_path and os.path.exists(image_path):
                try:
                    os.unlink(image_path)
                except:
                    pass

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@app.post("/api/detect")
async def detect_disease(
    image_file: UploadFile = File(...),
    conf_threshold: float = 0.25,
):
    """
    Endpoint riêng cho YOLO detection
    """
    try:
        from yolo.inference_yolo import YOLOInference

        # Tạo folder image nếu chưa có
        image_folder = "image"
        os.makedirs(image_folder, exist_ok=True)

        # Lưu file vào folder image
        import time

        timestamp = int(time.time() * 1000)  # milliseconds
        image_filename = f"detected_{timestamp}.jpg"
        image_path = os.path.join(image_folder, image_filename)

        content = await image_file.read()
        with open(image_path, "wb") as f:
            f.write(content)

        # Chạy YOLO
        model_path = r"D:/TTNT2/models/yolo_detection_s.pt"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        yolo = YOLOInference(model_path, conf_threshold=conf_threshold)
        results = yolo.predict_single(image_path, show=False)

        # Cleanup
        if os.path.exists(image_path):
            try:
                os.unlink(image_path)
            except:
                pass

        return JSONResponse(
            {
                "status": "success",
                "detections": results,
            }
        )

    except Exception as e:
        print(f"Error in detect_disease: {e}")
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="localhost", port=port)
