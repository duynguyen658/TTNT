"""
FastAPI server để expose Python AI models cho Next.js frontend
"""
import asyncio
import base64
import json
import os
import tempfile
from typing import Any, Dict, Optional

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from orchestrator import AgentOrchestrator

app = FastAPI(title="Plant Disease AI API", version="1.0.0")

# CORS middleware để cho phép frontend gọi
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://*.vercel.app",
        "https://*.vercel.sh",
    ],  # Trong production, chỉ định domain cụ thể
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


@app.post("/api/chat")
async def chat_endpoint(
    user_query: str,
    user_context: Optional[Dict[str, Any]] = None,
    image_data: Optional[str] = None,  # base64 encoded image
):
    """
    Endpoint chính để xử lý chat với AI

    Args:
        user_query: Câu hỏi của người dùng
        user_context: Context bổ sung (plant_type, location, etc.)
        image_data: Base64 encoded image (nếu có)
    """
    try:
        orchestrator = get_orchestrator()

        # Xử lý image nếu có
        image_path = None
        if image_data:
            try:
                # Decode base64 và lưu file tạm
                image_bytes = base64.b64decode(image_data)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                    tmp.write(image_bytes)
                    image_path = tmp.name
            except Exception as e:
                print(f"Error processing image: {e}")

        # Chuẩn bị input
        user_input: Dict[str, Any] = {
            "user_query": user_query,
            "user_context": user_context or {},
        }

        if image_path:
            user_input["image_path"] = image_path

        # Gọi orchestrator
        result = await orchestrator.execute(user_input)

        # Cleanup
        if image_path and os.path.exists(image_path):
            try:
                os.unlink(image_path)
            except:
                pass

        return JSONResponse(
            {
                "status": "success",
                "result": result,
            }
        )

    except Exception as e:
        print(f"Error in chat_endpoint: {e}")
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


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
                    image_bytes = base64.b64decode(image_data)
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                        tmp.write(image_bytes)
                        image_path = tmp.name
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

        # Lưu file tạm
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            content = await image_file.read()
            tmp.write(content)
            image_path = tmp.name

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
