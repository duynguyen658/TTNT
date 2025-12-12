# Hướng Dẫn Tích Hợp Mô Hình AI Với Frontend trên Vercel

## 📋 Tổng Quan

Bạn có:

- **Backend Python**: YOLO model + Multi-Agent System (orchestrator)
- **Frontend Next.js**: Đang dùng Vercel AI Gateway với xAI models
- **Mục tiêu**: Kết nối Python backend với Next.js frontend trên Vercel

## 🎯 Có 3 Cách Tiếp Cận

### Cách 1: Python API Server (Khuyến nghị cho Production)

- Deploy Python backend riêng (Railway, Render, AWS Lambda, etc.)
- Next.js gọi API qua HTTP
- ✅ Tách biệt, dễ scale
- ✅ Có thể dùng GPU cho YOLO

### Cách 2: Vercel Serverless Functions (Python)

- Deploy Python code trực tiếp trên Vercel
- ✅ Tất cả trong một project
- ⚠️ Giới hạn về thời gian chạy và dependencies

### Cách 3: Custom AI Provider trong Next.js

- Tạo custom provider gọi Python backend
- ✅ Tích hợp sâu với AI SDK
- ✅ Streaming support

---

## 🚀 Cách 1: Python API Server (Khuyến nghị)

### Bước 1: Tạo Python API Server

Tạo file `api_server.py` trong thư mục gốc:

```python
"""
FastAPI server để expose Python AI models
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import asyncio
from typing import Optional
import base64
from io import BytesIO
from PIL import Image

from orchestrator import AgentOrchestrator

app = FastAPI(title="Plant Disease AI API")

# CORS middleware để cho phép frontend gọi
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Trong production, chỉ định domain cụ thể
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Khởi tạo orchestrator
orchestrator = AgentOrchestrator()

@app.get("/")
async def root():
    return {"message": "Plant Disease AI API", "status": "running"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/api/chat")
async def chat_endpoint(
    user_query: str,
    user_context: Optional[dict] = None,
    image_data: Optional[str] = None,  # base64 encoded image
    image_file: Optional[UploadFile] = None,
):
    """
    Endpoint chính để xử lý chat với AI

    Args:
        user_query: Câu hỏi của người dùng
        user_context: Context bổ sung (plant_type, location, etc.)
        image_data: Base64 encoded image (nếu có)
        image_file: Uploaded image file (nếu có)
    """
    try:
        # Xử lý image nếu có
        image_path = None
        if image_file:
            # Lưu file tạm
            import tempfile
            import os
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                tmp.write(await image_file.read())
                image_path = tmp.name
        elif image_data:
            # Decode base64 và lưu
            import tempfile
            image_bytes = base64.b64decode(image_data)
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                tmp.write(image_bytes)
                image_path = tmp.name

        # Chuẩn bị input
        user_input = {
            "user_query": user_query,
            "user_context": user_context or {},
        }

        if image_path:
            user_input["image_path"] = image_path

        # Gọi orchestrator
        result = await orchestrator.execute(user_input)

        # Cleanup
        if image_path and os.path.exists(image_path):
            os.unlink(image_path)

        return JSONResponse({
            "status": "success",
            "result": result,
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/detect")
async def detect_disease(
    image_file: UploadFile,
    conf_threshold: float = 0.25,
):
    """
    Endpoint riêng cho YOLO detection
    """
    try:
        from yolo.inference_yolo import YOLOInference

        # Lưu file tạm
        import tempfile
        import os
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
            tmp.write(await image_file.read())
            image_path = tmp.name

        # Chạy YOLO
        yolo = YOLOInference("models/yolo_detection_s.pt", conf_threshold=conf_threshold)
        results = yolo.predict_single(image_path)

        # Cleanup
        if os.path.exists(image_path):
            os.unlink(image_path)

        return JSONResponse({
            "status": "success",
            "detections": results,
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Bước 2: Tạo requirements cho API server

Tạo file `requirements-api.txt`:

```txt
fastapi==0.104.1
uvicorn[standard]==0.24.0
python-multipart==0.0.6
pydantic==2.5.0
```

### Bước 3: Tạo Next.js API Route để gọi Python backend

Tạo file `FE/ai-chatbot-main/app/(chat)/api/plant-ai/route.ts`:

```typescript
import { NextRequest, NextResponse } from 'next/server';

const PYTHON_API_URL = process.env.PYTHON_API_URL || 'http://localhost:8000';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { user_query, user_context, image_data } = body;

    // Gọi Python API
    const response = await fetch(`${PYTHON_API_URL}/api/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        user_query,
        user_context,
        image_data, // base64 encoded
      }),
    });

    if (!response.ok) {
      throw new Error(`Python API error: ${response.statusText}`);
    }

    const data = await response.json();

    return NextResponse.json({
      status: 'success',
      result: data.result,
    });
  } catch (error: any) {
    console.error('Error calling Python API:', error);
    return NextResponse.json({ error: error.message }, { status: 500 });
  }
}

export async function GET() {
  // Health check
  try {
    const response = await fetch(`${PYTHON_API_URL}/health`);
    const data = await response.json();
    return NextResponse.json(data);
  } catch (error: any) {
    return NextResponse.json({ status: 'unhealthy', error: error.message }, { status: 503 });
  }
}
```

### Bước 4: Tạo Custom Tool trong AI SDK

Tạo file `FE/ai-chatbot-main/lib/ai/tools/plant-diagnosis.ts`:

```typescript
import { tool } from 'ai';
import { z } from 'zod';

const PYTHON_API_URL = process.env.PYTHON_API_URL || 'http://localhost:8000';

export const plantDiagnosis = tool({
  description: `
    Chẩn đoán bệnh cây trồng dựa trên:
    - Câu hỏi của người dùng về triệu chứng
    - Hình ảnh cây bị bệnh (nếu có)
    - Context về loại cây, vị trí, mùa vụ

    Sử dụng YOLO model và multi-agent system để đưa ra:
    - Chẩn đoán bệnh
    - Khuyến nghị điều trị
    - Biện pháp phòng ngừa
  `,
  parameters: z.object({
    user_query: z.string().describe('Câu hỏi hoặc mô tả về vấn đề cây trồng'),
    plant_type: z.string().optional().describe('Loại cây (ví dụ: cà chua, lúa, v.v.)'),
    location: z.string().optional().describe('Vị trí (ví dụ: miền Bắc, miền Nam)'),
    season: z.string().optional().describe('Mùa vụ (ví dụ: mùa mưa, mùa khô)'),
    image_url: z.string().url().optional().describe('URL của hình ảnh cây bị bệnh'),
  }),
  execute: async ({ user_query, plant_type, location, season, image_url }) => {
    try {
      // Nếu có image_url, fetch và convert sang base64
      let image_data: string | undefined;
      if (image_url) {
        const imageResponse = await fetch(image_url);
        const imageBuffer = await imageResponse.arrayBuffer();
        image_data = Buffer.from(imageBuffer).toString('base64');
      }

      const response = await fetch(`${PYTHON_API_URL}/api/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          user_query,
          user_context: {
            plant_type,
            location,
            season,
          },
          image_data,
        }),
      });

      if (!response.ok) {
        throw new Error(`Python API error: ${response.statusText}`);
      }

      const data = await response.json();
      const result = data.result;

      // Format response cho AI
      const finalAdvice = result.final_advice || {};

      return {
        diagnosis: finalAdvice.diagnosis || 'Không xác định được bệnh',
        confidence: finalAdvice.confidence_score || 0,
        recommendations: finalAdvice.recommendations || [],
        treatment: finalAdvice.treatment_plan || 'Chưa có kế hoạch điều trị',
        prevention: finalAdvice.prevention_measures || [],
        full_advice: finalAdvice.full_advice || finalAdvice.summary || '',
      };
    } catch (error: any) {
      return {
        error: error.message,
        diagnosis: 'Lỗi khi chẩn đoán',
      };
    }
  },
});
```

### Bước 5: Thêm Tool vào Chat Route

Cập nhật `FE/ai-chatbot-main/app/(chat)/api/chat/route.ts`:

```typescript
// Thêm import
import { plantDiagnosis } from '@/lib/ai/tools/plant-diagnosis';

// Trong streamText, thêm tool:
tools: {
  getWeather,
  createDocument: createDocument({ session, dataStream }),
  updateDocument: updateDocument({ session, dataStream }),
  requestSuggestions: requestSuggestions({
    session,
    dataStream,
  }),
  plantDiagnosis, // ← Thêm dòng này
},
```

### Bước 6: Thêm Environment Variable

Thêm vào `.env.local`:

```env
# Python API URL
PYTHON_API_URL=http://localhost:8000  # Local development
# PYTHON_API_URL=https://your-python-api.railway.app  # Production
```

---

## 🚀 Cách 2: Vercel Serverless Functions (Python)

### Bước 1: Tạo Vercel Function

Tạo file `FE/ai-chatbot-main/api/plant-ai.py`:

```python
from http.server import BaseHTTPRequestHandler
import json
import sys
import os

# Add parent directory to path để import orchestrator
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from orchestrator import AgentOrchestrator

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))

            orchestrator = AgentOrchestrator()
            result = await orchestrator.execute({
                "user_query": data.get("user_query", ""),
                "user_context": data.get("user_context", {}),
            })

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())

        except Exception as e:
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())
```

### Bước 2: Cấu hình Vercel

Tạo file `vercel.json`:

```json
{
  "functions": {
    "api/plant-ai.py": {
      "runtime": "python3.9"
    }
  }
}
```

⚠️ **Lưu ý**: Vercel Python functions có giới hạn về:

- Thời gian chạy (10s cho Hobby, 60s cho Pro)
- Dependencies size
- Không hỗ trợ GPU

---

## 🚀 Cách 3: Custom AI Provider

Tạo custom provider để tích hợp sâu hơn với AI SDK:

Tạo file `FE/ai-chatbot-main/lib/ai/providers-custom.ts`:

```typescript
import { customProvider, languageModel } from 'ai';

const PYTHON_API_URL = process.env.PYTHON_API_URL || 'http://localhost:8000';

// Custom language model wrapper
export const plantDiseaseModel = languageModel({
  provider: 'custom',
  modelId: 'plant-disease-ai',
  doStream: async ({ prompt, messages }) => {
    // Gọi Python API
    const response = await fetch(`${PYTHON_API_URL}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        user_query: messages[messages.length - 1]?.content || prompt,
      }),
    });

    const data = await response.json();
    const advice = data.result?.final_advice?.full_advice || '';

    // Return streaming response
    return {
      stream: (async function* () {
        // Simulate streaming
        const words = advice.split(' ');
        for (const word of words) {
          yield { type: 'text-delta', textDelta: word + ' ' };
        }
      })(),
      rawCall: { rawPrompt: prompt, rawSettings: {} },
    };
  },
});

export const customPlantProvider = customProvider({
  languageModels: {
    'plant-disease-model': plantDiseaseModel,
  },
});
```

---

## 📦 Deploy Python Backend

### Option A: Railway (Khuyến nghị)

1. Tạo account tại [Railway.app](https://railway.app)
2. Tạo new project từ GitHub repo
3. Set build command: `pip install -r requirements-api.txt`
4. Set start command: `python api_server.py`
5. Add environment variables
6. Deploy!

### Option B: Render

1. Tạo account tại [Render.com](https://render.com)
2. Tạo new Web Service
3. Connect GitHub repo
4. Set:
   - Build: `pip install -r requirements-api.txt`
   - Start: `uvicorn api_server:app --host 0.0.0.0 --port $PORT`
5. Deploy!

### Option C: AWS Lambda + API Gateway

- Phức tạp hơn nhưng scalable
- Cần setup Lambda layers cho dependencies

---

## 🔧 Cấu Hình Frontend

### 1. Thêm Model vào Model Selector

Cập nhật `FE/ai-chatbot-main/lib/ai/models.ts`:

```typescript
export const chatModels = [
  // ... existing models
  {
    id: 'plant-disease-model',
    name: 'Plant Disease AI',
    description: 'Chẩn đoán bệnh cây trồng với YOLO + Multi-Agent',
  },
] as const;
```

### 2. Update Providers

Cập nhật `FE/ai-chatbot-main/lib/ai/providers.ts`:

```typescript
import { customPlantProvider } from './providers-custom';

export const myProvider = isTestEnvironment
  ? // ... existing
  : customProvider({
      languageModels: {
        // ... existing models
        'plant-disease-model': customPlantProvider.languageModel('plant-disease-model'),
      },
    });
```

---

## 🧪 Testing

### Test Python API locally:

```bash
# Terminal 1: Start Python API
cd D:\TTNT2
python api_server.py

# Terminal 2: Test API
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"user_query": "Cây cà chua bị vàng lá", "user_context": {"plant_type": "cà chua"}}'
```

### Test từ Frontend:

1. Start Next.js: `npm run dev`
2. Mở browser: http://localhost:3000
3. Chọn model "Plant Disease AI"
4. Gửi message: "Cây cà chua của tôi bị vàng lá"

---

## 📝 Environment Variables

### Frontend (.env.local):

```env
PYTHON_API_URL=http://localhost:8000  # Development
# PYTHON_API_URL=https://your-api.railway.app  # Production
```

### Python Backend:

```env
OPENAI_API_KEY=your-key-here
# Other env vars...
```

---

## 🎯 Next Steps

1. ✅ Tạo Python API server
2. ✅ Deploy Python backend (Railway/Render)
3. ✅ Tạo Next.js API route
4. ✅ Tạo custom tool/provider
5. ✅ Test integration
6. ✅ Deploy frontend lên Vercel
7. ✅ Set production PYTHON_API_URL

---

## 📚 Tài Liệu Tham Khảo

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Vercel Serverless Functions](https://vercel.com/docs/functions)
- [AI SDK Custom Providers](https://ai-sdk.dev/docs/guides/providers/custom-provider)
- [Railway Deployment](https://docs.railway.app/)
- [Render Deployment](https://render.com/docs)
