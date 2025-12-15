import { NextRequest, NextResponse } from 'next/server';

const PYTHON_API_URL = process.env.PYTHON_API_URL || 'http://localhost:8000';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { user_query, user_context, image_data, image_url } = body;

    // Nếu có image_url, fetch và convert sang base64
    let final_image_data = image_data;
    if (image_url && !image_data) {
      try {
        const imageResponse = await fetch(image_url);
        if (imageResponse.ok) {
          const imageBuffer = await imageResponse.arrayBuffer();
          final_image_data = Buffer.from(imageBuffer).toString('base64');
        }
      } catch (error) {
        console.error('Error fetching image:', error);
      }
    }

    // Gọi Python API với timeout
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 60000); // 60 seconds timeout

    let response: Response;
    try {
      response = await fetch(`${PYTHON_API_URL}/api/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          user_query,
          user_context: user_context || {},
          image_data: final_image_data,
        }),
        signal: controller.signal,
      });
      clearTimeout(timeoutId);
    } catch (fetchError: any) {
      clearTimeout(timeoutId);

      if (fetchError.name === 'AbortError') {
        throw new Error(
          'Backend không phản hồi sau 60 giây. Vui lòng kiểm tra backend có đang chạy không.'
        );
      }

      if (
        fetchError.message?.includes('fetch failed') ||
        fetchError.message?.includes('ECONNREFUSED') ||
        fetchError.message?.includes('Failed to fetch') ||
        fetchError.message?.includes('NetworkError')
      ) {
        throw new Error(
          `Không thể kết nối đến backend tại ${PYTHON_API_URL}. ` +
            'Vui lòng đảm bảo backend Python đang chạy và Cloudflare Tunnel đang hoạt động.'
        );
      }

      throw new Error(`Lỗi khi gọi backend: ${fetchError.message || 'Unknown error'}`);
    }

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Python API error: ${response.status} - ${errorText}`);
    }

    const data = await response.json();

    return NextResponse.json({
      status: 'success',
      result: data.result,
    });
  } catch (error: any) {
    console.error('[plant-ai] Error calling Python API:', error);
    console.error('[plant-ai] PYTHON_API_URL:', PYTHON_API_URL);
    console.error('[plant-ai] Error details:', {
      message: error.message,
      name: error.name,
      stack: error.stack,
    });

    // Trả về 503 nếu là lỗi connection (backend không chạy)
    const isConnectionError =
      error.message?.includes('fetch failed') ||
      error.message?.includes('ECONNREFUSED') ||
      error.message?.includes('Failed to fetch') ||
      error.message?.includes('NetworkError') ||
      error.message?.includes('không thể kết nối') ||
      error.message?.includes('Backend không phản hồi');

    const statusCode = isConnectionError ? 503 : 500;

    return NextResponse.json(
      {
        error: error.message || 'Failed to call Python API',
        status: 'error',
        details: isConnectionError
          ? 'Backend không thể kết nối được. Vui lòng kiểm tra backend có đang chạy và Cloudflare Tunnel có hoạt động không.'
          : undefined,
      },
      { status: statusCode }
    );
  }
}

export async function GET() {
  // Health check
  try {
    const response = await fetch(`${PYTHON_API_URL}/health`, {
      cache: 'no-store',
    });

    if (!response.ok) {
      return NextResponse.json(
        { status: 'unhealthy', error: 'Python API not responding' },
        { status: 503 }
      );
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error: any) {
    return NextResponse.json({ status: 'unhealthy', error: error.message }, { status: 503 });
  }
}
