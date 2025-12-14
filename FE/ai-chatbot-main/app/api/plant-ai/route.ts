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

    // Gọi Python API
    const response = await fetch(`${PYTHON_API_URL}/api/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        user_query,
        user_context: user_context || {},
        image_data: final_image_data,
      }),
    });

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
    console.error('Error calling Python API:', error);
    return NextResponse.json(
      {
        error: error.message || 'Failed to call Python API',
        status: 'error',
      },
      { status: 500 }
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
