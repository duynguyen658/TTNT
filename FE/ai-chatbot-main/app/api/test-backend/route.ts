import { NextResponse } from 'next/server';

const PYTHON_API_URL = process.env.PYTHON_API_URL || 'http://localhost:8000';

export async function GET() {
  try {
    const info = {
      PYTHON_API_URL,
      timestamp: new Date().toISOString(),
      tests: [] as Array<{ name: string; status: 'ok' | 'error'; message: string }>,
    };

    // Test 1: Health check
    try {
      const healthResponse = await fetch(`${PYTHON_API_URL}/health`, {
        cache: 'no-store',
        signal: AbortSignal.timeout(10000), // 10s timeout
      });

      if (healthResponse.ok) {
        const healthData = await healthResponse.json();
        info.tests.push({
          name: 'Health Check',
          status: 'ok',
          message: `Backend is healthy. Agents: ${Object.keys(healthData.agents || {}).length}`,
        });
      } else {
        info.tests.push({
          name: 'Health Check',
          status: 'error',
          message: `Backend returned status ${healthResponse.status}`,
        });
      }
    } catch (error: any) {
      info.tests.push({
        name: 'Health Check',
        status: 'error',
        message: error.message || 'Failed to connect to backend',
      });
    }

    // Test 2: Simple chat request
    try {
      const chatResponse = await fetch(`${PYTHON_API_URL}/api/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          user_query: 'test',
          user_context: {},
        }),
        signal: AbortSignal.timeout(30000), // 30s timeout
      });

      if (chatResponse.ok) {
        info.tests.push({
          name: 'Chat Endpoint',
          status: 'ok',
          message: 'Chat endpoint is working',
        });
      } else {
        const errorText = await chatResponse.text();
        info.tests.push({
          name: 'Chat Endpoint',
          status: 'error',
          message: `Chat endpoint returned status ${chatResponse.status}: ${errorText.substring(0, 100)}`,
        });
      }
    } catch (error: any) {
      info.tests.push({
        name: 'Chat Endpoint',
        status: 'error',
        message: error.message || 'Failed to call chat endpoint',
      });
    }

    const allPassed = info.tests.every(test => test.status === 'ok');

    return NextResponse.json(
      {
        ...info,
        overall: allPassed ? 'ok' : 'error',
        message: allPassed
          ? 'All backend tests passed!'
          : 'Some backend tests failed. Check details above.',
      },
      { status: allPassed ? 200 : 503 }
    );
  } catch (error: any) {
    return NextResponse.json(
      {
        PYTHON_API_URL,
        error: error.message || 'Unknown error',
        overall: 'error',
      },
      { status: 500 }
    );
  }
}
