import { geolocation } from '@vercel/functions';
import {
  convertToModelMessages,
  createUIMessageStream,
  JsonToSseTransformStream,
  smoothStream,
  stepCountIs,
  streamText,
} from 'ai';
import { unstable_cache as cache } from 'next/cache';
import { after } from 'next/server';
import { createResumableStreamContext, type ResumableStreamContext } from 'resumable-stream';
import type { ModelCatalog } from 'tokenlens/core';
import { fetchModels } from 'tokenlens/fetch';
import { getUsage } from 'tokenlens/helpers';
import { auth, type UserType } from '@/app/(auth)/auth';
import type { VisibilityType } from '@/components/visibility-selector';
import { entitlementsByUserType } from '@/lib/ai/entitlements';
import type { ChatModel } from '@/lib/ai/models';
import { type RequestHints, systemPrompt } from '@/lib/ai/prompts';
import { myProvider } from '@/lib/ai/providers';
import { createDocument } from '@/lib/ai/tools/create-document';
import { getWeather } from '@/lib/ai/tools/get-weather';
import { createPlantDiagnosisTool } from '@/lib/ai/tools/plant-diagnosis';
import { requestSuggestions } from '@/lib/ai/tools/request-suggestions';
import { updateDocument } from '@/lib/ai/tools/update-document';
import { isProductionEnvironment } from '@/lib/constants';
import {
  createStreamId,
  deleteChatById,
  getChatById,
  getMessageCountByUserId,
  getMessagesByChatId,
  saveChat,
  saveMessages,
  updateChatLastContextById,
} from '@/lib/db/queries';
import type { Chat, DBMessage } from '@/lib/db/schema';
import { ChatSDKError } from '@/lib/errors';
import type { ChatMessage } from '@/lib/types';
import type { AppUsage } from '@/lib/usage';
import { convertToUIMessages, generateUUID } from '@/lib/utils';
import { generateTitleFromUserMessage } from '../../actions';
import { type PostRequestBody, postRequestBodySchema } from './schema';

export const maxDuration = 60;

let globalStreamContext: ResumableStreamContext | null = null;

const getTokenlensCatalog = cache(
  async (): Promise<ModelCatalog | undefined> => {
    try {
      return await fetchModels();
    } catch (err) {
      console.warn('TokenLens: catalog fetch failed, using default catalog', err);
      return; // tokenlens helpers will fall back to defaultCatalog
    }
  },
  ['tokenlens-catalog'],
  { revalidate: 24 * 60 * 60 } // 24 hours
);

export function getStreamContext() {
  if (!globalStreamContext) {
    try {
      globalStreamContext = createResumableStreamContext({
        waitUntil: after,
      });
    } catch (error: any) {
      if (error.message.includes('REDIS_URL')) {
        console.log(' > Resumable streams are disabled due to missing REDIS_URL');
      } else {
        console.error(error);
      }
    }
  }

  return globalStreamContext;
}

export async function POST(request: Request) {
  let requestBody: PostRequestBody;

  try {
    const json = await request.json();
    requestBody = postRequestBodySchema.parse(json);
  } catch (error) {
    console.error('Request validation error:', error);
    if (error instanceof Error) {
      console.error('Error details:', error.message);
    }
    // Log request body for debugging (only in development)
    try {
      const body = await request.clone().json();
      console.error('Request body received:', JSON.stringify(body, null, 2));
    } catch (e) {
      console.error('Could not parse request body for logging');
    }
    return new ChatSDKError('bad_request:api').toResponse();
  }

  try {
    const {
      id,
      message,
      selectedChatModel,
      selectedVisibilityType,
    }: {
      id: string;
      message: ChatMessage;
      selectedChatModel: ChatModel['id'];
      selectedVisibilityType: VisibilityType;
    } = requestBody;

    const session = await auth();

    if (!session?.user) {
      return new ChatSDKError('unauthorized:chat').toResponse();
    }

    const userType: UserType = session.user.type;

    let messageCount = 0;
    try {
      messageCount = await getMessageCountByUserId({
        id: session.user.id,
        differenceInHours: 24,
      });
    } catch (dbError: any) {
      console.error('[chat] Error getting message count:', dbError);
      // Continue if database error, don't block the request
    }

    if (messageCount > entitlementsByUserType[userType].maxMessagesPerDay) {
      return new ChatSDKError('rate_limit:chat').toResponse();
    }

    let chat: Chat | null = null;
    let messagesFromDb: DBMessage[] = [];
    try {
      chat = await getChatById({ id });
    } catch (dbError: any) {
      console.error('[chat] Error getting chat:', dbError);
      // Continue if database error, create new chat if needed
      chat = null;
    }

    if (chat) {
      if (chat.userId !== session.user.id) {
        return new ChatSDKError('forbidden:chat').toResponse();
      }
      // Only fetch messages if chat already exists
      try {
        messagesFromDb = await getMessagesByChatId({ id });
      } catch (dbError: any) {
        console.error('[chat] Error getting messages:', dbError);
        // Continue with empty messages if database error
        messagesFromDb = [];
      }
    } else {
      try {
        const title = await generateTitleFromUserMessage({
          message,
        });

        await saveChat({
          id,
          userId: session.user.id,
          title,
          visibility: selectedVisibilityType,
        });
        // New chat - no need to fetch messages, it's empty
      } catch (dbError: any) {
        console.error('[chat] Error saving chat:', dbError);
        // Continue even if saveChat fails, chat will be created on first message save
      }
    }

    // Không giới hạn messages - để model tự xử lý với context window lớn
    // Model hiện tại (grok-2-vision-1212) hỗ trợ context window rất lớn
    const uiMessages = [...convertToUIMessages(messagesFromDb), message];

    console.log(
      `[Chat API] Total messages in DB: ${messagesFromDb.length}, Sending: ${uiMessages.length} messages`
    );

    const { longitude, latitude, city, country } = geolocation(request);

    const requestHints: RequestHints = {
      longitude,
      latitude,
      city,
      country,
    };

    try {
      await saveMessages({
        messages: [
          {
            chatId: id,
            id: message.id,
            role: 'user',
            parts: message.parts,
            attachments: [],
            createdAt: new Date(),
          },
        ],
      });
    } catch (dbError: any) {
      console.error('[chat] Error saving user message:', dbError);
      // Continue even if saveMessages fails
    }

    const streamId = generateUUID();
    try {
      await createStreamId({ streamId, chatId: id });
    } catch (dbError: any) {
      console.error('[chat] Error creating stream ID:', dbError);
      // Continue even if createStreamId fails
    }

    let finalMergedUsage: AppUsage | undefined;

    // Check if message contains images - if yes, use Python backend directly with SSE
    const imageParts =
      message.parts?.filter(
        (part: any) => part.type === 'file' && part.mediaType?.startsWith('image/')
      ) || [];
    const hasImage = imageParts.length > 0;

    console.log('[Chat API] Message has image:', hasImage, 'Image count:', imageParts.length);

    // If has image, use Python backend directly with SSE (no AI SDK)
    if (hasImage) {
      console.log('[Chat API] Image detected, using Python backend with SSE (no AI SDK)');

      // Extract user query
      let userQuery = '';
      if (message.parts && message.parts.length > 0) {
        const textParts = message.parts
          .filter((part: any) => part.type === 'text')
          .map((part: any) => part.text);
        userQuery = textParts.join(' ').trim();
      }

      if (!userQuery) {
        userQuery = 'Chẩn đoán bệnh cây trồng từ hình ảnh';
      }

      // Extract and convert image to base64
      let imageData: string | null = null;
      if (imageParts.length > 0) {
        const firstImage = imageParts[0] as any;
        if (firstImage.url) {
          try {
            console.log('[Chat API] Fetching image from URL:', firstImage.url);
            const imageResponse = await fetch(firstImage.url);
            if (imageResponse.ok) {
              const imageBuffer = await imageResponse.arrayBuffer();
              const bytes = new Uint8Array(imageBuffer);
              const binary = bytes.reduce((acc, byte) => acc + String.fromCharCode(byte), '');
              imageData = btoa(binary);
              console.log('[Chat API] Image converted to base64, size:', imageData.length);
            }
          } catch (error) {
            console.error('[Chat API] Error fetching image:', error);
          }
        }
      }

      // Use createUIMessageStream for Python backend - format compatible with AI SDK
      const pythonStream = createUIMessageStream({
        execute: async ({ writer: dataStream }) => {
          const messageId = generateUUID();

          try {
            // Call Python backend
            const PYTHON_API_URL = process.env.PYTHON_API_URL || 'http://localhost:8000';
            const response = await fetch(`${PYTHON_API_URL}/api/chat`, {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json',
              },
              body: JSON.stringify({
                user_query: userQuery,
                user_context: {},
                image_data: imageData,
              }),
            });

            if (!response.ok) {
              throw new Error(`Python backend error: ${response.status}`);
            }

            const data = await response.json();
            const result = data.result || data;

            // Extract response text
            let responseText = '';
            if (result?.final_advice) {
              const agent5Output = result.final_advice;
              if (typeof agent5Output === 'string') {
                responseText = agent5Output;
              } else if (agent5Output && typeof agent5Output === 'object') {
                if (agent5Output.final_advice && typeof agent5Output.final_advice === 'object') {
                  responseText = agent5Output.final_advice.full_advice || '';
                }
                if (!responseText) {
                  responseText =
                    agent5Output.full_advice ||
                    agent5Output.diagnosis ||
                    agent5Output.summary ||
                    '';
                }
              }
            }

            if (!responseText && result?.agent_results?.agent5?.output) {
              const output = result.agent_results.agent5.output;
              if (typeof output === 'string') {
                responseText = output;
              } else if (output && typeof output === 'object') {
                responseText =
                  output.final_advice?.full_advice ||
                  output.full_advice ||
                  output.diagnosis ||
                  output.summary ||
                  '';
              }
            }

            // DO NOT fallback to agent1 llm_analysis - it's just analysis, not final advice
            if (!responseText || typeof responseText !== 'string' || responseText.trim() === '') {
              console.error('[Chat API] Could not extract response text from Python backend');
              console.error('[Chat API] Result keys:', Object.keys(result || {}));
              console.error('[Chat API] final_advice:', result?.final_advice);
              console.error('[Chat API] agent5 output:', result?.agent_results?.agent5?.output);
              responseText = 'Không thể trích xuất phản hồi từ backend. Vui lòng thử lại.';
            }

            // Stream response word by word using dataStream.write
            const words = responseText.split(/(\s+)/);

            // Send text-start
            dataStream.write({ type: 'text-start', id: messageId });

            for (const word of words) {
              if (word) {
                // Send text-delta
                dataStream.write({ type: 'text-delta', delta: word, id: messageId });
                await new Promise(resolve => setTimeout(resolve, 10));
              }
            }

            // Send text-end
            dataStream.write({ type: 'text-end', id: messageId });

            // Prepare assistant message for database saving
            const assistantMessage: ChatMessage = {
              id: messageId,
              role: 'assistant',
              parts: [{ type: 'text', text: responseText }],
            };

            // Save assistant message to database
            try {
              await saveMessages({
                messages: [
                  {
                    chatId: id,
                    id: messageId,
                    role: 'assistant',
                    parts: assistantMessage.parts,
                    attachments: [],
                    createdAt: new Date(),
                  },
                ],
              });
            } catch (dbError: any) {
              console.error('[Chat API] Error saving assistant message:', dbError);
              // Continue even if saveMessages fails
            }

            // Send usage info
            const estimatedTokens = Math.ceil(responseText.length / 4);
            const promptTokens = Math.ceil(userQuery.length / 4);
            const usage = {
              promptTokens,
              completionTokens: estimatedTokens,
              totalTokens: promptTokens + estimatedTokens,
            };
            dataStream.write({ type: 'data-usage', data: usage });
          } catch (error: any) {
            console.error('[Chat API] Python backend error:', error);
            const errorMessage = `Lỗi: ${error.message || 'Không thể kết nối đến backend'}`;
            dataStream.write({ type: 'error', errorText: errorMessage });
          }
        },
        generateId: generateUUID,
      });

      return new Response(pythonStream.pipeThrough(new JsonToSseTransformStream()));
    }

    // No image - check if query is related to agriculture/plant disease
    console.log('[Chat API] No image, checking if query is related to agriculture');

    // Extract user query
    let userQuery = '';
    if (message.parts && message.parts.length > 0) {
      const textParts = message.parts
        .filter((part: any) => part.type === 'text')
        .map((part: any) => part.text);
      userQuery = textParts.join(' ').trim();
    }

    if (!userQuery) {
      userQuery = 'Xin chào';
    }

    // Check if query is related to agriculture/plant disease/pesticides
    const isAgricultureQuery = (query: string): boolean => {
      const queryLower = query.toLowerCase();

      // Pattern 1: Câu hỏi về thuốc (có từ "thuốc" + tên thuốc hoặc từ khóa liên quan)
      const hasMedicineQuestion =
        /thuốc\s+(gì|nào|là|để|dùng|sử dụng|trị|chữa)/i.test(query) ||
        /(pesticide|insecticide|fungicide|herbicide)/i.test(query) ||
        /trừ\s+sâu|diệt\s+côn\s+trùng|bảo\s+vệ\s+thực\s+vật/i.test(query);

      // Pattern 2: Tên thuốc (thường là từ tiếng Anh, có thể có số hoặc ký tự đặc biệt)
      // Pattern: từ có chữ cái + số hoặc từ dài > 8 ký tự (có thể là tên thuốc khoa học)
      const hasMedicineName =
        /\b[a-z]{6,}(?:idin|phos|ate|ide|ol|in)\b/i.test(query) ||
        /\b[a-z]+(?:[0-9]+|[a-z]{4,})\b/i.test(query);

      // Pattern 3: Câu hỏi về bệnh cây trồng
      const hasDiseaseQuestion =
        /(bệnh|chẩn\s+đoán|nhận\s+dạng|triệu\s+chứng|điều\s+trị|chữa)/i.test(query);

      // Pattern 4: Câu hỏi về cây trồng
      const hasPlantQuestion = /\b(cây|lá|thân|rễ|quả|hoa|trái)\b/i.test(query);

      // Pattern 5: Câu hỏi về nông nghiệp
      const hasAgricultureQuestion = /(nông\s+nghiệp|trồng\s+trọt|canh\s+tác|vườn|ruộng)/i.test(
        query
      );

      // Pattern 6: Câu hỏi về sâu bệnh
      const hasPestQuestion = /\b(sâu|bọ|nấm|vi\s+khuẩn|virus|rầy|rệp|bọ\s+xít)\b/i.test(query);

      // Nếu có bất kỳ pattern nào → là câu hỏi về nông nghiệp
      return (
        hasMedicineQuestion ||
        (hasMedicineName && (hasMedicineQuestion || /thuốc/i.test(query))) ||
        hasDiseaseQuestion ||
        hasPlantQuestion ||
        hasAgricultureQuestion ||
        hasPestQuestion
      );
    };

    // If query is related to agriculture, use Python backend
    if (isAgricultureQuery(userQuery)) {
      console.log('[Chat API] Agriculture-related query detected, using Python backend');

      // Prepare conversation history for context
      const conversationHistory = uiMessages
        .slice(-10) // Last 10 messages for context (5 pairs)
        .map((msg: ChatMessage) => ({
          role: msg.role,
          content:
            msg.parts
              ?.filter((part: any) => part.type === 'text')
              .map((part: any) => part.text)
              .join(' ') || '',
        }))
        .filter((msg: any) => msg.content.trim().length > 0);

      // Use Python backend for agriculture queries
      const pythonStream = createUIMessageStream({
        execute: async ({ writer: dataStream }) => {
          const messageId = generateUUID();

          try {
            // Call Python backend
            const PYTHON_API_URL = process.env.PYTHON_API_URL || 'http://localhost:8000';
            const response = await fetch(`${PYTHON_API_URL}/api/chat`, {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json',
              },
              body: JSON.stringify({
                user_query: userQuery,
                user_context: {
                  conversation_history: conversationHistory,
                },
                image_data: null,
              }),
            });

            if (!response.ok) {
              throw new Error(`Python backend error: ${response.status}`);
            }

            const data = await response.json();
            const result = data.result || data;

            // Extract response text
            let responseText = '';
            if (result?.final_advice) {
              const agent5Output = result.final_advice;
              if (typeof agent5Output === 'string') {
                responseText = agent5Output;
              } else if (agent5Output && typeof agent5Output === 'object') {
                if (agent5Output.final_advice && typeof agent5Output.final_advice === 'object') {
                  responseText = agent5Output.final_advice.full_advice || '';
                }
                if (!responseText) {
                  responseText =
                    agent5Output.full_advice ||
                    agent5Output.diagnosis ||
                    agent5Output.summary ||
                    '';
                }
              }
            }

            if (!responseText && result?.agent_results?.agent5?.output) {
              const output = result.agent_results.agent5.output;
              if (typeof output === 'string') {
                responseText = output;
              } else if (output && typeof output === 'object') {
                responseText =
                  output.final_advice?.full_advice ||
                  output.full_advice ||
                  output.diagnosis ||
                  output.summary ||
                  '';
              }
            }

            // DO NOT fallback to agent1 llm_analysis - it's just analysis, not final advice
            if (!responseText || typeof responseText !== 'string' || responseText.trim() === '') {
              console.error('[Chat API] Could not extract response text from Python backend');
              console.error('[Chat API] Result keys:', Object.keys(result || {}));
              console.error('[Chat API] final_advice:', result?.final_advice);
              console.error('[Chat API] agent5 output:', result?.agent_results?.agent5?.output);
              responseText = 'Không thể trích xuất phản hồi từ backend. Vui lòng thử lại.';
            }

            // Stream response word by word using dataStream.write
            const words = responseText.split(/(\s+)/);

            // Send text-start
            dataStream.write({ type: 'text-start', id: messageId });

            for (const word of words) {
              if (word) {
                // Send text-delta
                dataStream.write({ type: 'text-delta', delta: word, id: messageId });
                await new Promise(resolve => setTimeout(resolve, 10));
              }
            }

            // Send text-end
            dataStream.write({ type: 'text-end', id: messageId });

            // Prepare assistant message for database saving
            const assistantMessage: ChatMessage = {
              id: messageId,
              role: 'assistant',
              parts: [{ type: 'text', text: responseText }],
            };

            // Save assistant message to database
            try {
              await saveMessages({
                messages: [
                  {
                    chatId: id,
                    id: messageId,
                    role: 'assistant',
                    parts: assistantMessage.parts,
                    attachments: [],
                    createdAt: new Date(),
                  },
                ],
              });
            } catch (dbError: any) {
              console.error('[Chat API] Error saving assistant message:', dbError);
              // Continue even if saveMessages fails
            }

            // Send usage info
            const estimatedTokens = Math.ceil(responseText.length / 4);
            const promptTokens = Math.ceil(userQuery.length / 4);
            const usage = {
              promptTokens,
              completionTokens: estimatedTokens,
              totalTokens: promptTokens + estimatedTokens,
            };
            dataStream.write({ type: 'data-usage', data: usage });
          } catch (error: any) {
            console.error('[Chat API] Python backend error:', error);
            const errorMessage = `Lỗi: ${error.message || 'Không thể kết nối đến backend'}`;
            dataStream.write({ type: 'error', errorText: errorMessage });
          }
        },
        generateId: generateUUID,
      });

      return new Response(pythonStream.pipeThrough(new JsonToSseTransformStream()));
    }

    // Not agriculture-related - return simple response
    console.log('[Chat API] Not agriculture-related, returning simple response');

    // Use createUIMessageStream for simple response
    const simpleStream = createUIMessageStream({
      execute: async ({ writer: dataStream }) => {
        const messageId = generateUUID();

        // Simple response for normal chat
        const responseText = `Xin chào! Tôi là AI chuyên về chẩn đoán bệnh cây trồng.

Để tôi có thể hỗ trợ bạn tốt nhất, vui lòng:
- Upload hình ảnh cây bị bệnh để tôi có thể chẩn đoán
- Hoặc mô tả chi tiết về triệu chứng bệnh của cây

Tôi sẽ sử dụng hệ thống 5 agents để phân tích và đưa ra lời khuyên chi tiết.`;

        // Stream response word by word using dataStream.write
        const words = responseText.split(/(\s+)/);

        // Send text-start
        dataStream.write({ type: 'text-start', id: messageId });

        for (const word of words) {
          if (word) {
            // Send text-delta
            dataStream.write({ type: 'text-delta', delta: word, id: messageId });
            await new Promise(resolve => setTimeout(resolve, 10));
          }
        }

        // Send text-end
        dataStream.write({ type: 'text-end', id: messageId });

        // Prepare assistant message for database saving
        const assistantMessage: ChatMessage = {
          id: messageId,
          role: 'assistant',
          parts: [{ type: 'text', text: responseText }],
        };

        // Save assistant message to database
        try {
          await saveMessages({
            messages: [
              {
                chatId: id,
                id: messageId,
                role: 'assistant',
                parts: assistantMessage.parts,
                attachments: [],
                createdAt: new Date(),
              },
            ],
          });
        } catch (dbError: any) {
          console.error('[Chat API] Error saving assistant message:', dbError);
          // Continue even if saveMessages fails
        }

        // Send usage info
        const estimatedTokens = Math.ceil(responseText.length / 4);
        const promptTokens = Math.ceil(userQuery.length / 4);
        const usage = {
          promptTokens,
          completionTokens: estimatedTokens,
          totalTokens: promptTokens + estimatedTokens,
        };
        dataStream.write({ type: 'data-usage', data: usage });
      },
      generateId: generateUUID,
    });

    return new Response(simpleStream.pipeThrough(new JsonToSseTransformStream()));
  } catch (error) {
    const vercelId = request.headers.get('x-vercel-id');

    if (error instanceof ChatSDKError) {
      return error.toResponse();
    }

    // Check for Vercel AI Gateway errors
    if (error instanceof Error) {
      if (
        error.message?.includes(
          'AI Gateway requires a valid credit card on file to service requests'
        )
      ) {
        return new ChatSDKError('bad_request:activate_gateway').toResponse();
      }

      // Handle insufficient funds error (should not happen now, but keep for safety)
      if (
        error.message?.includes('Insufficient funds') ||
        error.message?.includes('insufficient_funds')
      ) {
        console.error('[chat] Vercel AI Gateway insufficient funds (should not happen)');
        return new ChatSDKError('bad_request:api').toResponse();
      }

      // Handle other AI Gateway errors (should not happen now, but keep for safety)
      if (
        error.message?.includes('AI Gateway') ||
        error.message?.includes('GatewayInternalServerError')
      ) {
        console.error('[chat] AI Gateway error (should not happen):', error.message);
        return new ChatSDKError('bad_request:api').toResponse();
      }
    }

    // Log detailed error information
    console.error('[chat] Unhandled error in chat API:', error, { vercelId });
    if (error instanceof Error) {
      console.error('[chat] Error message:', error.message);
      console.error('[chat] Error stack:', error.stack);
      console.error('[chat] Error name:', error.name);
    }

    // Check if it's a database error
    const isDatabaseError =
      error instanceof Error &&
      (error.message?.includes('database') ||
        error.message?.includes('POSTGRES') ||
        error.message?.includes('connection') ||
        error.message?.includes('timeout'));

    // Check if it's a tool execution error (plantDiagnosis, etc.)
    const isToolError =
      error instanceof Error &&
      (error.message?.includes('plantDiagnosis') ||
        error.message?.includes('tool') ||
        error.message?.includes('backend'));

    // Return more specific error based on error type
    if (isDatabaseError) {
      console.error('[chat] Database error detected, but continuing...');
      // Don't return error, let it continue (database errors are non-critical)
      // Return a generic error instead
      return new ChatSDKError('bad_request:api').toResponse();
    }

    if (isToolError) {
      // Tool errors should be handled by the tool itself
      // But if it reaches here, return a more helpful error
      return new ChatSDKError('bad_request:api').toResponse();
    }

    // For other errors, return offline:chat
    return new ChatSDKError('offline:chat').toResponse();
  }
}

export async function DELETE(request: Request) {
  const { searchParams } = new URL(request.url);
  const id = searchParams.get('id');

  if (!id) {
    return new ChatSDKError('bad_request:api').toResponse();
  }

  const session = await auth();

  if (!session?.user) {
    return new ChatSDKError('unauthorized:chat').toResponse();
  }

  const chat = await getChatById({ id });

  if (chat?.userId !== session.user.id) {
    return new ChatSDKError('forbidden:chat').toResponse();
  }

  const deletedChat = await deleteChatById({ id });

  return Response.json(deletedChat, { status: 200 });
}
