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

    const stream = createUIMessageStream({
      execute: ({ writer: dataStream }) => {
        // Tối ưu messages: Nếu có quá nhiều messages, chỉ giữ lại message mới nhất và system prompt
        // Điều này giúp tránh vượt quá token limit của Vercel AI Gateway (8192 tokens)
        let optimizedMessages = convertToModelMessages(uiMessages);

        // Nếu có quá nhiều messages (>20), chỉ giữ lại:
        // - System message (từ systemPrompt)
        // - 2 messages gần nhất (1 cặp user-assistant) để giữ context
        // - Message hiện tại
        if (uiMessages.length > 20) {
          console.log(
            `[Chat API] Optimizing messages: ${uiMessages.length} -> keeping last 2 + current`
          );
          // Lấy 2 messages gần nhất (trước message hiện tại)
          const recentMessages = uiMessages.slice(-3, -1); // 2 messages trước message cuối
          optimizedMessages = convertToModelMessages([...recentMessages, message]);
        }

        const result = streamText({
          model: myProvider.languageModel(selectedChatModel),
          system: systemPrompt({ selectedChatModel, requestHints }),
          messages: optimizedMessages,
          stopWhen: stepCountIs(5),
          // Note: maxTokens không còn được hỗ trợ trong AI SDK
          // Model sẽ tự quyết định số lượng tokens dựa trên context window
          experimental_activeTools:
            selectedChatModel === 'chat-model-reasoning'
              ? []
              : [
                  'getWeather',
                  'createDocument',
                  'updateDocument',
                  'requestSuggestions',
                  'plantDiagnosis',
                ],
          experimental_transform: smoothStream({ chunking: 'word' }),
          tools: {
            getWeather,
            plantDiagnosis: createPlantDiagnosisTool(uiMessages),
            createDocument: createDocument({ session, dataStream }),
            updateDocument: updateDocument({ session, dataStream }),
            requestSuggestions: requestSuggestions({
              session,
              dataStream,
            }),
          },
          experimental_telemetry: {
            isEnabled: isProductionEnvironment,
            functionId: 'stream-text',
          },
          onFinish: async ({ usage }) => {
            try {
              const providers = await getTokenlensCatalog();
              const modelId = myProvider.languageModel(selectedChatModel).modelId;
              if (!modelId) {
                finalMergedUsage = usage;
                dataStream.write({
                  type: 'data-usage',
                  data: finalMergedUsage,
                });
                return;
              }

              if (!providers) {
                finalMergedUsage = usage;
                dataStream.write({
                  type: 'data-usage',
                  data: finalMergedUsage,
                });
                return;
              }

              const summary = getUsage({ modelId, usage, providers });
              finalMergedUsage = { ...usage, ...summary, modelId } as AppUsage;
              dataStream.write({ type: 'data-usage', data: finalMergedUsage });
            } catch (err) {
              console.warn('TokenLens enrichment failed', err);
              finalMergedUsage = usage;
              dataStream.write({ type: 'data-usage', data: finalMergedUsage });
            }
          },
        });

        result.consumeStream();

        dataStream.merge(
          result.toUIMessageStream({
            sendReasoning: true,
          })
        );
      },
      generateId: generateUUID,
      onFinish: async ({ messages }) => {
        try {
          await saveMessages({
            messages: messages.map(currentMessage => ({
              id: currentMessage.id,
              role: currentMessage.role,
              parts: currentMessage.parts,
              createdAt: new Date(),
              attachments: [],
              chatId: id,
            })),
          });
        } catch (dbError: any) {
          console.error('[chat] Error saving messages in onFinish:', dbError);
          // Continue even if saveMessages fails
        }

        if (finalMergedUsage) {
          try {
            await updateChatLastContextById({
              chatId: id,
              context: finalMergedUsage,
            });
          } catch (err) {
            console.warn('Unable to persist last usage for chat', id, err);
          }
        }
      },
      onError: error => {
        console.error('[chat] Error in streamText:', error);
        // Return user-friendly error message
        if (error instanceof Error) {
          return `Đã xảy ra lỗi: ${error.message}`;
        }
        return 'Oops, an error occurred!';
      },
    });

    // const streamContext = getStreamContext();

    // if (streamContext) {
    //   return new Response(
    //     await streamContext.resumableStream(streamId, () =>
    //       stream.pipeThrough(new JsonToSseTransformStream())
    //     )
    //   );
    // }

    return new Response(stream.pipeThrough(new JsonToSseTransformStream()));
  } catch (error) {
    const vercelId = request.headers.get('x-vercel-id');

    if (error instanceof ChatSDKError) {
      return error.toResponse();
    }

    // Check for Vercel AI Gateway credit card error
    if (
      error instanceof Error &&
      error.message?.includes('AI Gateway requires a valid credit card on file to service requests')
    ) {
      return new ChatSDKError('bad_request:activate_gateway').toResponse();
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
