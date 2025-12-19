'use server';

import type { UIMessage } from 'ai';
import { cookies } from 'next/headers';
import type { VisibilityType } from '@/components/visibility-selector';
import {
  deleteMessagesByChatIdAfterTimestamp,
  getMessageById,
  updateChatVisibilityById,
} from '@/lib/db/queries';
import { getTextFromMessage } from '@/lib/utils';

export async function saveChatModelAsCookie(model: string) {
  const cookieStore = await cookies();
  cookieStore.set('chat-model', model);
}

export async function generateTitleFromUserMessage({ message }: { message: UIMessage }) {
  // Không dùng Vercel AI Gateway nữa, dùng simple title từ user message
  // Để tránh lỗi "Insufficient funds" và không phụ thuộc vào Vercel AI Gateway

  const userText = getTextFromMessage(message);

  if (userText && userText.length > 0) {
    // Lấy 50 ký tự đầu tiên làm title
    let title = userText.substring(0, 50).trim();

    // Nếu có dấu câu, cắt tại dấu câu gần nhất
    const punctuation = ['.', '!', '?', '。', '！', '？'];
    for (const punc of punctuation) {
      const index = title.lastIndexOf(punc);
      if (index > 10) {
        title = title.substring(0, index + 1);
        break;
      }
    }

    return title || 'New Chat';
  }

  return 'New Chat';
}

export async function deleteTrailingMessages({ id }: { id: string }) {
  const [message] = await getMessageById({ id });

  await deleteMessagesByChatIdAfterTimestamp({
    chatId: message.chatId,
    timestamp: message.createdAt,
  });
}

export async function updateChatVisibility({
  chatId,
  visibility,
}: {
  chatId: string;
  visibility: VisibilityType;
}) {
  await updateChatVisibilityById({ chatId, visibility });
}
