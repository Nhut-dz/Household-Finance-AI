import { apiGet, apiPost } from '../lib/api'
import { getGuestSessionId, guestQuery } from '../lib/guestSession'

export interface ChatMessage {
  id: number
  role: 'ai' | 'user'
  content: string
  created_at: string
  /** Chỉ có ở tin nhắn của AI. */
  suggested_questions?: string[] | null
}

/** Một phiên trò chuyện. Phiên `closed` chỉ để xem lại, không nhận tin nhắn mới. */
export interface Conversation {
  id: number
  status: 'active' | 'closed'
  /** 'profile_updated' khi phiên bị thay vì hồ sơ tài chính đổi. */
  closed_reason: string | null
  /** Số LƯỢT hỏi đáp; mỗi lượt hiển thị thành 2 message. */
  turn_count: number
  created_at: string
  closed_at: string | null
}

export interface ChatHistory {
  /** Phiên đang mở; null khi hộ chưa hỏi câu nào. */
  conversation_id: number | null
  messages: ChatMessage[]
}

/**
 * Hội thoại của PHIÊN ĐANG MỞ, sắp xếp theo thời gian tăng dần.
 *
 * Không phải toàn bộ lịch sử của hộ. Sau khi người dùng sửa dữ liệu tài chính,
 * backend đóng phiên cũ nên hàm này trả về danh sách rỗng và một conversation_id
 * mới — đó là tín hiệu để xoá hội thoại đang hiển thị.
 */
export const getMessages = (householdId: number) =>
  apiGet<ChatHistory>(`/households/${householdId}/messages${guestQuery()}`)

/**
 * Gửi câu hỏi tự do. Backend chuyển sang service tư vấn của nhóm Python rồi trả
 * về cả câu hỏi lẫn câu trả lời để render một lần.
 *
 * Không gửi kèm conversation_id: phiên được xác định ở server theo hồ sơ. Gửi
 * id từ client lên thì một client giữ id cũ sau khi hồ sơ đổi sẽ ghi tin nhắn
 * mới vào phiên đã đóng.
 */
export const sendMessage = (householdId: number, content: string) =>
  apiPost<{
    conversation_id: number
    user_message: ChatMessage
    ai_message: ChatMessage
  }>(`/households/${householdId}/messages`, {
    content,
    guest_session_id: getGuestSessionId(),
  })

/** Danh sách phiên của hộ, mới nhất trước — cho màn xem lại lịch sử. */
export const getConversations = (householdId: number) =>
  apiGet<Conversation[]>(`/households/${householdId}/conversations${guestQuery()}`)

/** Nội dung của một phiên cụ thể, kể cả phiên đã đóng. */
export const getConversationMessages = (
  householdId: number,
  conversationId: number,
) =>
  apiGet<ChatMessage[]>(
    `/households/${householdId}/conversations/${conversationId}/messages${guestQuery()}`,
  )
