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

/** Lịch sử hội thoại, sắp xếp theo thời gian tăng dần. */
export const getMessages = (householdId: number) =>
  apiGet<ChatMessage[]>(`/households/${householdId}/messages${guestQuery()}`)

/**
 * Gửi câu hỏi tự do. Backend chuyển sang service tư vấn của nhóm Python rồi trả
 * về cả câu hỏi lẫn câu trả lời để render một lần.
 */
export const sendMessage = (householdId: number, content: string) =>
  apiPost<{ user_message: ChatMessage; ai_message: ChatMessage }>(
    `/households/${householdId}/messages`,
    { content, guest_session_id: getGuestSessionId() },
  )
