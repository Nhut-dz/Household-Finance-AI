import { apiDelete, apiGet, apiPost } from '../lib/api'
import { getGuestSessionId, guestQuery } from '../lib/guestSession'

/**
 * Mã ý định của một lượt hỏi. Giữ khớp `IntentCodeEnum` phía backend và
 * `IntentCode` của service Python — ba nơi lệch nhau thì mã gửi xuống không
 * khớp giá trị nào, engine coi như không có mã và quay về đoán bằng từ khoá,
 * tức im lặng mất đúng tính năng này.
 */
export type IntentCode =
  | 'SAVINGS_PACKAGE'
  | 'FINANCIAL_HEALTH_DIAGNOSIS'
  | 'LOAN_RISK_DIAGNOSIS'
  | 'BUDGET_50_30_20'
  | 'LOAN_CAPACITY'
  | 'DEBT'
  | 'INVESTMENT'
  | 'GENERAL'

export interface ChatMessage {
  id: number
  role: 'ai' | 'user'
  content: string
  created_at: string
  /** Chỉ có ở tin nhắn của AI. */
  suggested_questions?: string[] | null
  /**
   * Ý định mà engine THỰC SỰ đã chạy. Chỉ có ở response của lần gửi, không có
   * khi tải lại lịch sử — đây là thông tin của một lượt gọi, không lưu vào DB.
   */
  intent_code?: IntentCode | null
  /** ML02 báo hộ chưa khai thông tin khoản vay → hiện nút điều hướng. */
  requires_loan_application?: boolean
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
 * Gửi một lượt hỏi. Backend chuyển sang service tư vấn của nhóm Python rồi trả
 * về cả câu hỏi lẫn câu trả lời để render một lần.
 *
 * `intentCode` chỉ truyền khi người dùng bấm chip gợi ý — khi đó ý định là đã
 * biết chắc, engine đi thẳng vào đúng chức năng. Câu tự gõ để trống, engine mới
 * đoán bằng từ khoá. Hai chức năng chạy model (`FINANCIAL_HEALTH_DIAGNOSIS` →
 * ML01, `LOAN_RISK_DIAGNOSIS` → ML02) chỉ kích hoạt được bằng đường thứ nhất:
 * đoán từ nhãn tiếng Việt của chúng sẽ rơi vào nhánh sai mà vẫn trả lời trôi
 * chảy, nên đường đoán cố ý không dẫn tới model.
 *
 * Không gửi kèm conversation_id: phiên được xác định ở server theo hồ sơ. Gửi
 * id từ client lên thì một client giữ id cũ sau khi hồ sơ đổi sẽ ghi tin nhắn
 * mới vào phiên đã đóng.
 */
export const sendMessage = (
  householdId: number,
  content: string,
  intentCode?: IntentCode,
) =>
  apiPost<{
    conversation_id: number
    user_message: ChatMessage
    ai_message: ChatMessage
  }>(`/households/${householdId}/messages`, {
    content,
    intent_code: intentCode ?? null,
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

/**
 * Xoá HẲN mọi phiên trò chuyện của hộ, giữ nguyên hồ sơ.
 *
 * Dùng cho nút "Xóa dữ liệu" ở form khoản vay. Không nhầm với việc backend tự
 * xoay phiên khi hồ sơ đổi: xoay phiên chỉ đánh dấu `closed`, nội dung vẫn đọc
 * được qua `getConversations()`. Hàm này xoá thật.
 */
export const deleteConversations = (householdId: number) =>
  apiDelete<null>(`/households/${householdId}/conversations${guestQuery()}`)
