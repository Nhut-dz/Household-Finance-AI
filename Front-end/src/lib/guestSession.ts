const STORAGE_KEY = 'hf.guestSessionId'

/** UUID v4 dự phòng cho môi trường không có crypto.randomUUID (http, trình duyệt cũ). */
function fallbackUuid(): string {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0
    const v = c === 'x' ? r : (r & 0x3) | 0x8
    return v.toString(16)
  })
}

/**
 * Mã phiên dùng cho người dùng chưa đăng nhập. Backend bắt buộc trường
 * `guest_session_id` khi request không kèm Bearer token, và dùng nó để gom các
 * bản ghi của cùng một khách, nên mã phải giữ nguyên giữa các lần vào trang.
 */
export function getGuestSessionId(): string {
  const stored = window.localStorage.getItem(STORAGE_KEY)
  if (stored) return stored

  const id = crypto.randomUUID?.() ?? fallbackUuid()
  window.localStorage.setItem(STORAGE_KEY, id)
  return id
}

/**
 * Query string định danh phiên khách cho các endpoint GET / DELETE. Khi đã đăng
 * nhập, backend bỏ qua tham số này và xét quyền theo user_id.
 */
export const guestQuery = () =>
  `?guest_session_id=${encodeURIComponent(getGuestSessionId())}`
