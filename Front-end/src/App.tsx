import { useEffect, useState } from 'react'
import TopNav from './components/TopNav'
import HomePage from './pages/HomePage'
import InfoFormPage from './pages/InfoFormPage'
import LoanFormPage from './pages/LoanFormPage'
import ChatbotPage from './pages/ChatbotPage'
import ProposalPage from './pages/ProposalPage'
import { fromHouseholdResponse, getLatestHousehold } from './api/households'
import {
  emptyProfile,
  type HouseholdProfile,
  type PageKey,
} from './data/profile'

export default function App() {
  const [page, setPage] = useState<PageKey>('home')
  const [profile, setProfile] = useState<HouseholdProfile>(emptyProfile)
  /** Id bản ghi trên backend; null khi phiên này chưa từng gửi hồ sơ. */
  const [householdId, setHouseholdId] = useState<number | null>(null)
  /**
   * Tăng lên mỗi khi backend xoay phiên trò chuyện vì hồ sơ tài chính đổi.
   * Màn Chatbot lấy giá trị này làm phụ thuộc của effect nạp hội thoại, nên
   * cùng một `householdId` vẫn nạp lại được — đây chính là chỗ trước đây hỏng:
   * sửa hồ sơ không đổi id nên effect không chạy lại và hội thoại cũ nằm nguyên.
   */
  const [chatResetToken, setChatResetToken] = useState(0)
  /**
   * VÌ SAO hội thoại bị dọn — quyết định câu giải thích màn Chatbot hiện ra.
   *
   * Hai lý do dẫn tới cùng một màn chat trống nhưng sự thật khác hẳn nhau:
   * `rotated` là phiên cũ bị đóng và VẪN CÒN trong DB, `cleared` là người dùng
   * bấm "Xóa dữ liệu" và phiên cũ đã bị XOÁ HẲN. Nói nhầm câu "hội thoại trước
   * vẫn được lưu lại" cho trường hợp thứ hai là hứa với người dùng một thứ
   * không còn tồn tại.
   */
  const [chatResetReason, setChatResetReason] =
    useState<'rotated' | 'cleared' | null>(null)

  const resetChat = (reason: 'rotated' | 'cleared') => {
    setChatResetToken((token) => token + 1)
    setChatResetReason(reason)
  }

  /**
   * Gỡ lời giải thích khi nó đã hết đúng.
   *
   * `chatResetReason` mô tả VÌ SAO khung chat đang trống NGAY LÚC NÀY, không
   * phải nhật ký việc đã từng xảy ra. Người dùng xoá dữ liệu rồi nhập hồ sơ
   * mới thì khung chat vẫn trống, nhưng lý do đã đổi: không còn là "vừa xoá
   * sạch" mà là "hồ sơ mới, chưa hỏi câu nào". Giữ nguyên câu cũ là nói với
   * người vừa nhập xong rằng dữ liệu của họ không được dùng lại — đúng thứ
   * gây hiểu lầm.
   */
  const clearChatNotice = () => setChatResetReason(null)

  // Khôi phục hồ sơ gần nhất của phiên để refresh trang không mất dữ liệu.
  // 404 nghĩa là chưa có hồ sơ nào, đó là trạng thái bình thường của người mới.
  useEffect(() => {
    getLatestHousehold()
      .then((data) => {
        setHouseholdId(data.id)
        setProfile(fromHouseholdResponse(data))
      })
      .catch(() => undefined)
  }, [])

  const patchProfile = (patch: Partial<HouseholdProfile>) =>
    setProfile((prev) => ({ ...prev, ...patch }))

  /**
   * Đưa toàn bộ state của phiên về đúng trạng thái người dùng mới.
   *
   * Ba việc, thiếu việc nào cũng để lại dữ liệu cũ:
   *   - `profile` → rỗng, nếu không form vẫn hiện số đã nhập.
   *   - `householdId` → null, nếu không lần lưu sau là PUT lên một hồ sơ đã bị
   *     xoá và nhận 404.
   *   - `chatResetToken` tăng — màn Chatbot lấy nó làm phụ thuộc của effect nạp
   *     hội thoại. Đặt `householdId` về null thôi thì effect có chạy lại, nhưng
   *     token tăng là thứ đảm bảo chạy lại ngay cả khi id không đổi.
   */
  const resetSession = () => {
    setProfile(emptyProfile)
    setHouseholdId(null)
    resetChat('cleared')
  }

  return (
    <div className="min-h-screen w-full bg-slate-50">
      <div className="mx-auto w-full max-w-6xl px-4 py-6 sm:px-6">
        <TopNav active={page} onNavigate={setPage} />

        <div className="mt-6">
          {page === 'home' && <HomePage onNavigate={setPage} />}

          {page === 'info' && (
            <InfoFormPage
              profile={profile}
              householdId={householdId}
              onChange={patchProfile}
              onNavigate={setPage}
              onSaved={(id, info) => {
                setHouseholdId(id)
                if (info?.conversationRotated) {
                  resetChat('rotated')
                } else {
                  // Vừa lưu hồ sơ mà phiên không bị xoay: hồ sơ mới tạo, hoặc
                  // sửa thứ không ảnh hưởng phân tích. Cả hai trường hợp đều
                  // không còn là "vừa xoá dữ liệu" nữa.
                  clearChatNotice()
                }
                setPage('chatbot')
              }}
              onCleared={resetSession}
            />
          )}

          {page === 'loan' && (
            <LoanFormPage
              householdId={householdId}
              onNavigate={setPage}
              // Chỉ hội thoại bị xoá, hồ sơ giữ nguyên — nên KHÔNG gọi
              // `resetSession()`: nó sẽ xoá luôn `profile` và `householdId`
              // đang còn hiệu lực, và màn Nhập thông tin trống trơn dù dữ liệu
              // vẫn nằm nguyên trên backend.
              onConversationCleared={() => resetChat('cleared')}
              onSaved={clearChatNotice}
            />
          )}

          {page === 'chatbot' && (
            <ChatbotPage
              profile={profile}
              householdId={householdId}
              chatResetToken={chatResetToken}
              chatResetReason={chatResetReason}
              onNavigate={setPage}
            />
          )}

          {page === 'proposal' && (
            <ProposalPage
              householdId={householdId}
              onNavigate={setPage}
              onCleared={resetSession}
            />
          )}
        </div>
      </div>
    </div>
  )
}
