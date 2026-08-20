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
                  setChatResetToken((token) => token + 1)
                }
                setPage('chatbot')
              }}
            />
          )}

          {page === 'loan' && (
            <LoanFormPage householdId={householdId} onNavigate={setPage} />
          )}

          {page === 'chatbot' && (
            <ChatbotPage
              profile={profile}
              householdId={householdId}
              chatResetToken={chatResetToken}
              onNavigate={setPage}
            />
          )}

          {page === 'proposal' && (
            <ProposalPage
              householdId={householdId}
              onNavigate={setPage}
              onCleared={() => {
                setHouseholdId(null)
                setProfile(emptyProfile)
              }}
            />
          )}
        </div>
      </div>
    </div>
  )
}
