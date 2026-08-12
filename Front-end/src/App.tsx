import { useEffect, useState } from 'react'
import TopNav from './components/TopNav'
import HomePage from './pages/HomePage'
import InfoFormPage from './pages/InfoFormPage'
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
              onSaved={(id) => {
                setHouseholdId(id)
                setPage('chatbot')
              }}
            />
          )}

          {page === 'chatbot' && (
            <ChatbotPage
              profile={profile}
              householdId={householdId}
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
