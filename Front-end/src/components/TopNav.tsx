import type { PageKey } from '../data/profile'

const ITEMS: { key: PageKey; label: string }[] = [
  { key: 'home', label: 'Home' },
  { key: 'info', label: 'Nhập thông tin' },
  { key: 'chatbot', label: 'Chatbot AI' },
  { key: 'proposal', label: 'Chẩn đoán hồ sơ' },
]

/**
 * The pill navigation shown at the top of every screen. The active page is a
 * filled royal-blue pill; the rest are outlined blue pills.
 */
export default function TopNav({
  active,
  onNavigate,
}: {
  active: PageKey
  onNavigate: (page: PageKey) => void
}) {
  return (
    <nav className="flex flex-wrap items-center gap-3">
      {ITEMS.map(({ key, label }) => {
        const isActive = key === active
        return (
          <button
            key={key}
            type="button"
            onClick={() => onNavigate(key)}
            className={`rounded-xl px-5 py-2.5 text-sm font-semibold transition ${
              isActive
                ? 'bg-brand-600 text-white shadow-sm'
                : 'border border-brand-200 bg-white text-brand-600 hover:bg-brand-50'
            }`}
          >
            {label}
          </button>
        )
      })}
    </nav>
  )
}
