import { useEffect, useState } from 'react'
import {
  AlertCircle,
  Lightbulb,
  Loader2,
  LogOut,
  Mic,
  Pencil,
  Send,
  Volume2,
} from 'lucide-react'
import {
  NEED_LABELS,
  type HouseholdProfile,
  type PageKey,
} from '../data/profile'
import { dong } from '../lib/format'
import { getMessages, sendMessage, type ChatMessage } from '../api/messages'
import { ApiError } from '../lib/api'

/** Đọc câu trả lời bằng giọng nói của trình duyệt (không cần API riêng). */
function speak(text: string) {
  if (!('speechSynthesis' in window)) return

  window.speechSynthesis.cancel()
  const utterance = new SpeechSynthesisUtterance(text)
  utterance.lang = 'vi-VN'
  window.speechSynthesis.speak(utterance)
}

const QUICK = [
  {
    label: 'Gói tiết kiệm',
    tone: 'bg-brand-100 text-brand-700 hover:bg-brand-200',
    iconTone: 'text-amber-500 fill-amber-300',
  },
  {
    label: 'Gói đầu tư',
    tone: 'bg-amber-400 text-white hover:bg-amber-500',
    iconTone: 'text-white fill-white/40',
  },
  {
    label: 'Gói vay mua nhà',
    tone: 'bg-brand-600 text-white hover:bg-brand-700',
    iconTone: 'text-amber-300 fill-amber-300/40',
  },
  {
    label: 'Quy tắc 50/30/20',
    tone: 'bg-purple-600 text-white hover:bg-purple-700',
    iconTone: 'text-amber-300 fill-amber-300/40',
  },
]

/** App logo used as the assistant's avatar. */
function BotAvatar({ className = '' }: { className?: string }) {
  return (
    <img
      src="/iconlogo.png"
      alt=""
      className={`shrink-0 rounded-lg object-cover ${className}`}
    />
  )
}

function ProfileRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between border-t border-slate-100 py-2.5 first:border-t-0">
      <span className="text-sm text-slate-400">{label}</span>
      <span className="text-sm font-semibold text-slate-800">{value}</span>
    </div>
  )
}

export default function ChatbotPage({
  profile,
  householdId,
  onNavigate,
}: {
  profile: HouseholdProfile
  householdId: number | null
  onNavigate: (page: PageKey) => void
}) {
  const need = profile.needs.map((n) => NEED_LABELS[n]).join(', ') || 'Mua nhà'
  const memberText = `${profile.members} người (${Math.max(
    0,
    profile.members - profile.children - (profile.supportingElderly ? 1 : 0),
  )} người lớn, ${profile.children} con${
    profile.supportingElderly ? ', 1 người già' : ''
  })`

  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [draft, setDraft] = useState('')
  const [loading, setLoading] = useState(false)
  const [sending, setSending] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Tải lại hội thoại đã lưu mỗi khi mở màn Chatbot.
  useEffect(() => {
    if (householdId === null) return

    setLoading(true)
    getMessages(householdId)
      .then(setMessages)
      .catch(() => undefined)
      .finally(() => setLoading(false))
  }, [householdId])

  const send = async (text: string) => {
    const trimmed = text.trim()
    if (!trimmed || sending) return

    if (householdId === null) {
      setError('Bạn cần gửi thông tin hộ gia đình trước khi trò chuyện với AI.')
      return
    }

    setSending(true)
    setError(null)
    setDraft('')

    try {
      const { user_message, ai_message } = await sendMessage(householdId, trimmed)
      setMessages((prev) => [...prev, user_message, ai_message])
    } catch (err) {
      setError(
        err instanceof ApiError
          ? err.message
          : 'Không gửi được câu hỏi, vui lòng thử lại.',
      )
      // Trả lại nội dung để người dùng không phải gõ lại.
      setDraft(trimmed)
    } finally {
      setSending(false)
    }
  }

  return (
    <div className="rounded-3xl border border-slate-100 bg-white p-6 shadow-sm sm:p-8">
      <button
        type="button"
        onClick={() => onNavigate('info')}
        className="mb-6 inline-flex items-center gap-2 rounded-xl bg-slate-100 px-4 py-2 text-sm font-medium text-slate-600 hover:bg-slate-200"
      >
        <LogOut size={16} /> Thoát
      </button>

      <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_320px]">
        {/* Conversation column */}
        <div className="order-2 flex flex-col lg:order-1">
          <div className="flex-1 space-y-4">
            <div className="flex items-center gap-2">
              <BotAvatar className="h-8 w-8" />
              <span className="font-semibold text-brand-600">
                Chatbot AI Tư Vấn Tài Chính
              </span>
            </div>

            {loading && (
              <p className="flex items-center gap-2 text-sm text-slate-400">
                <Loader2 size={16} className="animate-spin" /> Đang tải hội thoại...
              </p>
            )}

            {!loading && messages.length === 0 && (
              <p className="text-sm text-slate-400">
                Chưa có hội thoại nào. Hãy chọn một gợi ý bên dưới hoặc đặt câu hỏi
                cho AI.
              </p>
            )}

            {messages.map((m) =>
              m.role === 'ai' ? (
                <div key={m.id} className="flex gap-3">
                  <BotAvatar className="h-8 w-8" />
                  <div className="rounded-2xl rounded-tl-sm bg-slate-50 p-4">
                    <p className="whitespace-pre-line text-sm leading-relaxed text-slate-700">
                      {m.content}
                    </p>
                    <button
                      type="button"
                      onClick={() => speak(m.content)}
                      className="mt-3 inline-flex items-center gap-1.5 rounded-lg border border-slate-200 bg-white px-3 py-1.5 text-xs font-medium text-slate-500 hover:bg-slate-50"
                    >
                      <Volume2 size={14} /> Đọc bằng giọng nói
                    </button>
                  </div>
                </div>
              ) : (
                <div key={m.id} className="flex justify-end">
                  <div className="max-w-md rounded-2xl rounded-tr-sm bg-brand-600 p-4 text-sm font-medium text-white">
                    {m.content}
                  </div>
                </div>
              ),
            )}

            {sending && (
              <div className="flex gap-3">
                <BotAvatar className="h-8 w-8" />
                <div className="flex items-center gap-2 rounded-2xl rounded-tl-sm bg-slate-50 px-4 py-3 text-sm text-slate-400">
                  <Loader2 size={16} className="animate-spin" /> AI đang soạn câu trả
                  lời...
                </div>
              </div>
            )}

            {error && (
              <div className="flex items-start gap-2 rounded-2xl border border-rose-200 bg-rose-50 p-3 text-sm text-rose-700">
                <AlertCircle size={16} className="mt-0.5 shrink-0" />
                <p>{error}</p>
              </div>
            )}
          </div>

          {/* Quick actions */}
          <div className="mt-6 flex flex-wrap gap-3">
            {QUICK.map(({ label, tone, iconTone }) => (
              <button
                key={label}
                type="button"
                onClick={() => send(label)}
                disabled={sending}
                className={`inline-flex items-center gap-2 rounded-xl px-4 py-2.5 text-sm font-semibold transition disabled:cursor-not-allowed disabled:opacity-60 ${tone}`}
              >
                <Lightbulb size={16} className={iconTone} />
                {label}
              </button>
            ))}
          </div>

          {/* Composer */}
          <form
            onSubmit={(e) => {
              e.preventDefault()
              send(draft)
            }}
            className="mt-3 flex items-center gap-3"
          >
            <button
              type="button"
              className="grid h-11 w-11 shrink-0 place-items-center rounded-full bg-slate-100 text-slate-500 hover:bg-slate-200"
              title="Nhập bằng giọng nói"
            >
              <Mic size={18} />
            </button>
            <input
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              placeholder="Đặt câu hỏi tài chính cho gia đình bạn..."
              className="flex-1 rounded-xl border border-slate-200 bg-white px-4 py-3 text-sm focus:border-brand-500 focus:outline-none focus:ring-2 focus:ring-brand-100"
            />
            <button
              type="submit"
              disabled={sending}
              className="inline-flex items-center gap-2 rounded-xl bg-brand-500 px-5 py-3 text-sm font-semibold text-white hover:bg-brand-600 disabled:cursor-not-allowed disabled:opacity-60"
            >
              Gửi
              {sending ? (
                <Loader2 size={16} className="animate-spin" />
              ) : (
                <Send size={16} />
              )}
            </button>
          </form>
        </div>

        {/* Profile card */}
        <div className="order-1 lg:order-2">
          <div className="rounded-2xl border border-slate-200 p-5">
            <div className="mb-3 flex items-center justify-between">
              <h3 className="font-bold text-slate-800">Hồ sơ gia đình</h3>
              <button
                onClick={() => onNavigate('info')}
                className="inline-flex items-center gap-1.5 text-sm font-semibold text-brand-600 hover:underline"
              >
                Sửa <Pencil size={14} />
              </button>
            </div>
            <ProfileRow label="Đại diện" value={profile.name || '—'} />
            <ProfileRow label="Thu nhập" value={`${dong(profile.income)}/tháng`} />
            <ProfileRow label="Chi tiêu mỗi tháng" value={dong(profile.spending)} />
            <ProfileRow label="Tiết kiệm" value={dong(profile.savings)} />
            <ProfileRow label="Dư nợ" value={dong(profile.debt)} />
            <ProfileRow label="Thành viên" value={memberText} />
            <ProfileRow label="Nhu cầu" value={need} />
          </div>
          <img
            src="/img4.jpg"
            alt="Minh họa gia đình đi dã ngoại"
            className="mt-4 h-auto w-full rounded-2xl"
          />
        </div>
      </div>
    </div>
  )
}
