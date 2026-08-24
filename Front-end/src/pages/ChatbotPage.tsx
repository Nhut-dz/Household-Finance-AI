import { useEffect, useState } from 'react'
import {
  AlertCircle,
  FileText,
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
import {
  getMessages,
  sendMessage,
  type ChatMessage,
  type IntentCode,
} from '../api/messages'
import { getLoanApplication } from '../api/loanApplication'
import { ApiError } from '../lib/api'

/** Đọc câu trả lời bằng giọng nói của trình duyệt (không cần API riêng). */
function speak(text: string) {
  if (!('speechSynthesis' in window)) return

  window.speechSynthesis.cancel()
  const utterance = new SpeechSynthesisUtterance(text)
  utterance.lang = 'vi-VN'
  window.speechSynthesis.speak(utterance)
}

/**
 * Bốn chip gợi ý (cập nhật 15/08/2026).
 *
 * Mỗi chip mang một `intent` gửi thẳng xuống engine — Hướng 1 của PLAN §8.2.
 * Trước đây chip chỉ là nhãn chữ và engine phải đoán ý định bằng từ khoá; cách
 * đó hỏng ngay với hai chức năng ML mới, vì "Chẩn đoán rủi ro vay vốn" chứa chữ
 * "vay" nên rơi vào nhánh hạn mức vay, còn "Chẩn đoán sức khỏe tài chính" không
 * chứa từ khoá nào nên rơi xuống nhánh trả lời chung. Cả hai vẫn trả lời trôi
 * chảy, chỉ là bằng nhánh sai.
 *
 * "Gói đầu tư" và "Gói vay mua nhà" đã rút khỏi nhóm này. Hai nhánh xử lý tương
 * ứng vẫn còn nguyên ở engine và người dùng vẫn gõ tay để dùng được.
 */
const QUICK: {
  label: string
  intent: IntentCode
  tone: string
  iconTone: string
}[] = [
  {
    label: 'Gói tiết kiệm',
    intent: 'SAVINGS_PACKAGE',
    tone: 'bg-brand-100 text-brand-700 hover:bg-brand-200',
    iconTone: 'text-amber-500 fill-amber-300',
  },
  {
    label: 'Chẩn đoán sức khỏe tài chính',
    intent: 'FINANCIAL_HEALTH_DIAGNOSIS',
    tone: 'bg-brand-600 text-white hover:bg-brand-700',
    iconTone: 'text-amber-300 fill-amber-300/40',
  },
  {
    label: 'Chẩn đoán rủi ro vay vốn',
    intent: 'LOAN_RISK_DIAGNOSIS',
    tone: 'bg-amber-400 text-white hover:bg-amber-500',
    iconTone: 'text-white fill-white/40',
  },
  {
    label: 'Quy tắc 50/30/20',
    intent: 'BUDGET_50_30_20',
    tone: 'bg-purple-600 text-white hover:bg-purple-700',
    iconTone: 'text-amber-300 fill-amber-300/40',
  },
]

/** App logo used as the assistant's avatar. */
function FormattedText({ content }: { content: string }) {
  const parseInline = (text: string) => {
    const parts = text.split(/(\*\*.*?\*\*|\*.*?\*|_.*?_)/g);
    return parts.map((part, idx) => {
      if (part.startsWith("**") && part.endsWith("**") && part.length > 4) {
        return (
          <strong key={idx} className="font-bold text-slate-900">
            {part.slice(2, -2)}
          </strong>
        );
      }
      if (
        (part.startsWith("*") && part.endsWith("*") && part.length > 2) ||
        (part.startsWith("_") && part.endsWith("_") && part.length > 2)
      ) {
        return (
          <em key={idx} className="italic text-slate-500">
            {part.slice(1, -1)}
          </em>
        );
      }
      return part;
    });
  };

  const lines = content.split("\n");

  return (
    <div className="space-y-1 text-sm leading-relaxed text-slate-700">
      {lines.map((line, lineIdx) => {
        const trimmed = line.trim();
        if (!trimmed) {
          return <div key={lineIdx} className="h-1" />;
        }
        if (trimmed.startsWith("- ")) {
          return (
            <div key={lineIdx} className="flex items-start gap-2 py-0.5 pl-1">
              <span className="mt-2 h-1.5 w-1.5 shrink-0 rounded-full bg-brand-500" />
              <div className="flex-1">{parseInline(trimmed.slice(2))}</div>
            </div>
          );
        }
        return <div key={lineIdx}>{parseInline(line)}</div>;
      })}
    </div>
  );
}

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
  chatResetToken,
  chatResetReason,
  onNavigate,
}: {
  profile: HouseholdProfile
  householdId: number | null
  /** Tăng khi backend xoay phiên; buộc effect nạp lại dù householdId không đổi. */
  chatResetToken: number
  /**
   * Vì sao màn chat trống. `rotated` = hồ sơ đổi nên mở phiên mới, phiên cũ vẫn
   * còn trong DB. `cleared` = người dùng bấm "Xóa dữ liệu", phiên cũ ĐÃ BỊ XOÁ
   * HẲN. Hai trường hợp phải nói hai câu khác nhau, nếu không là hứa với người
   * dùng rằng có thể xem lại thứ đã không còn.
   */
  chatResetReason: 'rotated' | 'cleared' | null
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
  /** Đang kiểm hộ đã khai thông tin khoản vay chưa, trước khi chạy ML02. */
  const [checkingLoan, setCheckingLoan] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Nạp hội thoại của PHIÊN ĐANG MỞ. Chạy lại khi đổi hồ sơ, và khi
  // `chatResetToken` tăng — tức backend vừa xoay phiên vì dữ liệu tài chính đổi.
  useEffect(() => {
    // Xoá ngay chứ không đợi request xong: giữ lại trong lúc chờ nghĩa là người
    // dùng vẫn đọc được lời khuyên tính trên số liệu cũ thêm một nhịp nữa.
    setMessages([])
    setError(null)

    if (householdId === null) return

    let cancelled = false
    setLoading(true)
    getMessages(householdId)
      .then((history) => {
        // Bỏ qua phản hồi của lần nạp đã bị thay thế: hai lần đổi hồ sơ liên
        // tiếp có thể để response cũ về sau và ghi đè hội thoại mới.
        if (cancelled) return
        setMessages(history.messages)
      })
      .catch(() => undefined)
      .finally(() => {
        if (!cancelled) setLoading(false)
      })

    return () => {
      cancelled = true
    }
  }, [householdId, chatResetToken])

  const send = async (text: string, intent?: IntentCode) => {
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
      // `conversation_id` trong phản hồi do server quyết định — FE không giữ và
      // không gửi lên, nên không thể ghi nhầm vào một phiên đã đóng.
      const sent = await sendMessage(householdId, trimmed, intent)
      setMessages((prev) => [...prev, sent.user_message, sent.ai_message])
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

  /**
   * Bấm một chip gợi ý.
   *
   * "Chẩn đoán rủi ro vay vốn" cần dữ liệu của màn *Thông tin khoản vay*, nên
   * hộ chưa khai thì đưa thẳng sang đó thay vì gửi một câu hỏi chỉ để nhận lại
   * lời nhắc — người dùng bấm nút xong mà vẫn phải tự đi tìm màn nhập là một
   * bước thừa.
   *
   * Backend vẫn kiểm lại điều kiện này và trả `requires_loan_application`. Kiểm
   * hai nơi ở đây KHÔNG phải chép luật nghiệp vụ: phía FE chỉ quyết định điều
   * hướng cho mượt, còn quyết định "có chạy model hay không" vẫn chỉ có một
   * chỗ, là engine.
   */
  const handleQuick = async (label: string, intent: IntentCode) => {
    if (intent !== 'LOAN_RISK_DIAGNOSIS' || householdId === null) {
      return send(label, intent)
    }

    setCheckingLoan(true)
    try {
      await getLoanApplication(householdId)
    } catch (err) {
      // 404 nghĩa là hộ chưa khai — trạng thái bình thường, không phải lỗi.
      // Lỗi khác (mất mạng, 500) thì cứ gửi, để engine trả lời cho thống nhất
      // thay vì FE tự đoán rồi điều hướng nhầm.
      if (err instanceof ApiError && err.status === 404) {
        setError(null)
        onNavigate('loan')
        return
      }
    } finally {
      setCheckingLoan(false)
    }

    await send(label, intent)
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

            {/*
              Sau khi hồ sơ tài chính đổi, hội thoại cũ bị dọn đi. Nói ra lý do —
              màn chat bỗng trống trơn mà không giải thích thì người dùng tưởng
              mất dữ liệu.
            */}
            {!loading && messages.length === 0 && chatResetReason === 'rotated' && (
              <p className="rounded-xl border border-brand-100 bg-brand-50 p-3 text-sm text-slate-600">
                Hồ sơ đã được cập nhật nên đây là một phiên trò chuyện mới. Hội
                thoại trước vẫn được lưu lại, nhưng AI sẽ phân tích lại từ đầu
                theo số liệu mới.
              </p>
            )}

            {/*
              Sau khi bấm "Xóa dữ liệu": KHÔNG nói "hội thoại trước vẫn được lưu
              lại" như trường hợp xoay phiên — ở đây phiên cũ đã bị xoá hẳn khỏi
              DB, không xem lại được nữa.
            */}
            {!loading && messages.length === 0 && chatResetReason === 'cleared' && (
              <p className="rounded-xl border border-rose-100 bg-rose-50 p-3 text-sm text-slate-600">
                Đã xoá toàn bộ dữ liệu và lịch sử trò chuyện. Đây là một phiên
                hoàn toàn mới — không còn thông tin nào từ lần trước được dùng
                lại.
              </p>
            )}

            {!loading && messages.length === 0 && chatResetReason === null && (
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
                    <FormattedText content={m.content} />


                    <div className="mt-3 flex flex-wrap items-center gap-2">
                      <button
                        type="button"
                        onClick={() => speak(m.content)}
                        className="inline-flex items-center gap-1.5 rounded-lg border border-slate-200 bg-white px-3 py-1.5 text-xs font-medium text-slate-500 hover:bg-slate-50"
                      >
                        <Volume2 size={14} /> Đọc bằng giọng nói
                      </button>

                      {/*
                        Engine báo thiếu dữ liệu khoản vay. Đưa luôn nút sang màn
                        nhập: bảo người dùng "vui lòng điền màn Thông tin khoản
                        vay" rồi để họ tự đi tìm là bỏ dở việc giữa chừng.
                      */}
                      {m.requires_loan_application && (
                        <button
                          type="button"
                          onClick={() => onNavigate('loan')}
                          className="inline-flex items-center gap-1.5 rounded-lg bg-brand-600 px-3 py-1.5 text-xs font-semibold text-white hover:bg-brand-700"
                        >
                          <FileText size={14} /> Nhập thông tin khoản vay
                        </button>
                      )}
                    </div>
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
            {QUICK.map(({ label, intent, tone, iconTone }) => (
              <button
                key={intent}
                type="button"
                onClick={() => handleQuick(label, intent)}
                disabled={sending || checkingLoan}
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
