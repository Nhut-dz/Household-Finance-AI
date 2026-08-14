import { useEffect, useState } from 'react'
import {
  AlertCircle,
  ArrowLeft,
  ArrowRight,
  BadgeDollarSign,
  CheckCircle2,
  Download,
  Loader2,
  PiggyBank,
  Sparkles,
  Trash2,
  Pencil,
} from 'lucide-react'
import type { PageKey } from '../data/profile'
import { dong } from '../lib/format'
import { getProposal, type Proposal } from '../api/proposal'
import { deleteHousehold } from '../api/households'
import PredictionCard from '../components/PredictionCard'
import { ApiError } from '../lib/api'

/** "—" khi backend chưa có số liệu cho ô đó. */
const money = (value: number | null | undefined) =>
  value === null || value === undefined ? '—' : dong(value)

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-xl border border-brand-100 bg-brand-50/60 px-4 py-3">
      <p className="text-xs text-slate-500">{label}</p>
      <p className="mt-0.5 font-bold text-brand-700">{value}</p>
    </div>
  )
}

function PlanCard({
  index,
  title,
  icon: Icon,
  rows,
  note,
}: {
  index: number
  title: string
  icon: typeof PiggyBank
  rows: { label: string; value: React.ReactNode }[]
  note: string | null
}) {
  return (
    <div className="flex flex-col rounded-2xl border border-slate-200 p-5">
      <div className="mb-4 flex items-center gap-2">
        <span className="grid h-7 w-7 place-items-center rounded-lg bg-brand-600 text-sm font-bold text-white">
          {index}
        </span>
        <h4 className="flex-1 font-bold text-slate-800">{title}</h4>
        <Icon size={22} className="text-brand-500" />
      </div>
      <div className="flex-1 space-y-2.5">
        {rows.map(({ label, value }) => (
          <div
            key={label}
            className="flex items-center justify-between border-t border-slate-100 pt-2.5 first:border-t-0 first:pt-0"
          >
            <span className="text-sm text-slate-400">{label}</span>
            <span className="text-sm font-semibold text-slate-800">{value}</span>
          </div>
        ))}
      </div>
      {note && (
        <p className="mt-4 border-t border-slate-100 pt-3 text-xs text-slate-400">
          {note}
        </p>
      )}
    </div>
  )
}

function RoadmapCard({ index, text }: { index: number; text: string }) {
  return (
    <div className="flex gap-3 rounded-2xl border border-slate-200 p-4">
      <span className="grid h-7 w-7 shrink-0 place-items-center rounded-full bg-brand-100 text-sm font-bold text-brand-700">
        {index}
      </span>
      <p className="text-sm text-slate-600">{text}</p>
    </div>
  )
}

function EmptyState({
  message,
  onNavigate,
}: {
  message: string
  onNavigate: (page: PageKey) => void
}) {
  return (
    <div className="grid place-items-center rounded-3xl border border-slate-100 bg-white px-6 py-20 text-center shadow-sm">
      <Sparkles size={40} className="text-brand-500" />
      <h2 className="mt-4 text-2xl font-extrabold text-slate-800">
        Chưa có dữ liệu chẩn đoán hồ sơ
      </h2>
      <p className="mt-3 max-w-md text-sm font-medium text-slate-500">{message}</p>
      <button
        onClick={() => onNavigate('info')}
        className="mt-6 inline-flex items-center gap-2 rounded-xl bg-brand-600 px-6 py-3 text-sm font-semibold text-white hover:bg-brand-700"
      >
        <ArrowRight size={16} /> Nhập thông tin mới
      </button>
    </div>
  )
}

export default function ProposalPage({
  householdId,
  onNavigate,
  onCleared,
}: {
  householdId: number | null
  onNavigate: (page: PageKey) => void
  onCleared: () => void
}) {
  const [proposal, setProposal] = useState<Proposal | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [clearing, setClearing] = useState(false)

  useEffect(() => {
    if (householdId === null) {
      setProposal(null)
      return
    }

    setLoading(true)
    setError(null)
    getProposal(householdId)
      .then(setProposal)
      .catch((err) => {
        setProposal(null)
        // 404 = hồ sơ chưa được chẩn đoán, là trạng thái bình thường.
        if (!(err instanceof ApiError) || err.status !== 404) {
          setError(
            err instanceof ApiError
              ? err.message
              : 'Không tải được chẩn đoán hồ sơ.',
          )
        }
      })
      .finally(() => setLoading(false))
  }, [householdId])

  const handleClear = async () => {
    if (householdId === null) return

    setClearing(true)
    try {
      await deleteHousehold(householdId)
      setProposal(null)
      onCleared()
    } catch (err) {
      setError(
        err instanceof ApiError
          ? err.message
          : 'Không xoá được dữ liệu, vui lòng thử lại.',
      )
    } finally {
      setClearing(false)
    }
  }

  if (loading) {
    return (
      <div className="grid place-items-center rounded-3xl border border-slate-100 bg-white px-6 py-20 text-center shadow-sm">
        <Loader2 size={32} className="animate-spin text-brand-500" />
        <p className="mt-4 text-sm font-medium text-slate-500">
          Đang tải chẩn đoán hồ sơ...
        </p>
      </div>
    )
  }

  if (householdId === null) {
    return (
      <EmptyState
        message="Vui lòng nhập thông tin hộ gia đình để AI phân tích và chẩn đoán hồ sơ tài chính."
        onNavigate={onNavigate}
      />
    )
  }

  if (proposal === null) {
    return (
      <div className="space-y-4">
        {/* Chưa có chẩn đoán không có nghĩa là chưa phân loại được. */}
        <div className="rounded-3xl border border-slate-100 bg-white p-6 shadow-sm sm:p-8">
          <PredictionCard householdId={householdId} />
        </div>
        {error && (
          <div className="flex items-start gap-2 rounded-2xl border border-rose-200 bg-rose-50 p-4 text-sm text-rose-700">
            <AlertCircle size={18} className="mt-0.5 shrink-0" />
            <p>{error}</p>
          </div>
        )}
        <EmptyState
          message="Hồ sơ của bạn chưa được chẩn đoán. Hãy trò chuyện với AI hoặc cập nhật thông tin để hệ thống phân tích lại."
          onNavigate={onNavigate}
        />
      </div>
    )
  }

  const { overview, savings_plan, investment_plan, loan_plan, roadmap } = proposal
  // Đánh số thẻ theo những gói thực sự có dữ liệu.
  let planIndex = 0

  return (
    <div className="rounded-3xl border border-slate-100 bg-white p-6 shadow-sm sm:p-8">
      {/* Header */}
      <div className="mb-6 flex flex-wrap items-center gap-3">
        <button
          onClick={() => onNavigate('chatbot')}
          className="inline-flex items-center gap-2 rounded-xl bg-slate-100 px-4 py-2 text-sm font-medium text-slate-600 hover:bg-slate-200"
        >
          <ArrowLeft size={16} /> Quay lại
        </button>
        <button
          onClick={() => window.print()}
          className="inline-flex items-center gap-2 rounded-xl bg-brand-600 px-4 py-2 text-sm font-semibold text-white hover:bg-brand-700"
        >
          <Download size={16} /> Xuất PDF
        </button>
        <h1 className="flex-1 text-center text-xl font-extrabold text-slate-800">
          Chẩn đoán hồ sơ tài chính
        </h1>
        <button
          onClick={handleClear}
          disabled={clearing}
          className="inline-flex items-center gap-2 rounded-xl border border-rose-200 bg-rose-50 px-4 py-2 text-sm font-semibold text-rose-600 hover:bg-rose-100 disabled:opacity-60"
        >
          {clearing ? (
            <Loader2 size={16} className="animate-spin" />
          ) : (
            <Trash2 size={16} />
          )}
          Xóa dữ liệu
        </button>
        <span className="inline-flex items-center gap-2 rounded-xl border border-emerald-200 bg-emerald-50 px-4 py-2 text-sm font-semibold text-emerald-600">
          <CheckCircle2 size={16} /> AI đã phân tích
        </span>
      </div>

      {error && (
        <div className="mb-4 flex items-start gap-2 rounded-2xl border border-rose-200 bg-rose-50 p-4 text-sm text-rose-700">
          <AlertCircle size={18} className="mt-0.5 shrink-0" />
          <p>{error}</p>
        </div>
      )}

      {/* Overview */}
      <PredictionCard householdId={householdId} />

      <section className="rounded-2xl border border-slate-200 p-5">
        <span className="inline-block rounded-lg bg-brand-600 px-3 py-1.5 text-sm font-bold text-white">
          TỔNG QUAN TƯ VẤN
        </span>
        {proposal.summary && (
          <p className="mt-3 whitespace-pre-line text-sm text-slate-600">
            {proposal.summary}
          </p>
        )}
        <div className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          <Stat
            label="Thu nhập ròng"
            value={
              overview.net_income === null
                ? '—'
                : `${dong(overview.net_income)}/tháng`
            }
          />
          <Stat label="Nợ hiện tại" value={money(overview.current_debt)} />
          <Stat label="Tiết kiệm hiện có" value={money(overview.current_savings)} />
          <Stat
            label="Nhu cầu tích lũy tài chính"
            value={money(overview.target_accumulation)}
          />
        </div>
      </section>

      {/* Plans */}
      <div className="mt-6 grid gap-4 lg:grid-cols-3">
        {savings_plan && (
          <PlanCard
            index={++planIndex}
            title="Tiết kiệm tích lũy"
            icon={PiggyBank}
            rows={[
              {
                label: 'Góp hàng tháng',
                value: money(savings_plan.monthly_contribution),
              },
              {
                label: 'Kỳ hạn đề xuất',
                value:
                  savings_plan.term_months === null
                    ? '—'
                    : `${savings_plan.term_months} tháng`,
              },
              {
                label: 'Lãi suất tham khảo',
                value:
                  savings_plan.interest_rate === null
                    ? '—'
                    : `${savings_plan.interest_rate}%/năm`,
              },
            ]}
            note={savings_plan.note}
          />
        )}

        {investment_plan && (
          <PlanCard
            index={++planIndex}
            title="Danh mục đầu tư"
            icon={BadgeDollarSign}
            rows={[
              { label: 'Phân bổ vốn', value: investment_plan.allocation ?? '—' },
              {
                label: 'Mức rủi ro',
                value: investment_plan.risk_level ? (
                  <span className="rounded-md bg-slate-100 px-2 py-0.5 text-xs">
                    {investment_plan.risk_level}
                  </span>
                ) : (
                  '—'
                ),
              },
            ]}
            note={investment_plan.note}
          />
        )}

        {loan_plan && (
          <PlanCard
            index={++planIndex}
            title="Gói vay ưu đãi"
            icon={BadgeDollarSign}
            rows={[
              { label: 'Loại sản phẩm', value: loan_plan.product ?? '—' },
              { label: 'Hạn mức ước tính', value: money(loan_plan.credit_limit) },
              {
                label: 'Trả góp tháng',
                value:
                  loan_plan.monthly_payment === null
                    ? '—'
                    : `${dong(loan_plan.monthly_payment)}/tháng`,
              },
            ]}
            note={loan_plan.note}
          />
        )}
      </div>

      {/* Roadmap */}
      {roadmap.length > 0 && (
        <section className="mt-6 rounded-2xl border border-slate-200 p-5">
          <span className="inline-block rounded-lg bg-brand-600 px-3 py-1.5 text-sm font-bold text-white">
            LỘ TRÌNH KHUYẾN NGHỊ CHO GIA ĐÌNH
          </span>
          <div className="mt-4 grid gap-3 lg:grid-cols-3">
            {roadmap.map(({ step, text }) => (
              <RoadmapCard key={step} index={step} text={text} />
            ))}
          </div>
        </section>
      )}

      {/* Footer actions */}
      <div className="mt-6 flex flex-wrap items-center gap-3">
        <button
          onClick={() => onNavigate('chatbot')}
          className="flex flex-1 items-center justify-center gap-2 rounded-2xl bg-brand-600 py-4 text-base font-bold text-white hover:bg-brand-700"
        >
          Hỏi Chatbot để nhận tư vấn chuyên sâu <ArrowRight size={18} />
        </button>
        <button
          onClick={() => window.print()}
          className="inline-flex items-center gap-2 rounded-2xl bg-brand-600 px-5 py-4 text-sm font-semibold text-white hover:bg-brand-700"
        >
          <Download size={16} /> Xuất PDF
        </button>
        <button
          onClick={() => onNavigate('info')}
          className="inline-flex items-center gap-2 rounded-2xl bg-slate-100 px-5 py-4 text-sm font-semibold text-slate-600 hover:bg-slate-200"
        >
          <Pencil size={16} /> Sửa hồ sơ
        </button>
      </div>
    </div>
  )
}
