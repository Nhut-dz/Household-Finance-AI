import { useEffect, useState } from 'react'
import { AlertCircle, ArrowRight, FileText, Loader2, ShieldCheck } from 'lucide-react'
import { ApiError } from '../lib/api'
import { RISK_TONE, getLoanRisk, type LoanRisk } from '../api/loanRisk'
import type { PageKey } from '../data/profile'

/** 0–1 → "3,8%" theo kiểu Việt Nam, khớp với cách chatbot viết. */
const percent = (value: number) => `${(value * 100).toFixed(1).replace('.', ',')}%`

/**
 * Rủi ro khoản vay do model ML02 ước lượng — thẻ song song với PredictionCard
 * (ML01) trên màn "Chẩn đoán hồ sơ".
 *
 * Tự gọi API như PredictionCard, vì cùng lý do: kết quả model độc lập với
 * "Chẩn đoán hồ sơ" đã lưu. Khác ở một trạng thái: phần lớn hộ KHÔNG khai
 * khoản vay, và đó không phải lỗi — thẻ hiện lối sang màn nhập thay vì một
 * hộp đỏ.
 */
export default function LoanRiskCard({
  householdId,
  onNavigate,
}: {
  householdId: number | null
  onNavigate: (page: PageKey) => void
}) {
  const [risk, setRisk] = useState<LoanRisk | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  /** Backend trả 422 với field `loan_application`: hộ chưa khai, chưa chạy model. */
  const [undeclared, setUndeclared] = useState(false)

  useEffect(() => {
    if (householdId === null) {
      setRisk(null)
      return
    }

    setLoading(true)
    setError(null)
    setUndeclared(false)
    getLoanRisk(householdId)
      .then(setRisk)
      .catch((err) => {
        setRisk(null)
        if (err instanceof ApiError && err.status === 422 && err.fieldErrors.loan_application) {
          setUndeclared(true)
          return
        }
        setError(
          err instanceof ApiError
            ? err.message
            : 'Không lấy được kết quả ước lượng rủi ro khoản vay.',
        )
      })
      .finally(() => setLoading(false))
  }, [householdId])

  if (householdId === null) return null

  if (loading) {
    return (
      <section className="mb-6 flex items-center gap-3 rounded-2xl border border-slate-200 p-5 text-sm text-slate-500">
        <Loader2 size={18} className="animate-spin text-brand-500" />
        Đang ước lượng rủi ro khoản vay bằng mô hình ML02…
      </section>
    )
  }

  if (undeclared) {
    return (
      <section className="mb-6 rounded-2xl border border-slate-200 p-5">
        <div className="mb-3 flex items-center gap-2">
          <ShieldCheck size={22} className="text-brand-500" />
          <h4 className="flex-1 font-bold text-slate-800">Rủi ro khoản vay</h4>
        </div>
        <p className="text-sm text-slate-500">
          Hồ sơ chưa có thông tin khoản vay nên mô hình chưa ước lượng được. Khai
          khoản vay đang cân nhắc để xem mức rủi ro và xác suất gặp khó khăn trả nợ.
        </p>
        <button
          type="button"
          onClick={() => onNavigate('loan')}
          className="mt-3 inline-flex items-center gap-2 rounded-xl bg-brand-600 px-4 py-2 text-sm font-semibold text-white hover:bg-brand-700"
        >
          <FileText size={16} /> Nhập thông tin khoản vay <ArrowRight size={16} />
        </button>
      </section>
    )
  }

  if (error !== null) {
    return (
      <section className="mb-6 flex items-start gap-2 rounded-2xl border border-amber-200 bg-amber-50 p-4 text-sm text-amber-800">
        <AlertCircle size={18} className="mt-0.5 shrink-0" />
        <span>{error}</span>
      </section>
    )
  }

  if (risk === null) return null

  const tone = RISK_TONE[risk.prediction]
  const { confidence, low_confidence, threshold, description, probabilities } =
    risk.model_confidence

  return (
    <section className="mb-6 rounded-2xl border border-slate-200 p-5">
      <div className="mb-4 flex items-center gap-2">
        <ShieldCheck size={22} className="text-brand-500" />
        <h4 className="flex-1 font-bold text-slate-800">Rủi ro khoản vay</h4>
        <span className="text-xs text-slate-400">{risk.model_version}</span>
      </div>

      {/*
        ML02 là phân loại nhị phân: kết quả đúng một mức. Xác suất đi kèm ngay
        dưới nhãn vì đó là con số nhãn được cắt ra từ — tách nó xuống phần kỹ
        thuật thì "Rủi ro thấp" trở thành một lời khẳng định không có độ đo.
      */}
      <div className={`rounded-xl border p-4 ${tone.badge}`}>
        <p className="text-xs font-semibold uppercase tracking-wider opacity-70">
          {risk.prediction}
        </p>
        <p className="mt-1 text-xl font-bold leading-tight">{risk.prediction_vi}</p>
        <p className="mt-2 text-sm opacity-90">
          Xác suất gặp khó khăn trả nợ ước tính:{' '}
          <span className="font-bold tabular-nums">{percent(risk.risk_probability)}</span>
          {description && <span> — {description}</span>}
        </p>
      </div>

      {/*
        Sát ngưỡng quyết định thì phải nói ra: đổi một chút dữ liệu là đổi
        nhãn, và hiển thị nhãn như kết luận chắc là chỗ người dùng bị dẫn sai.
      */}
      {low_confidence && (
        <p className="mt-3 flex items-start gap-2 rounded-xl border border-amber-200 bg-amber-50 p-3 text-sm text-amber-800">
          <AlertCircle size={16} className="mt-0.5 shrink-0" />
          Hồ sơ nằm sát ngưỡng phân loại của mô hình nên kết quả này chưa chắc
          chắn. Hãy xem như một gợi ý tham khảo và đối chiếu với phần đánh giá
          khả năng vay theo quy tắc.
        </p>
      )}

      <details className="mt-4 rounded-xl border border-slate-200">
        <summary className="cursor-pointer px-4 py-2.5 text-xs font-semibold text-slate-600">
          Chi tiết kỹ thuật · ngưỡng phân loại {percent(threshold)} · độ tin cậy{' '}
          {percent(confidence)}
        </summary>
        <div className="space-y-2 border-t border-slate-200 px-4 py-3">
          <p className="text-xs text-slate-400">
            Mô hình ước lượng xác suất gặp khó khăn trả nợ; từ {percent(threshold)}{' '}
            trở lên xếp vào rủi ro cao. Độ tin cậy là khoảng cách tới ngưỡng đó —
            đây không phải hai kết quả dự đoán.
          </p>
          {probabilities.map((row) => (
            <div key={row.label} className="flex items-center gap-3">
              <span className="w-52 shrink-0 text-xs text-slate-500">{row.label_vi}</span>
              <div className="h-2 flex-1 overflow-hidden rounded-full bg-slate-100">
                <div
                  className={`h-full rounded-full ${RISK_TONE[row.label].bar}`}
                  style={{ width: `${Math.max(row.probability * 100, 0.5)}%` }}
                />
              </div>
              <span className="w-12 shrink-0 text-right text-xs tabular-nums text-slate-600">
                {percent(row.probability)}
              </span>
            </div>
          ))}
        </div>
      </details>

      <p className="mt-4 text-xs text-slate-400">
        Đây là ước lượng tham khảo dựa trên dữ liệu bạn tự khai, không phải kết quả
        thẩm định và không thay thế quyết định của tổ chức tín dụng.
      </p>
    </section>
  )
}
