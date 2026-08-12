import { useEffect, useState } from 'react'
import { AlertCircle, BrainCircuit, Loader2 } from 'lucide-react'
import { ApiError } from '../lib/api'
import {
  LABEL_TONE,
  getPrediction,
  type Prediction,
} from '../api/prediction'

/**
 * Nhóm khuyến nghị do model ML01 phân loại.
 *
 * Thẻ tự gọi API chứ không nhận dữ liệu từ trang cha: dự đoán độc lập với
 * "phương án đề xuất". Hồ sơ chưa được tính phương án (backend trả 404) vẫn
 * phải xem được nhóm của mình, nên hai thứ không dùng chung một lần tải.
 */
export default function PredictionCard({ householdId }: { householdId: number | null }) {
  const [prediction, setPrediction] = useState<Prediction | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (householdId === null) {
      setPrediction(null)
      return
    }

    setLoading(true)
    setError(null)
    getPrediction(householdId)
      .then(setPrediction)
      .catch((err) => {
        setPrediction(null)
        setError(
          err instanceof ApiError
            ? err.message
            : 'Không lấy được kết quả phân loại.',
        )
      })
      .finally(() => setLoading(false))
  }, [householdId])

  if (householdId === null) return null

  if (loading) {
    return (
      <section className="mb-6 flex items-center gap-3 rounded-2xl border border-slate-200 p-5 text-sm text-slate-500">
        <Loader2 size={18} className="animate-spin text-brand-500" />
        Đang phân loại hồ sơ bằng mô hình ML01…
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

  if (prediction === null) return null

  const tone = LABEL_TONE[prediction.label]

  return (
    <section className="mb-6 rounded-2xl border border-slate-200 p-5">
      <div className="mb-4 flex items-center gap-2">
        <BrainCircuit size={22} className="text-brand-500" />
        <h4 className="flex-1 font-bold text-slate-800">Nhóm khuyến nghị tài chính</h4>
        <span className="text-xs text-slate-400">{prediction.model_version}</span>
      </div>

      <div className="flex flex-wrap items-center gap-3">
        <span className={`rounded-lg border px-3 py-1.5 text-sm font-bold ${tone.badge}`}>
          {prediction.label_vi}
        </span>
        <span className="text-sm text-slate-500">
          Độ tin cậy {(prediction.confidence * 100).toFixed(1)}%
        </span>
      </div>

      {/*
        Xác suất dưới ngưỡng thì phải nói ra. Hiển thị nhãn như một kết luận
        chắc chắn trong khi model đang phân vân là chỗ người dùng bị dẫn sai.
      */}
      {prediction.low_confidence && (
        <p className="mt-3 flex items-start gap-2 rounded-xl border border-amber-200 bg-amber-50 p-3 text-sm text-amber-800">
          <AlertCircle size={16} className="mt-0.5 shrink-0" />
          Mô hình chưa đủ chắc chắn với hồ sơ này. Hãy xem kết quả như một gợi ý
          tham khảo và đối chiếu thêm với phần phân tích quy tắc bên dưới.
        </p>
      )}

      <div className="mt-4 space-y-2">
        {prediction.probabilities.map((row) => (
          <div key={row.label} className="flex items-center gap-3">
            <span className="w-52 shrink-0 text-xs text-slate-500">{row.label_vi}</span>
            <div className="h-2 flex-1 overflow-hidden rounded-full bg-slate-100">
              <div
                className={`h-full rounded-full ${LABEL_TONE[row.label].bar}`}
                style={{ width: `${Math.max(row.probability * 100, 0.5)}%` }}
              />
            </div>
            <span className="w-12 shrink-0 text-right text-xs tabular-nums text-slate-600">
              {(row.probability * 100).toFixed(1)}%
            </span>
          </div>
        ))}
      </div>

      <p className="mt-4 text-xs text-slate-400">
        Kết quả do mô hình học máy đưa ra, mang tính tham khảo — không thay thế
        tư vấn tài chính chuyên môn.
      </p>
    </section>
  )
}
