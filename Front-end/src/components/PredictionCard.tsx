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
 * "Chẩn đoán hồ sơ". Hồ sơ chưa được chẩn đoán (backend trả 404) vẫn
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

  const tone = LABEL_TONE[prediction.prediction]
  const { confidence, low_confidence, probabilities } = prediction.model_confidence

  return (
    <section className="mb-6 rounded-2xl border border-slate-200 p-5">
      <div className="mb-4 flex items-center gap-2">
        <BrainCircuit size={22} className="text-brand-500" />
        <h4 className="flex-1 font-bold text-slate-800">Nhóm định hướng tài chính</h4>
        <span className="text-xs text-slate-400">{prediction.model_version}</span>
      </div>

      {/*
        ML01 là phân loại ĐƠN NHÃN: kết quả đúng một nhóm. Nhãn thắng phải nổi
        bật hẳn so với mọi thứ khác trong thẻ — trước đây nó là một badge nhỏ
        đặt ngang hàng với bốn thanh xác suất, và người xem đọc thành "model
        trả về bốn kết quả".
      */}
      <div className={`rounded-xl border p-4 ${tone.badge}`}>
        <p className="text-xs font-semibold uppercase tracking-wider opacity-70">
          {prediction.prediction}
        </p>
        <p className="mt-1 text-xl font-bold leading-tight">
          {prediction.prediction_vi}
        </p>
      </div>

      {/*
        Xác suất dưới ngưỡng thì phải nói ra. Hiển thị nhãn như một kết luận
        chắc chắn trong khi model đang phân vân là chỗ người dùng bị dẫn sai.
      */}
      {low_confidence && (
        <p className="mt-3 flex items-start gap-2 rounded-xl border border-amber-200 bg-amber-50 p-3 text-sm text-amber-800">
          <AlertCircle size={16} className="mt-0.5 shrink-0" />
          Mô hình chưa đủ chắc chắn với hồ sơ này. Hãy xem kết quả như một gợi ý
          tham khảo và đối chiếu thêm với phần phân tích quy tắc bên dưới.
        </p>
      )}

      {/*
        Xác suất 4 lớp thu vào phần kỹ thuật, đóng sẵn. Giữ lại để soi model khi
        cần, nhưng không bày ra như bốn kết quả ngang nhau.
      */}
      <details className="mt-4 rounded-xl border border-slate-200">
        <summary className="cursor-pointer px-4 py-2.5 text-xs font-semibold text-slate-600">
          Chi tiết kỹ thuật · độ tin cậy {(confidence * 100).toFixed(1)}%
        </summary>
        <div className="space-y-2 border-t border-slate-200 px-4 py-3">
          <p className="text-xs text-slate-400">
            Phân bố xác suất của mô hình trên 4 nhóm. Nhóm có xác suất cao nhất
            chính là kết quả ở trên — đây không phải 4 kết quả dự đoán.
          </p>
          {probabilities.map((row) => (
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
      </details>

      <p className="mt-4 text-xs text-slate-400">
        Kết quả do mô hình học máy đưa ra, mang tính tham khảo — không thay thế
        tư vấn tài chính chuyên môn.
      </p>
    </section>
  )
}
