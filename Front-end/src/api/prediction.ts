import { apiGet } from '../lib/api'
import { guestQuery } from '../lib/guestSession'

/**
 * Kết quả ML01 — Financial Recommendation Group Classification.
 *
 * ML01 dự đoán hộ gia đình thuộc **nhóm định hướng tài chính** nào dựa trên
 * dữ liệu tài chính đầu vào. Đây KHÔNG phải điểm/mức độ sức khỏe tài chính —
 * việc đó do tầng rule RB02 làm với bộ nhãn riêng (EXCELLENT → CRITICAL).
 *
 * Backend gọi sang service Python, service này chạy XGBoost đã train sẵn.
 * Bốn nhãn xếp theo mức độ cấp thiết của việc cần làm, giảm dần.
 */
export type PredictionLabel =
  | 'EMERGENCY'
  | 'DEBT_FOCUS'
  | 'BUILD_BUFFER'
  | 'GROWTH'

export interface PredictionProbability {
  label: PredictionLabel
  label_vi: string
  probability: number
}

/** Số liệu kỹ thuật. Không trình bày như kết quả dự đoán. */
export interface ModelConfidence {
  confidence: number
  /** Xác suất cao nhất dưới ngưỡng tin cậy — phải nói ra, không hiển thị như kết luận chắc. */
  low_confidence: boolean
  /** Đủ 4 lớp, thứ tự cố định theo thang cấp thiết 🔴 → 🟢 nên biểu đồ không nhảy. */
  probabilities: PredictionProbability[]
}

export interface Prediction {
  /**
   * Output nghiệp vụ: ĐÚNG MỘT nhóm định hướng. ML01 là phân loại đơn nhãn —
   * bốn xác suất trong `model_confidence` là bên trong phép chọn nhãn này,
   * không phải bốn kết quả song song.
   */
  prediction: PredictionLabel
  prediction_vi: string
  model_confidence: ModelConfidence
  model_version: string
}

/** Màu theo mức độ cấp thiết, dùng chung cho badge và thanh xác suất. */
export const LABEL_TONE: Record<PredictionLabel, { badge: string; bar: string }> = {
  EMERGENCY: { badge: 'bg-red-100 text-red-700 border-red-200', bar: 'bg-red-500' },
  DEBT_FOCUS: { badge: 'bg-orange-100 text-orange-700 border-orange-200', bar: 'bg-orange-500' },
  BUILD_BUFFER: { badge: 'bg-amber-100 text-amber-700 border-amber-200', bar: 'bg-amber-500' },
  GROWTH: { badge: 'bg-emerald-100 text-emerald-700 border-emerald-200', bar: 'bg-emerald-500' },
}

/**
 * Backend trả 422 khi hồ sơ thiếu năm sinh (model bắt buộc có tuổi),
 * 503 khi service ML chưa cấu hình hoặc không phản hồi.
 */
export const getPrediction = (householdId: number) =>
  apiGet<Prediction>(`/households/${householdId}/prediction${guestQuery()}`)
