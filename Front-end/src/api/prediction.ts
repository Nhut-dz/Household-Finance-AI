import { apiGet } from '../lib/api'
import { guestQuery } from '../lib/guestSession'

/**
 * Kết quả phân loại của model ML01 (F03).
 *
 * Backend gọi sang service Python, service này chạy XGBoost đã train sẵn.
 * Bốn nhãn xếp theo mức độ nghiêm trọng giảm dần.
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

export interface Prediction {
  label: PredictionLabel
  label_vi: string
  confidence: number
  /** Đủ 4 lớp, thứ tự cố định theo thang mức độ 🔴 → 🟢 nên biểu đồ không nhảy. */
  probabilities: PredictionProbability[]
  /** Xác suất cao nhất dưới ngưỡng tin cậy — phải nói ra, không hiển thị như kết luận chắc. */
  low_confidence: boolean
  model_version: string
}

/** Màu theo mức độ, dùng chung cho badge và thanh xác suất. */
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
