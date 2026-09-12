import { apiGet } from '../lib/api'
import { guestQuery } from '../lib/guestSession'

/**
 * Kết quả ML02 — Credit Risk: rủi ro của khoản vay hộ đang xét.
 *
 * Cùng khuôn với `prediction.ts` (ML01): output nghiệp vụ là ĐÚNG MỘT nhãn,
 * phần kỹ thuật tách riêng. Khác ở chỗ ML02 là bài toán nhị phân có ngưỡng
 * quyết định riêng — `risk_probability` là xác suất gặp khó khăn trả nợ, và
 * nhãn được cắt từ con số đó theo `model_confidence.threshold`.
 *
 * Backend đọc bản ghi ở màn "Thông tin khoản vay"; chưa khai thì trả 422 với
 * field `loan_application` — thẻ hiện nút sang màn nhập thay vì báo lỗi.
 */
export type RiskLabel = 'LOW_RISK' | 'HIGH_RISK'

export interface RiskProbability {
  label: RiskLabel
  label_vi: string
  probability: number
}

/** Số liệu kỹ thuật. Không trình bày như kết quả dự đoán. */
export interface RiskModelConfidence {
  /**
   * KHÔNG phải xác suất của nhãn thắng như ML01: đây là khoảng cách tới ngưỡng
   * quyết định (0–1). Hồ sơ sát ngưỡng đổi chút dữ liệu là đổi nhãn.
   */
  confidence: number
  low_confidence: boolean
  /** Ngưỡng cắt LOW/HIGH đã chốt khi huấn luyện — không phải 0,5. */
  threshold: number
  /** Mức xác suất bằng chữ, ví dụ "khả năng rất thấp". */
  description: string
  /** Hai lớp, thứ tự cố định HIGH_RISK → LOW_RISK. */
  probabilities: RiskProbability[]
}

export interface LoanRisk {
  prediction: RiskLabel
  prediction_vi: string
  /** Xác suất gặp khó khăn trả nợ ước tính, 0–1. */
  risk_probability: number
  model_confidence: RiskModelConfidence
  model_version: string
}

/** Màu theo mức rủi ro, dùng chung cho badge và thanh xác suất. */
export const RISK_TONE: Record<RiskLabel, { badge: string; bar: string }> = {
  HIGH_RISK: { badge: 'bg-red-100 text-red-700 border-red-200', bar: 'bg-red-500' },
  LOW_RISK: { badge: 'bg-emerald-100 text-emerald-700 border-emerald-200', bar: 'bg-emerald-500' },
}

/**
 * 422 khi hộ chưa khai khoản vay (field `loan_application`), 503 khi service
 * ML chưa cấu hình hoặc không phản hồi.
 */
export const getLoanRisk = (householdId: number) =>
  apiGet<LoanRisk>(`/households/${householdId}/loan-risk${guestQuery()}`)
