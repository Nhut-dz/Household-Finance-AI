import { apiDelete, apiGet, apiPut } from '../lib/api'
import { getGuestSessionId, guestQuery } from '../lib/guestSession'
import {
  emptyLoanForm,
  type EducationLevel,
  type Gender,
  type LoanApplicationForm,
  type LoanPurpose,
  type MaritalStatus,
  type Occupation,
  type SavedPayment,
} from '../data/loan'

/** Body của PUT /households/{id}/loan-application. */
export interface StoreLoanApplicationPayload {
  borrower_age: number | null
  gender: string
  marital_status: string
  education_level: string
  occupation: string
  employment_years: number | null

  loan_amount: number
  loan_term_months: number | null
  interest_rate: number | null
  // Không gửi `monthly_payment`: backend tự tính từ ba trường trên và bỏ qua
  // mọi giá trị client đặt vào (StoreLoanApplicationRequest không còn rule cho
  // nó). Gửi kèm chỉ tạo ảo giác là FE quyết định được con số đó.
  asset_price: number
  loan_purpose: string

  previous_loan_count: number
  late_payment_count: number
  has_overdue_loan: boolean
  total_overdue_amount: number | null

  guest_session_id: string
}

/**
 * Bản ghi trả về từ LoanApplicationResource.
 *
 * Mỗi trường enum đi kèm `*_label` tiếng Việt do backend sinh. Dùng nhãn đó khi
 * hiển thị lại bản ghi đã lưu, thay vì tra bảng ở FE: bảng nhãn của
 * `data/loan.ts` chỉ tồn tại để dựng dropdown trước lần gọi API đầu tiên.
 */
export interface LoanApplicationResponse {
  id: number
  household_id: number

  borrower_age: number
  gender: Gender
  gender_label: string
  marital_status: MaritalStatus
  marital_status_label: string
  education_level: EducationLevel
  education_level_label: string
  occupation: Occupation
  occupation_label: string
  employment_years: number

  loan_amount: number
  loan_term_months: number
  /** `null` khi hộ chưa khai lãi suất. */
  interest_rate: number | null
  /** EMI backend tự tính. Chỉ để đọc — form không gửi trường này lên. */
  monthly_payment: number
  asset_price: number
  loan_purpose: LoanPurpose
  loan_purpose_label: string

  previous_loan_count: number
  late_payment_count: number
  has_overdue_loan: boolean
  total_overdue_amount: number

  /** loan_amount / asset_price, backend tính sẵn. Null khi giá tài sản = 0. */
  loan_to_value: number | null
  created_at: string
  updated_at: string
}

export function toStoreLoanApplicationPayload(
  form: LoanApplicationForm,
): StoreLoanApplicationPayload {
  return {
    borrower_age: form.borrower_age,
    gender: form.gender,
    marital_status: form.marital_status,
    education_level: form.education_level,
    occupation: form.occupation,
    employment_years: form.employment_years,

    loan_amount: form.loan_amount,
    loan_term_months: form.loan_term_months,
    interest_rate: form.interest_rate,
    asset_price: form.asset_price,
    loan_purpose: form.loan_purpose,

    previous_loan_count: form.previous_loan_count,
    late_payment_count: form.late_payment_count,
    has_overdue_loan: form.has_overdue_loan,
    // Bỏ chọn "có khoản vay quá hạn" thì gửi null, để backend khỏi phải phân
    // biệt "0 vì không có" với "0 vì người dùng gõ số 0".
    total_overdue_amount: form.has_overdue_loan ? form.total_overdue_amount : null,

    guest_session_id: getGuestSessionId(),
  }
}

/** Chuyển bản ghi của backend về đúng shape mà form đang dùng. */
export function fromLoanApplicationResponse(
  data: LoanApplicationResponse,
): LoanApplicationForm {
  return {
    ...emptyLoanForm,
    borrower_age: data.borrower_age,
    gender: data.gender,
    marital_status: data.marital_status,
    education_level: data.education_level,
    occupation: data.occupation,
    employment_years: Number(data.employment_years),

    loan_amount: Number(data.loan_amount),
    loan_term_months: data.loan_term_months,
    // Phải qua `== null` chứ không `Number(...)` thẳng: `Number(null)` là 0,
    // và 0 ở đây nghĩa là "vay không lãi" — một điều người dùng chưa hề khai.
    interest_rate: data.interest_rate == null ? null : Number(data.interest_rate),
    asset_price: Number(data.asset_price),
    loan_purpose: data.loan_purpose,

    previous_loan_count: data.previous_loan_count,
    late_payment_count: data.late_payment_count,
    has_overdue_loan: Boolean(data.has_overdue_loan),
    total_overdue_amount: Number(data.total_overdue_amount),
  }
}

/**
 * Tách phần "khoản trả hàng tháng đã lưu" khỏi bản ghi trả về.
 *
 * Đây là thứ DUY NHẤT của response không đi vào `LoanApplicationForm`: form
 * không sở hữu `monthly_payment` nữa, nhưng vẫn phải nhớ con số đã lưu để khỏi
 * hiện đè một giá trị tự tính lên bản ghi mà backend sẽ giữ nguyên.
 */
export const savedPaymentOf = (data: LoanApplicationResponse): SavedPayment => ({
  loan_amount: Number(data.loan_amount),
  loan_term_months: data.loan_term_months,
  interest_rate: data.interest_rate == null ? null : Number(data.interest_rate),
  monthly_payment: Number(data.monthly_payment),
})

/**
 * Lưu hoặc ghi đè phương án vay của hộ. PUT là idempotent nên FE không cần biết
 * trước hộ đã khai khoản vay hay chưa.
 */
export const saveLoanApplication = (
  householdId: number,
  form: LoanApplicationForm,
) =>
  apiPut<LoanApplicationResponse>(
    `/households/${householdId}/loan-application`,
    toStoreLoanApplicationPayload(form),
  )

/** Backend trả 404 khi hộ chưa khai khoản vay — trạng thái bình thường. */
export const getLoanApplication = (householdId: number) =>
  apiGet<LoanApplicationResponse>(
    `/households/${householdId}/loan-application${guestQuery()}`,
  )

export const deleteLoanApplication = (householdId: number) =>
  apiDelete<null>(`/households/${householdId}/loan-application${guestQuery()}`)
