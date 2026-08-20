/**
 * Dữ liệu màn "Thông tin khoản vay" — đầu vào của ML02 (Home Credit Risk).
 *
 * Tên trường giữ đúng tên field của API để khỏi phải dịch qua lại: FE và
 * backend gọi cùng một thứ bằng cùng một tên, và lỗi validate backend trả về
 * gắn thẳng được vào đúng ô trên form.
 *
 * Nhãn tiếng Việt của các giá trị enum được chép từ App\Enums phía backend.
 * Đây là bản sao có chủ đích: form phải dựng được dropdown trước khi gọi API
 * lần nào. Backend vẫn trả kèm `*_label` trong response, và đó mới là nguồn
 * dùng khi hiển thị lại bản ghi đã lưu.
 */

export type Gender = 'male' | 'female'

export type MaritalStatus =
  | 'single'
  | 'married'
  | 'civil_marriage'
  | 'separated'
  | 'widow'

export type EducationLevel =
  | 'lower_secondary'
  | 'secondary'
  | 'incomplete_higher'
  | 'higher'
  | 'academic_degree'

export type Occupation =
  | 'office_staff'
  | 'manager'
  | 'accountant'
  | 'it_staff'
  | 'teacher'
  | 'medical_staff'
  | 'sales_staff'
  | 'driver'
  | 'security_staff'
  | 'service_staff'
  | 'laborer'
  | 'farmer'
  | 'self_employed'
  | 'retired'
  | 'unemployed'
  | 'other'

export type LoanPurpose =
  | 'buy_house'
  | 'buy_land'
  | 'buy_car'
  | 'home_repair'
  | 'business'
  | 'education'
  | 'medical'
  | 'consumer'
  | 'debt_consolidation'
  | 'other'

/**
 * Trạng thái form. Các ô số để `null` khi chưa nhập chứ không phải 0 — "chưa
 * điền tuổi" và "0 tuổi" là hai chuyện khác nhau, và ô hiện số 0 sẵn thì người
 * dùng phải xoá đi mới gõ được.
 *
 * Ba ô của mục C mặc định 0 vì ở đó 0 là câu trả lời thật và phổ biến nhất:
 * chưa từng vay, chưa từng trả chậm.
 */
export interface LoanApplicationForm {
  // A. Thông tin người vay
  borrower_age: number | null
  gender: Gender | ''
  marital_status: MaritalStatus | ''
  children_count: number
  education_level: EducationLevel | ''
  occupation: Occupation | ''
  employment_years: number | null

  // B. Thông tin khoản vay
  loan_amount: number
  loan_term_months: number | null
  monthly_payment: number
  asset_price: number
  loan_purpose: LoanPurpose | ''

  // C. Lịch sử tín dụng
  previous_loan_count: number
  late_payment_count: number
  has_overdue_loan: boolean
  total_overdue_amount: number
}

export const emptyLoanForm: LoanApplicationForm = {
  borrower_age: null,
  gender: '',
  marital_status: '',
  children_count: 0,
  education_level: '',
  occupation: '',
  employment_years: null,

  loan_amount: 0,
  loan_term_months: null,
  monthly_payment: 0,
  asset_price: 0,
  loan_purpose: '',

  previous_loan_count: 0,
  late_payment_count: 0,
  has_overdue_loan: false,
  total_overdue_amount: 0,
}

export const GENDER_LABELS: Record<Gender, string> = {
  male: 'Nam',
  female: 'Nữ',
}

export const MARITAL_STATUS_LABELS: Record<MaritalStatus, string> = {
  single: 'Độc thân',
  married: 'Đã kết hôn',
  civil_marriage: 'Sống chung, chưa đăng ký kết hôn',
  separated: 'Ly thân, ly hôn',
  widow: 'Góa',
}

/** Thứ tự khai báo là thứ bậc học vấn từ thấp lên cao, không phải tuỳ ý. */
export const EDUCATION_LABELS: Record<EducationLevel, string> = {
  lower_secondary: 'Trung học cơ sở',
  secondary: 'Trung học phổ thông, trung cấp',
  incomplete_higher: 'Cao đẳng, đại học dở dang',
  higher: 'Đại học',
  academic_degree: 'Sau đại học',
}

export const OCCUPATION_LABELS: Record<Occupation, string> = {
  office_staff: 'Nhân viên văn phòng',
  manager: 'Quản lý, lãnh đạo',
  accountant: 'Kế toán, tài chính',
  it_staff: 'Công nghệ thông tin',
  teacher: 'Giáo viên, giảng viên',
  medical_staff: 'Y tế',
  sales_staff: 'Kinh doanh, bán hàng',
  driver: 'Lái xe',
  security_staff: 'Bảo vệ',
  service_staff: 'Dịch vụ, giúp việc',
  laborer: 'Công nhân, lao động phổ thông',
  farmer: 'Nông, lâm, ngư nghiệp',
  self_employed: 'Tự kinh doanh, tự do',
  retired: 'Nghỉ hưu',
  unemployed: 'Chưa có việc làm',
  other: 'Khác',
}

export const LOAN_PURPOSE_LABELS: Record<LoanPurpose, string> = {
  buy_house: 'Mua nhà, căn hộ',
  buy_land: 'Mua đất',
  buy_car: 'Mua xe',
  home_repair: 'Sửa chữa, xây dựng nhà',
  business: 'Kinh doanh, sản xuất',
  education: 'Học tập',
  medical: 'Chữa bệnh',
  consumer: 'Tiêu dùng, mua sắm',
  debt_consolidation: 'Trả nợ khoản vay khác',
  other: 'Khác',
}

/**
 * Kỳ hạn cho chọn. Giữ khớp `StoreLoanApplicationRequest::TERM_CHOICES` —
 * backend từ chối mọi giá trị ngoài danh sách này.
 */
export const LOAN_TERM_CHOICES = [12, 24, 36, 60, 120, 180, 240, 300] as const

/** `240` → `"20 năm (240 tháng)"`. Đọc bằng năm dễ hình dung hơn. */
export const loanTermLabel = (months: number) =>
  months % 12 === 0
    ? `${months / 12} năm (${months} tháng)`
    : `${months} tháng`

/** Chuyển `Record<key, label>` thành danh sách option cho <Select>. */
export const toOptions = <K extends string>(labels: Record<K, string>) =>
  (Object.keys(labels) as K[]).map((value) => ({ value, label: labels[value] }))

/**
 * Trả góp tối thiểu để trả hết GỐC trong kỳ hạn, chưa tính lãi. Backend chặn
 * mọi giá trị nhỏ hơn số này, nên form gợi ý sẵn thay vì để người dùng gửi lên
 * rồi mới bị trả về.
 */
export const minimumMonthlyPayment = (amount: number, termMonths: number | null) =>
  amount > 0 && termMonths ? Math.ceil(amount / termMonths) : 0
