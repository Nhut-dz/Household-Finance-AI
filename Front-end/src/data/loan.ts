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
  // KHÔNG có `children_count`: số con đã khai ở màn "Nhập thông tin" và thuộc
  // hồ sơ hộ gia đình. Hỏi lại ở đây là bắt người dùng gõ hai lần cùng một con
  // số, rồi để hai bản sao đó lệch nhau.
  education_level: EducationLevel | ''
  occupation: Occupation | ''
  employment_years: number | null

  // B. Thông tin khoản vay
  //
  // KHÔNG có `monthly_payment`: từ khi có lãi suất, khoản trả hàng tháng là
  // giá trị SUY RA chứ không phải trạng thái form. Giữ nó ở đây thì sẽ có hai
  // nguồn sự thật cho cùng một con số — cái người dùng thấy và cái backend
  // tính — và chúng chỉ cần lệch nhau một lần là đủ gây khó hiểu.
  loan_amount: number
  loan_term_months: number | null
  /** Lãi suất %/năm. `null` = chưa khai; khác hẳn 0 = vay không lãi. */
  interest_rate: number | null
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
  education_level: '',
  occupation: '',
  employment_years: null,

  loan_amount: 0,
  loan_term_months: null,
  interest_rate: null,
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
 * Khoảng lãi suất %/năm được chấp nhận, và số chữ số thập phân tối đa.
 *
 * Giữ khớp `StoreLoanApplicationRequest::MIN_INTEREST_RATE` / `MAX_INTEREST_RATE`
 * / `INTEREST_RATE_DECIMALS` và CHECK `chk_loan_interest_rate` ở DB. Ba nơi
 * lệch nhau thì form cho gõ một giá trị mà backend từ chối, hoặc ngược lại.
 */
export const INTEREST_RATE_MIN = 6
export const INTEREST_RATE_MAX = 10
export const INTEREST_RATE_DECIMALS = 2

/** Số chữ số sau dấu thập phân. `9.99` → 2, `10` → 0. */
const decimalPlaces = (n: number) => (String(n).split('.')[1] ?? '').length

/**
 * Khoản trả hàng tháng (EMI). Người dùng KHÔNG nhập số này — form hiển thị nó
 * read-only, và backend tính lại y hệt khi lưu.
 *
 * Chưa khai lãi suất thì chỉ chia đều tiền gốc; có lãi suất thì dùng công thức
 * trả góp đều, gồm cả gốc lẫn lãi:
 *
 *     r = lãi suất năm / 100 / 12
 *     EMI = P · r · (1+r)^n / ((1+r)^n − 1)
 *
 * Phải khớp TỪNG BƯỚC với `LoanApplicationService::monthlyPayment()` phía
 * backend, kể cả việc làm tròn LÊN: lệch cách làm tròn thì người dùng thấy một
 * số trước khi lưu và một số khác sau khi lưu, mà chẳng có gì giải thích.
 */
export const monthlyPayment = (
  amount: number,
  termMonths: number | null,
  ratePercent: number | null,
) => {
  if (amount <= 0 || !termMonths) return 0
  if (ratePercent === null || ratePercent <= 0) return Math.ceil(amount / termMonths)

  const monthlyRate = ratePercent / 100 / 12
  const growth = (1 + monthlyRate) ** termMonths

  return Math.ceil((amount * monthlyRate * growth) / (growth - 1))
}

/**
 * Ảnh chụp khoản trả hàng tháng ĐÃ LƯU, kèm ba trường đã sinh ra nó.
 *
 * Tồn tại để form biết khi nào KHÔNG được hiện số tự tính. Backend chỉ tính lại
 * EMI khi một trong ba trường này đổi (`LoanApplicationService::paymentDriversUnchanged()`),
 * nên form phải theo đúng luật đó — nếu không, bản ghi cũ sẽ hiện một con số
 * trên màn hình rồi lưu xuống một con số khác, và không có gì giải thích.
 */
export interface SavedPayment {
  loan_amount: number
  loan_term_months: number
  interest_rate: number | null
  monthly_payment: number
}

/**
 * Ba trường quyết định EMI có còn y như lúc nạp bản ghi không?
 *
 * Phải khớp `LoanApplicationService::PAYMENT_DRIVERS` phía backend. Lệch một
 * trường là hai bên bất đồng về việc có tính lại hay không.
 */
export const paymentDriversUnchanged = (
  saved: SavedPayment | null,
  form: LoanApplicationForm,
): saved is SavedPayment =>
  saved !== null &&
  saved.loan_amount === form.loan_amount &&
  saved.loan_term_months === form.loan_term_months &&
  saved.interest_rate === form.interest_rate

/**
 * Lỗi của ô lãi suất, `null` khi hợp lệ (bỏ trống cũng là hợp lệ — đây là
 * trường tùy chọn).
 *
 * Kiểm ngay ở FE chứ không đợi 422: lãi suất sai thì EMI hiển thị bên dưới
 * cũng sai theo, nên phải chặn trước khi con số đó lên màn hình.
 */
export const interestRateError = (rate: number | null): string | null => {
  if (rate === null) return null

  if (!Number.isFinite(rate)) {
    return 'Lãi suất phải là một con số.'
  }

  if (rate < INTEREST_RATE_MIN || rate > INTEREST_RATE_MAX) {
    return `Lãi suất phải nằm trong khoảng ${INTEREST_RATE_MIN}% đến ${INTEREST_RATE_MAX}%/năm.`
  }

  if (decimalPlaces(rate) > INTEREST_RATE_DECIMALS) {
    return `Lãi suất chỉ được có tối đa ${INTEREST_RATE_DECIMALS} chữ số sau dấu thập phân.`
  }

  return null
}
