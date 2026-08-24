import { useEffect, useState } from 'react'
import {
  AlertCircle,
  ArrowLeft,
  CheckCircle2,
  Info,
  Loader2,
  Trash2,
} from 'lucide-react'
import {
  EDUCATION_LABELS,
  GENDER_LABELS,
  LOAN_PURPOSE_LABELS,
  LOAN_TERM_CHOICES,
  MARITAL_STATUS_LABELS,
  OCCUPATION_LABELS,
  emptyLoanForm,
  loanTermLabel,
  minimumMonthlyPayment,
  toOptions,
  type LoanApplicationForm,
} from '../data/loan'
import type { PageKey } from '../data/profile'
import { currency } from '../lib/format'
import {
  deleteLoanApplication,
  fromLoanApplicationResponse,
  getLoanApplication,
  saveLoanApplication,
} from '../api/loanApplication'
import { deleteConversations } from '../api/messages'
import { ApiError, type FieldErrors } from '../lib/api'
import ConfirmDialog from '../components/ConfirmDialog'
import {
  Field,
  MoneyField,
  NumberField,
  Section,
  Segmented,
  Select,
  Stepper,
} from '../components/FormControls'

const GENDER_OPTIONS = toOptions(GENDER_LABELS)
const MARITAL_OPTIONS = toOptions(MARITAL_STATUS_LABELS)
const EDUCATION_OPTIONS = toOptions(EDUCATION_LABELS)
const OCCUPATION_OPTIONS = toOptions(OCCUPATION_LABELS)
const PURPOSE_OPTIONS = toOptions(LOAN_PURPOSE_LABELS)
const TERM_OPTIONS = LOAN_TERM_CHOICES.map((months) => ({
  value: String(months),
  label: loanTermLabel(months),
}))

/** Nghề nghiệp không đi làm hưởng lương — với họ 0 năm là câu trả lời đúng. */
const OUT_OF_WORKFORCE = new Set(['retired', 'unemployed'])

/**
 * Màn "Thông tin khoản vay" — dữ liệu đầu vào của ML02 (Home Credit Risk).
 *
 * Tách khỏi màn "Nhập thông tin" chứ không nối thêm vào đó: 16 ô này chỉ có ý
 * nghĩa với người đang tính vay, mà phần lớn người dùng chỉ muốn xem sức khoẻ
 * tài chính. Bắt tất cả nhập thêm 16 ô cho một tính năng họ không dùng là đổi
 * một lượng lớn người bỏ dở lấy một tính năng thiểu số.
 *
 * Phương án vay gắn với một hồ sơ hộ gia đình (endpoint nằm dưới
 * `/households/{id}`), nên chưa có hồ sơ thì trang này chưa mở được — ML02 cần
 * thu nhập và chi tiêu của hộ để dựng feature tỉ lệ.
 */
export default function LoanFormPage({
  householdId,
  onNavigate,
  onConversationCleared,
  onSaved,
}: {
  householdId: number | null
  onNavigate: (page: PageKey) => void
  /**
   * Đã lưu khoản vay mới. App dùng tín hiệu này để gỡ lời nhắn "vừa xoá sạch
   * dữ liệu" ở khung chat — người dùng vừa nhập số liệu mới thì câu đó hết đúng.
   */
  onSaved: () => void
  /**
   * Đã xoá xong hội thoại ở backend — App phải tăng `chatResetToken` để màn
   * Chatbot dọn tin nhắn đang giữ trong state. Không có bước này thì dữ liệu
   * đã mất ở server nhưng người dùng chuyển tab vẫn đọc được hội thoại cũ.
   */
  onConversationCleared: () => void
}) {
  const [form, setForm] = useState<LoanApplicationForm>(emptyLoanForm)
  const [loading, setLoading] = useState(false)
  const [submitting, setSubmitting] = useState(false)
  const [saved, setSaved] = useState(false)
  const [formError, setFormError] = useState<string | null>(null)
  const [fieldErrors, setFieldErrors] = useState<FieldErrors>({})
  const [confirmingClear, setConfirmingClear] = useState(false)
  const [clearing, setClearing] = useState(false)

  const patch = (values: Partial<LoanApplicationForm>) => {
    setForm((prev) => ({ ...prev, ...values }))
    setSaved(false)
  }

  /** Lỗi đầu tiên của một field do backend trả về. */
  const errorOf = (field: string) => fieldErrors[field]?.[0]

  // Nạp phương án đã lưu để người dùng sửa tiếp thay vì gõ lại từ đầu.
  // 404 nghĩa là hộ chưa từng khai khoản vay — trạng thái bình thường, giữ
  // form trống chứ không báo lỗi.
  useEffect(() => {
    if (householdId === null) {
      setForm(emptyLoanForm)
      return
    }

    setLoading(true)
    getLoanApplication(householdId)
      .then((data) => setForm(fromLoanApplicationResponse(data)))
      .catch(() => setForm(emptyLoanForm))
      .finally(() => setLoading(false))
  }, [householdId])

  const handleSubmit = async () => {
    if (householdId === null) return

    setSubmitting(true)
    setFormError(null)
    setFieldErrors({})

    try {
      const data = await saveLoanApplication(householdId, form)
      setForm(fromLoanApplicationResponse(data))
      setSaved(true)
      onSaved()
    } catch (error) {
      if (error instanceof ApiError) {
        setFormError(error.message)
        setFieldErrors(error.fieldErrors)
      } else {
        setFormError('Không gửi được thông tin, vui lòng thử lại.')
      }
    } finally {
      setSubmitting(false)
    }
  }

  /**
   * Xoá thông tin khoản vay và lịch sử hội thoại.
   *
   * KHÔNG xoá hồ sơ hộ gia đình: form này không sở hữu hồ sơ, và người dùng
   * bấm "Xóa dữ liệu" ở đây là muốn làm lại phương án vay chứ không phải khai
   * lại thu nhập, chi tiêu, thành viên.
   *
   * Nhưng hội thoại thì phải xoá: các câu trả lời ML02 về khoản vay nằm chung
   * một phiên với phần tư vấn tài chính, nên xoá khoản vay mà giữ hội thoại là
   * để lại nguyên kết luận rủi ro tính trên con số vừa bị xoá.
   *
   * Xoá khoản vay TRƯỚC rồi mới xoá hội thoại: nếu đảo lại và bước hai hỏng,
   * hội thoại đã mất trong khi khoản vay vẫn còn — mất dữ liệu mà chẳng đạt
   * được gì. Thứ tự này thì bước một hỏng là dừng, chưa mất gì cả.
   *
   * 404 ở bước xoá khoản vay là chuyện bình thường: hộ chưa từng khai khoản
   * vay nào. Vẫn phải chạy tiếp để dọn hội thoại và reset form.
   */
  const handleClear = async () => {
    if (householdId === null) return

    setClearing(true)
    setFormError(null)

    try {
      try {
        await deleteLoanApplication(householdId)
      } catch (error) {
        if (!(error instanceof ApiError) || error.status !== 404) throw error
      }

      await deleteConversations(householdId)

      setForm(emptyLoanForm)
      setFieldErrors({})
      setSaved(false)
      setConfirmingClear(false)
      onConversationCleared()
    } catch (error) {
      setConfirmingClear(false)
      setFormError(
        error instanceof ApiError
          ? error.message
          : 'Không xoá được dữ liệu, vui lòng thử lại.',
      )
    } finally {
      setClearing(false)
    }
  }

  /**
   * Chọn nghề không đi làm hưởng lương thì điền sẵn 0 năm — nhưng chỉ khi ô
   * còn trống, để không ghi đè con số người dùng đã tự nhập.
   */
  const handleOccupationChange = (value: string) => {
    const occupation = value as LoanApplicationForm['occupation']
    patch({
      occupation,
      employment_years:
        OUT_OF_WORKFORCE.has(value) && form.employment_years === null
          ? 0
          : form.employment_years,
    })
  }

  /**
   * Chưa từng vay thì không thể có lần trả chậm hay khoản quá hạn. Ép về 0 tại
   * chỗ thay vì để backend trả 422 — người dùng không học được gì từ một thông
   * báo lỗi cho tình huống mà form lẽ ra không nên cho phép tạo ra.
   */
  const handlePreviousLoanCount = (value: number | null) => {
    const count = value ?? 0
    patch(
      count === 0
        ? {
            previous_loan_count: 0,
            late_payment_count: 0,
            has_overdue_loan: false,
            total_overdue_amount: 0,
          }
        : { previous_loan_count: count },
    )
  }

  const minimumPayment = minimumMonthlyPayment(
    form.loan_amount,
    form.loan_term_months,
  )
  const ltv =
    form.loan_amount > 0 && form.asset_price > 0
      ? form.loan_amount / form.asset_price
      : null
  const hasCreditHistory = form.previous_loan_count > 0

  if (householdId === null) {
    return (
      <div className="rounded-3xl border border-slate-100 bg-white p-6 shadow-sm sm:p-8">
        <div className="flex items-start gap-3 rounded-2xl border border-amber-200 bg-amber-50 p-5 text-sm text-amber-800">
          <Info size={20} className="mt-0.5 shrink-0" />
          <div>
            <p className="font-semibold">Cần hồ sơ hộ gia đình trước</p>
            <p className="mt-1 text-amber-700">
              Phần đánh giá khoản vay dựa trên thu nhập và chi tiêu của gia đình,
              nên hãy hoàn thành màn <strong>Nhập thông tin</strong> trước đã.
            </p>
          </div>
        </div>
        <button
          type="button"
          onClick={() => onNavigate('info')}
          className="mt-6 w-full rounded-2xl bg-brand-600 py-4 text-base font-bold text-white shadow-sm transition hover:bg-brand-700"
        >
          Sang màn Nhập thông tin
        </button>
      </div>
    )
  }

  return (
    <div className="rounded-3xl border border-slate-100 bg-white p-6 shadow-sm sm:p-8">
      <div className="mb-6 flex flex-wrap items-center justify-between gap-3">
        <button
          type="button"
          onClick={() => onNavigate('info')}
          className="inline-flex items-center gap-2 rounded-xl bg-slate-100 px-4 py-2 text-sm font-medium text-slate-600 hover:bg-slate-200"
        >
          <ArrowLeft size={16} /> Trang trước
        </button>
        <button
          type="button"
          onClick={() => setConfirmingClear(true)}
          disabled={clearing || submitting || loading}
          className="inline-flex items-center gap-2 rounded-xl border border-rose-200 bg-rose-50 px-4 py-2 text-sm font-semibold text-rose-600 hover:bg-rose-100 disabled:opacity-60"
        >
          {clearing ? (
            <Loader2 size={16} className="animate-spin" />
          ) : (
            <Trash2 size={16} />
          )}
          Xóa dữ liệu
        </button>
      </div>

      <ConfirmDialog
        open={confirmingClear}
        busy={clearing}
        title="Xóa dữ liệu khoản vay?"
        description={
          <>
            Thao tác này xoá thông tin khoản vay và{' '}
            <strong className="font-semibold text-slate-700">
              toàn bộ lịch sử trò chuyện với AI
            </strong>
            , vì các kết luận về rủi ro đã dựa trên số liệu sắp bị xoá. Hồ sơ hộ
            gia đình được giữ nguyên. Dữ liệu không khôi phục lại được.
          </>
        }
        confirmLabel="Xóa dữ liệu"
        onConfirm={handleClear}
        onCancel={() => setConfirmingClear(false)}
      />

      <div className="mb-6">
        <h1 className="text-xl font-bold text-slate-800">Thông tin khoản vay</h1>
        <p className="mt-1 text-sm text-slate-500">
          Dùng để ước lượng mức độ rủi ro của khoản vay bạn đang cân nhắc. Đây là
          thông tin tham khảo, không phải kết quả thẩm định của tổ chức tín dụng.
        </p>
      </div>

      {loading && (
        <div className="mb-6 flex items-center gap-2 text-sm text-slate-500">
          <Loader2 size={16} className="animate-spin" />
          Đang tải thông tin đã lưu...
        </div>
      )}

      <div className="space-y-5">
        <Section
          index="A"
          title="Thông tin người vay"
          description="Người đứng tên khoản vay, không nhất thiết là người đại diện hộ."
        >
          <NumberField
            label="Tuổi"
            value={form.borrower_age}
            onChange={(borrower_age) => patch({ borrower_age })}
            error={errorOf('borrower_age')}
            unit="tuổi"
          />

          <Field label="Giới tính" error={errorOf('gender')}>
            <Select
              value={form.gender}
              onChange={(gender) =>
                patch({ gender: gender as LoanApplicationForm['gender'] })
              }
              placeholder="Chọn giới tính"
              options={GENDER_OPTIONS}
            />
          </Field>

          <Field label="Tình trạng hôn nhân" error={errorOf('marital_status')}>
            <Select
              value={form.marital_status}
              onChange={(marital_status) =>
                patch({
                  marital_status:
                    marital_status as LoanApplicationForm['marital_status'],
                })
              }
              placeholder="Chọn tình trạng hôn nhân"
              options={MARITAL_OPTIONS}
            />
          </Field>

          <Field label="Số con" error={errorOf('children_count')}>
            <Stepper
              value={form.children_count}
              max={20}
              onChange={(children_count) => patch({ children_count })}
            />
          </Field>

          <Field label="Trình độ học vấn" error={errorOf('education_level')}>
            <Select
              value={form.education_level}
              onChange={(education_level) =>
                patch({
                  education_level:
                    education_level as LoanApplicationForm['education_level'],
                })
              }
              placeholder="Chọn trình độ học vấn"
              options={EDUCATION_OPTIONS}
            />
          </Field>

          <Field label="Nghề nghiệp" error={errorOf('occupation')}>
            <Select
              value={form.occupation}
              onChange={handleOccupationChange}
              placeholder="Chọn nghề nghiệp"
              options={OCCUPATION_OPTIONS}
            />
          </Field>

          <NumberField
            label="Thời gian làm việc"
            value={form.employment_years}
            onChange={(employment_years) => patch({ employment_years })}
            error={errorOf('employment_years')}
            hint="Số năm đi làm có thu nhập. Nghỉ hưu hoặc chưa đi làm thì điền 0."
            unit="năm"
          />
        </Section>

        <Section
          index="B"
          title="Thông tin khoản vay"
          description="Khoản vay bạn đang cân nhắc, không phải khoản nợ hiện có."
        >
          <MoneyField
            label="Số tiền vay"
            value={form.loan_amount}
            onChange={(loan_amount) => patch({ loan_amount })}
            error={errorOf('loan_amount')}
          />

          <Field label="Thời hạn vay" error={errorOf('loan_term_months')}>
            <Select
              value={form.loan_term_months ? String(form.loan_term_months) : ''}
              onChange={(value) => patch({ loan_term_months: Number(value) })}
              placeholder="Chọn thời hạn vay"
              options={TERM_OPTIONS}
            />
          </Field>

          <MoneyField
            label="Khoản trả hàng tháng"
            value={form.monthly_payment}
            onChange={(monthly_payment) => patch({ monthly_payment })}
            error={errorOf('monthly_payment')}
            hint={
              minimumPayment > 0
                ? `Tối thiểu ${currency(minimumPayment)} mới trả hết gốc, chưa tính lãi.`
                : undefined
            }
            suffix=" / tháng"
          />

          <MoneyField
            label="Giá trị tài sản"
            value={form.asset_price}
            onChange={(asset_price) => patch({ asset_price })}
            error={errorOf('asset_price')}
            hint="Giá tài sản định mua, hoặc giá trị tài sản dùng làm bảo đảm."
          />

          <Field label="Mục đích vay" error={errorOf('loan_purpose')}>
            <Select
              value={form.loan_purpose}
              onChange={(loan_purpose) =>
                patch({
                  loan_purpose: loan_purpose as LoanApplicationForm['loan_purpose'],
                })
              }
              placeholder="Chọn mục đích vay"
              options={PURPOSE_OPTIONS}
            />
          </Field>

          {ltv !== null && (
            <div className="flex items-center rounded-xl border border-brand-100 bg-brand-50 px-4 py-3 text-sm text-brand-700 md:self-end">
              Vay <strong className="mx-1">{(ltv * 100).toFixed(0)}%</strong> giá
              trị tài sản, tự có {((1 - ltv) * 100).toFixed(0)}%
            </div>
          )}
        </Section>

        <Section
          index="C"
          title="Lịch sử tín dụng"
          description="Các khoản vay trước đây tại ngân hàng hoặc công ty tài chính."
        >
          <NumberField
            label="Số khoản vay trước đây"
            value={form.previous_loan_count}
            onChange={handlePreviousLoanCount}
            error={errorOf('previous_loan_count')}
            hint="Tính cả khoản đã tất toán. Chưa từng vay thì để 0."
            unit="khoản"
          />

          <NumberField
            label="Số lần trả chậm"
            value={hasCreditHistory ? form.late_payment_count : 0}
            onChange={(late_payment_count) =>
              patch({ late_payment_count: late_payment_count ?? 0 })
            }
            error={errorOf('late_payment_count')}
            hint={
              hasCreditHistory
                ? 'Số kỳ đã trả trễ hạn ở tất cả khoản vay trước.'
                : 'Chưa có khoản vay nào trước đây.'
            }
            unit="lần"
          />

          <Field
            label="Có khoản vay quá hạn không?"
            error={errorOf('has_overdue_loan')}
            hint={
              hasCreditHistory
                ? undefined
                : 'Chưa có khoản vay nào trước đây nên mục này để trống.'
            }
          >
            <Segmented
              value={form.has_overdue_loan}
              onChange={(has_overdue_loan) =>
                hasCreditHistory &&
                patch({
                  has_overdue_loan,
                  // Rút lại lựa chọn thì bỏ luôn số tiền: giữ nó lại là để một
                  // con số người dùng đã thu hồi vẫn đi tiếp xuống tầng ML.
                  total_overdue_amount: has_overdue_loan
                    ? form.total_overdue_amount
                    : 0,
                })
              }
              yes="Có quá hạn"
              no="Không có"
            />

            {form.has_overdue_loan && (
              <div className="mt-3">
                <MoneyField
                  label="Tổng nợ quá hạn"
                  value={form.total_overdue_amount}
                  onChange={(total_overdue_amount) =>
                    patch({ total_overdue_amount })
                  }
                  error={errorOf('total_overdue_amount')}
                />
              </div>
            )}
          </Field>
        </Section>
      </div>

      {formError && (
        <div className="mt-6 flex items-start gap-2 rounded-2xl border border-rose-200 bg-rose-50 p-4 text-sm text-rose-700">
          <AlertCircle size={18} className="mt-0.5 shrink-0" />
          <p>{formError}</p>
        </div>
      )}

      {saved && (
        <div className="mt-6 flex items-start gap-2 rounded-2xl border border-emerald-200 bg-emerald-50 p-4 text-sm text-emerald-700">
          <CheckCircle2 size={18} className="mt-0.5 shrink-0" />
          <p>
            Đã lưu thông tin khoản vay. Phần đánh giá rủi ro sẽ dùng dữ liệu này.
          </p>
        </div>
      )}

      <button
        type="button"
        onClick={handleSubmit}
        disabled={submitting || loading}
        className="mt-8 flex w-full items-center justify-center gap-2 rounded-2xl bg-brand-600 py-4 text-base font-bold text-white shadow-sm transition hover:bg-brand-700 disabled:cursor-not-allowed disabled:opacity-70"
      >
        {submitting && <Loader2 size={20} className="animate-spin" />}
        {submitting ? 'Đang lưu...' : 'Lưu thông tin khoản vay'}
      </button>

      <p className="mt-4 text-center text-xs text-slate-400">
        Kết quả đánh giá là ước lượng tham khảo dựa trên dữ liệu bạn cung cấp,
        không thay thế quyết định của tổ chức tín dụng.
      </p>
    </div>
  )
}
