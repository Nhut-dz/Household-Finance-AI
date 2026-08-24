import { useState } from 'react'
import { AlertCircle, ArrowLeft, Loader2, Trash2 } from 'lucide-react'
import {
  ASSET_LABELS,
  NEED_LABELS,
  type AssetKey,
  type HouseholdProfile,
  type NeedKey,
  type PageKey,
} from '../data/profile'
import { BIRTH_YEARS, PROVINCES } from '../data/locations'
import { currency } from '../lib/format'
import {
  createHousehold,
  deleteHousehold,
  updateHousehold,
} from '../api/households'
import { ApiError, type FieldErrors } from '../lib/api'
import ConfirmDialog from '../components/ConfirmDialog'
import {
  Field,
  MoneyField,
  Segmented,
  Select,
  Stepper,
  inputClass,
} from '../components/FormControls'

/** Blue-tinted nested box holding a debt amount + estimated monthly payment. */
function DebtBox({
  value,
  onChange,
  error,
}: {
  value: number
  onChange: (v: number) => void
  error?: string
}) {
  return (
    <div className="mt-2 rounded-xl border border-slate-200 bg-slate-50/70 p-3">
      <p className="mb-1.5 text-xs font-medium text-slate-600">
        Tổng dư nợ hiện tại (VNĐ)
      </p>
      <input
        type="text"
        inputMode="numeric"
        value={value ? value.toLocaleString('vi-VN') : ''}
        onChange={(e) => onChange(Number(e.target.value.replace(/\D/g, '')) || 0)}
        placeholder="Vui lòng nhập thông tin"
        className="w-full rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm focus:border-brand-500 focus:outline-none"
      />
      <p className="mt-1.5 text-[11px] text-slate-400">
        Trả nợ ước tính: {currency(Math.round(value / 100))}/tháng
      </p>
      {error && <p className="mt-1.5 text-xs font-medium text-rose-600">{error}</p>}
    </div>
  )
}

function ChipGroup<T extends string>({
  options,
  labels,
  selected,
  onToggle,
}: {
  options: T[]
  labels: Record<T, string>
  selected: T[]
  onToggle: (key: T) => void
}) {
  return (
    <div className="grid grid-cols-2 gap-3">
      {options.map((key) => {
        const active = selected.includes(key)
        return (
          <button
            key={key}
            type="button"
            onClick={() => onToggle(key)}
            className={`rounded-xl border px-4 py-2.5 text-sm font-medium transition ${
              active
                ? 'border-brand-500 bg-brand-50 text-brand-700'
                : 'border-slate-200 bg-white text-slate-500 hover:border-slate-300'
            }`}
          >
            {labels[key]}
          </button>
        )
      })}
    </div>
  )
}

export default function InfoFormPage({
  profile,
  householdId,
  onChange,
  onNavigate,
  onSaved,
  onCleared,
}: {
  profile: HouseholdProfile
  /** Đã có hồ sơ trên backend thì cập nhật, chưa có thì tạo mới. */
  householdId: number | null
  onChange: (patch: Partial<HouseholdProfile>) => void
  onNavigate: (page: PageKey) => void
  /**
   * `conversationRotated` chỉ có khi cập nhật hồ sơ: backend báo lần sửa này có
   * chạm vào dữ liệu tài chính hay không, để màn Chatbot biết phải xoá hội thoại
   * cũ hay giữ lại.
   */
  onSaved: (householdId: number, info?: { conversationRotated: boolean }) => void
  /**
   * Đã xoá xong ở backend — App phải trả `profile`, `householdId` và phiên chat
   * về trạng thái của người dùng mới. Trang này không tự làm được: `profile`
   * nằm ở App, và màn Chatbot chỉ dọn tin nhắn khi `chatResetToken` tăng.
   */
  onCleared: () => void
}) {
  const toggleAsset = (key: AssetKey) =>
    onChange({
      assets: profile.assets.includes(key)
        ? profile.assets.filter((a) => a !== key)
        : [...profile.assets, key],
    })

  const toggleNeed = (key: NeedKey) =>
    onChange({
      needs: profile.needs.includes(key)
        ? profile.needs.filter((n) => n !== key)
        : [...profile.needs, key],
    })

  const [submitting, setSubmitting] = useState(false)
  const [formError, setFormError] = useState<string | null>(null)
  const [fieldErrors, setFieldErrors] = useState<FieldErrors>({})
  const [confirmingClear, setConfirmingClear] = useState(false)
  const [clearing, setClearing] = useState(false)

  /** Lỗi đầu tiên của một field do backend trả về. */
  const errorOf = (field: string) => fieldErrors[field]?.[0]

  const handleSubmit = async () => {
    setSubmitting(true)
    setFormError(null)
    setFieldErrors({})

    try {
      if (householdId) {
        const saved = await updateHousehold(householdId, profile)
        onSaved(saved.id, { conversationRotated: saved.conversation_rotated })
      } else {
        const saved = await createHousehold(profile)
        onSaved(saved.id)
      }
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
   * Xoá hồ sơ và mọi thứ bám theo nó.
   *
   * Chỉ cần gọi `deleteHousehold`: ràng buộc ON DELETE CASCADE của DB kéo theo
   * tài sản, nhu cầu tài chính, thông tin khoản vay và toàn bộ phiên trò
   * chuyện. Gọi thêm từng endpoint xoá lẻ là thừa và tạo ra cửa sổ mà một
   * request hỏng để lại dữ liệu xoá dở.
   *
   * `householdId === null` nghĩa là phiên này chưa từng gửi hồ sơ lên backend —
   * không có gì để xoá ở server, nhưng người dùng vẫn có thể đã gõ dở nửa form,
   * nên vẫn phải dọn state.
   */
  const handleClear = async () => {
    setClearing(true)
    setFormError(null)

    try {
      if (householdId !== null) {
        await deleteHousehold(householdId)
      }
      setFieldErrors({})
      setConfirmingClear(false)
      onCleared()
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

  return (
    <div className="rounded-3xl border border-slate-100 bg-white p-6 shadow-sm sm:p-8">
      <div className="mb-6 flex flex-wrap items-center justify-between gap-3">
        <button
          type="button"
          onClick={() => onNavigate('home')}
          className="inline-flex items-center gap-2 rounded-xl bg-slate-100 px-4 py-2 text-sm font-medium text-slate-600 hover:bg-slate-200"
        >
          <ArrowLeft size={16} /> Trang trước
        </button>
        <button
          type="button"
          onClick={() => setConfirmingClear(true)}
          disabled={clearing || submitting}
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
        title="Xóa toàn bộ dữ liệu?"
        description={
          <>
            Thao tác này xoá hồ sơ hộ gia đình, thông tin khoản vay và{' '}
            <strong className="font-semibold text-slate-700">
              toàn bộ lịch sử trò chuyện với AI
            </strong>
            . Dữ liệu không khôi phục lại được. Sau khi xoá, bạn bắt đầu lại từ
            một phiên hoàn toàn mới.
          </>
        }
        confirmLabel="Xóa dữ liệu"
        onConfirm={handleClear}
        onCancel={() => setConfirmingClear(false)}
      />

      <div className="grid gap-x-8 gap-y-6 md:grid-cols-2">
        <Field
          label="Họ và tên người đại diện"
          error={errorOf('representative_name')}
        >
          <input
            value={profile.name}
            onChange={(e) => onChange({ name: e.target.value })}
            placeholder="Vui lòng nhập thông tin"
            className={inputClass}
          />
        </Field>

        <Field label="Năm sinh" error={errorOf('birth_year')}>
          <Select
            value={profile.birthYear}
            onChange={(birthYear) => onChange({ birthYear })}
            placeholder="Chọn năm sinh (YYYY)"
            options={BIRTH_YEARS.map(String)}
          />
        </Field>

        {/* Members + children steppers share the left cell */}
        <div className="grid grid-cols-2 gap-4">
          <Field label="Nhà có mấy người" error={errorOf('household_size')}>
            <Stepper
              value={profile.members}
              min={1}
              onChange={(members) => onChange({ members })}
            />
          </Field>
          <Field label="Mấy con" error={errorOf('children_count')}>
            <Stepper
              value={profile.children}
              onChange={(children) => onChange({ children })}
            />
          </Field>
        </div>

        <Field label="Nơi ở" error={errorOf('residence')}>
          <Select
            value={profile.location}
            onChange={(location) => onChange({ location })}
            placeholder="Chọn tỉnh / thành phố"
            options={PROVINCES}
          />
        </Field>

        <MoneyField
          label="Thu nhập trung bình tháng của gia đình"
          value={profile.income}
          onChange={(income) => onChange({ income })}
          error={errorOf('average_monthly_income')}
          suffix=" / tháng"
        />
        <MoneyField
          label="Chi tiêu trung bình tháng của gia đình"
          value={profile.spending}
          onChange={(spending) => onChange({ spending })}
          error={errorOf('average_monthly_expense')}
          suffix=" / tháng"
        />

        <Field label="Gia đình có nợ không" error={errorOf('has_debt')}>
          <Segmented
            value={profile.hasDebt}
            onChange={(hasDebt) => onChange({ hasDebt })}
            yes="Có nợ"
            no="Không nợ"
          />
          {profile.hasDebt && (
            <DebtBox
              value={profile.debt}
              onChange={(debt) => onChange({ debt })}
              error={errorOf('total_current_debt')}
            />
          )}
        </Field>

        <Field label="Có tiền tiết kiệm không?" error={errorOf('has_savings')}>
          <Segmented
            value={profile.hasSavings}
            onChange={(hasSavings) => onChange({ hasSavings })}
            yes="Có tiết kiệm"
            no="Không có"
          />
          {profile.hasSavings && (
            <div className="mt-2 rounded-xl border border-slate-200 bg-slate-50/70 p-3">
              <p className="mb-1.5 text-xs font-medium text-slate-600">
                Số tiền tiết kiệm (VNĐ)
              </p>
              <input
                type="text"
                inputMode="numeric"
                value={profile.savings ? profile.savings.toLocaleString('vi-VN') : ''}
                onChange={(e) =>
                  onChange({
                    savings: Number(e.target.value.replace(/\D/g, '')) || 0,
                  })
                }
                placeholder="Vui lòng nhập thông tin"
                className="w-full rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm focus:border-brand-500 focus:outline-none"
              />
              {errorOf('savings_amount') && (
                <p className="mt-1.5 text-xs font-medium text-rose-600">
                  {errorOf('savings_amount')}
                </p>
              )}
            </div>
          )}
        </Field>

        <Field
          label="Gia đình có đang phụng dưỡng người già không?"
          error={errorOf('has_dependents')}
        >
          <Segmented
            value={profile.supportingElderly}
            onChange={(supportingElderly) => onChange({ supportingElderly })}
            yes="Có phụng dưỡng"
            no="Không"
          />
        </Field>

        <div className="hidden md:block" />

        <Field
          label="Tài sản đang sở hữu (Chọn tất cả phù hợp)"
          error={errorOf('assets')}
        >
          <ChipGroup
            options={['nha', 'xe', 'dat']}
            labels={ASSET_LABELS}
            selected={profile.assets}
            onToggle={toggleAsset}
          />
        </Field>

        <Field
          label="Nhu cầu tài chính (Chọn tất cả phù hợp)"
          error={errorOf('financial_needs')}
        >
          <ChipGroup
            options={['muaNha', 'muaXe', 'muaDat', 'vayVon']}
            labels={NEED_LABELS}
            selected={profile.needs}
            onToggle={toggleNeed}
          />
        </Field>
      </div>

      {formError && (
        <div className="mt-6 flex items-start gap-2 rounded-2xl border border-rose-200 bg-rose-50 p-4 text-sm text-rose-700">
          <AlertCircle size={18} className="mt-0.5 shrink-0" />
          <p>{formError}</p>
        </div>
      )}

      <button
        type="button"
        onClick={handleSubmit}
        disabled={submitting}
        className="mt-8 flex w-full items-center justify-center gap-2 rounded-2xl bg-brand-600 py-4 text-base font-bold text-white shadow-sm transition hover:bg-brand-700 disabled:cursor-not-allowed disabled:opacity-70"
      >
        {submitting ? (
          <Loader2 size={20} className="animate-spin" />
        ) : (
          <img src="/iconlogo.png" alt="" className="h-6 w-6 rounded-md object-cover" />
        )}
        {submitting ? 'Đang gửi thông tin...' : 'Bắt đầu trò chuyện với AI'}
      </button>

      <img
        src="/img3.jpg"
        alt="Minh họa trợ lý AI phân tích dữ liệu tài chính"
        className="mt-6 h-auto w-full rounded-2xl"
      />
    </div>
  )
}
