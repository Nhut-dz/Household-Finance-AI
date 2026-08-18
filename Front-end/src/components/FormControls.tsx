import { ChevronDown, Minus, Plus } from 'lucide-react'
import { currency } from '../lib/format'

/**
 * Các ô nhập dùng chung của hai màn form ("Nhập thông tin" và "Thông tin khoản
 * vay"). Tách ra khỏi InfoFormPage khi màn thứ hai ra đời: chép sang một bản
 * thứ hai thì hai form sẽ dần trông khác nhau ở những chỗ không ai chủ ý.
 */

export const inputClass =
  'w-full rounded-xl border border-slate-200 bg-white px-4 py-3 text-sm text-slate-800 placeholder:text-slate-400 focus:border-brand-500 focus:outline-none focus:ring-2 focus:ring-brand-100'

export function Field({
  label,
  error,
  hint,
  children,
}: {
  label: string
  error?: string
  /** Chú thích dưới nhãn — dùng cho ô cần giải thích cách nhập. */
  hint?: string
  children: React.ReactNode
}) {
  return (
    <div>
      <label className="mb-2 block text-sm font-semibold text-slate-700">
        {label}
      </label>
      {hint && <p className="-mt-1 mb-2 text-xs text-slate-400">{hint}</p>}
      {children}
      {error && <p className="mt-1.5 text-xs font-medium text-rose-600">{error}</p>}
    </div>
  )
}

/**
 * Một lựa chọn của dropdown. Chuỗi thuần nghĩa là hiển thị và gửi đi cùng một
 * giá trị (danh sách tỉnh, năm sinh); dạng object dành cho enum của backend,
 * nơi giá trị gửi đi là `occupation` còn người dùng đọc "Nhân viên văn phòng".
 */
export type SelectOption = string | { value: string; label: string }

const optionValue = (option: SelectOption) =>
  typeof option === 'string' ? option : option.value

const optionLabel = (option: SelectOption) =>
  typeof option === 'string' ? option : option.label

/** Dropdown dùng chung, trông giống ô nhập chữ nhưng có mũi tên chọn. */
export function Select({
  value,
  onChange,
  placeholder,
  options,
}: {
  value: string
  onChange: (v: string) => void
  placeholder: string
  options: readonly SelectOption[]
}) {
  return (
    <div className="relative">
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className={`${inputClass} appearance-none pr-10 ${
          value ? '' : 'text-slate-400'
        }`}
      >
        <option value="">{placeholder}</option>
        {options.map((option) => (
          <option
            key={optionValue(option)}
            value={optionValue(option)}
            className="text-slate-800"
          >
            {optionLabel(option)}
          </option>
        ))}
      </select>
      <ChevronDown
        size={18}
        className="pointer-events-none absolute right-3 top-1/2 -translate-y-1/2 text-slate-400"
      />
    </div>
  )
}

export function Stepper({
  value,
  onChange,
  min = 0,
  max,
}: {
  value: number
  onChange: (v: number) => void
  min?: number
  max?: number
}) {
  const atMax = max !== undefined && value >= max

  return (
    <div className="flex items-center gap-3 rounded-xl border border-slate-200 bg-white px-2 py-1.5">
      <button
        type="button"
        onClick={() => onChange(Math.max(min, value - 1))}
        disabled={value <= min}
        className="grid h-8 w-8 place-items-center rounded-lg bg-slate-100 text-slate-500 hover:bg-slate-200 disabled:cursor-not-allowed disabled:opacity-40"
      >
        <Minus size={16} />
      </button>
      <span className="flex-1 text-center text-sm font-semibold text-slate-800">
        {value}
      </span>
      <button
        type="button"
        onClick={() => onChange(atMax ? value : value + 1)}
        disabled={atMax}
        className="grid h-8 w-8 place-items-center rounded-lg bg-slate-100 text-slate-500 hover:bg-slate-200 disabled:cursor-not-allowed disabled:opacity-40"
      >
        <Plus size={16} />
      </button>
    </div>
  )
}

/** Cặp lựa chọn dạng segmented (ví dụ "Có nợ" / "Không nợ"). */
export function Segmented({
  value,
  onChange,
  yes,
  no,
}: {
  value: boolean
  onChange: (v: boolean) => void
  yes: string
  no: string
}) {
  return (
    <div className="grid grid-cols-2 gap-2">
      {[
        { label: yes, val: true },
        { label: no, val: false },
      ].map(({ label, val }) => {
        const active = value === val
        return (
          <button
            key={label}
            type="button"
            onClick={() => onChange(val)}
            className={`rounded-xl px-4 py-2.5 text-sm font-medium transition ${
              active
                ? 'bg-brand-600 text-white shadow-sm'
                : 'bg-slate-100 text-slate-500 hover:bg-slate-200'
            }`}
          >
            {label}
          </button>
        )
      })}
    </div>
  )
}

/**
 * Ô nhập số tiền: gõ số thuần, hiển thị có dấu phân cách nghìn.
 *
 * `suffix` để trang gọi tự nói rõ đơn vị kỳ — "35.000.000 ₫ / tháng" khác hẳn
 * "35.000.000 ₫" của một khoản vay. Mặc định không có kỳ.
 */
export function MoneyField({
  label,
  value,
  onChange,
  error,
  hint,
  suffix = '',
}: {
  label: string
  value: number
  onChange: (v: number) => void
  error?: string
  hint?: string
  suffix?: string
}) {
  return (
    <Field label={label} error={error} hint={hint}>
      <input
        type="text"
        inputMode="numeric"
        value={value ? value.toLocaleString('vi-VN') : ''}
        onChange={(e) => onChange(Number(e.target.value.replace(/\D/g, '')) || 0)}
        placeholder="Vui lòng nhập thông tin"
        className={inputClass}
      />
      <p className="mt-1.5 text-xs text-slate-400">
        Hiển thị: {currency(value)}
        {suffix}
      </p>
    </Field>
  )
}

/** Ô nhập số nguyên (tuổi, số lần trả chậm) — không định dạng nghìn. */
export function NumberField({
  label,
  value,
  onChange,
  error,
  hint,
  placeholder = 'Vui lòng nhập thông tin',
  unit,
}: {
  label: string
  value: number | null
  onChange: (v: number | null) => void
  error?: string
  hint?: string
  placeholder?: string
  /** Đơn vị hiện mờ bên phải ô, ví dụ "tuổi", "năm", "lần". */
  unit?: string
}) {
  return (
    <Field label={label} error={error} hint={hint}>
      <div className="relative">
        <input
          type="text"
          inputMode="decimal"
          value={value === null ? '' : String(value)}
          onChange={(e) => {
            const raw = e.target.value.replace(/[^\d.]/g, '')
            onChange(raw === '' ? null : Number(raw))
          }}
          placeholder={placeholder}
          className={`${inputClass} ${unit ? 'pr-16' : ''}`}
        />
        {unit && (
          <span className="pointer-events-none absolute right-4 top-1/2 -translate-y-1/2 text-sm text-slate-400">
            {unit}
          </span>
        )}
      </div>
    </Field>
  )
}

/**
 * Khung đề mục của một nhóm trường. Trang "Thông tin khoản vay" có ba nhóm nên
 * cần ranh giới nhìn thấy được — 16 ô rải phẳng trên một lưới thì người dùng
 * không biết mình đang ở đoạn nào.
 */
export function Section({
  index,
  title,
  description,
  children,
}: {
  /** Chữ cái đề mục: A · B · C. */
  index: string
  title: string
  description?: string
  children: React.ReactNode
}) {
  return (
    <section className="rounded-2xl border border-slate-100 bg-slate-50/50 p-5 sm:p-6">
      <div className="mb-5 flex items-start gap-3">
        <span className="grid h-8 w-8 shrink-0 place-items-center rounded-lg bg-brand-600 text-sm font-bold text-white">
          {index}
        </span>
        <div>
          <h2 className="text-base font-bold text-slate-800">{title}</h2>
          {description && (
            <p className="mt-0.5 text-xs text-slate-500">{description}</p>
          )}
        </div>
      </div>
      <div className="grid gap-x-8 gap-y-5 md:grid-cols-2">{children}</div>
    </section>
  )
}
