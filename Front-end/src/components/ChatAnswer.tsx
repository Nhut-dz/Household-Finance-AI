import { useMemo, type ReactNode } from 'react'
import {
  parseAnswer,
  splitLabel,
  tokenize,
  type Block,
  type ListItem,
  type Priority,
  type Tone,
} from '../lib/chatFormat'

// Câu trả lời của chatbot, dựng từ chữ thuần server trả về (xem `chatFormat`).
//
// Bảng màu cố ý hẹp để mắt còn phân biệt được: MỘT màu nhấn (brand) cho mọi
// con số — tiền, %, thời hạn — và ba sắc thái ngữ nghĩa cho kết luận: đỏ
// (rose) = cần xử lý, vàng (amber) = cần lưu ý, xanh (emerald) = ổn. Mức ưu
// tiên của việc nên làm dùng lại đúng ba sắc thái đó, nên "ưu tiên cao" và
// "mức cần xử lý ngay" cùng một màu — cùng một mức khẩn.

/**
 * Nền tô kiểu bút dạ: `-mx-0.5 px-1` cho nền lấn ra 4px nhưng chỉ đẩy dòng
 * chữ 2px, để dấu câu ngay sau ("…xử lý ngay.") không bị hở một khoảng.
 * `mix-blend-multiply` để phần nền lấn sang không che mất dấu ngoặc hay dấu
 * chấm đứng sát — nền nhạt nhân với chữ đậm thì chữ vẫn còn nguyên.
 */
const HIGHLIGHT = 'rounded -mx-0.5 px-1 py-0.5 box-decoration-clone mix-blend-multiply'

const TONE_CLASS: Record<Tone, string> = {
  danger: `${HIGHLIGHT} bg-rose-100 font-bold text-rose-700`,
  caution: `${HIGHLIGHT} bg-amber-100 font-semibold text-amber-800`,
  positive: `${HIGHLIGHT} bg-emerald-100 font-semibold text-emerald-700`,
}

const PRIORITY_CHIP: Record<Priority, { label: string; className: string }> = {
  high: { label: 'Ưu tiên cao', className: 'bg-rose-100 text-rose-700' },
  medium: { label: 'Ưu tiên vừa', className: 'bg-amber-100 text-amber-800' },
  low: { label: 'Ưu tiên thấp', className: 'bg-slate-200 text-slate-600' },
}

/** Chữ trong một dòng, đã tô số và cụm kết luận. */
function Inline({ text }: { text: string }) {
  return (
    <>
      {tokenize(text).map((token, index) => {
        switch (token.kind) {
          case 'strong':
            return (
              <strong key={index} className="font-semibold text-slate-900">
                {token.value}
              </strong>
            )
          case 'number':
            return (
              <span key={index} className="font-semibold tabular-nums text-brand-700">
                {token.value}
              </span>
            )
          case 'tone':
            return (
              <span key={index} className={TONE_CLASS[token.tone]}>
                {token.value}
              </span>
            )
          default:
            return token.value
        }
      })}
    </>
  )
}

function Icon({ children }: { children: ReactNode }) {
  return (
    <span aria-hidden className="shrink-0 select-none leading-none">
      {children}
    </span>
  )
}

/**
 * "Nhãn: giá trị" với nhãn in đậm; không có nhãn thì là dòng chữ thường.
 *
 * `maxRest` giới hạn độ dài phần sau dấu hai chấm — đoạn văn dài có dấu hai
 * chấm ở giữa câu ("…đang ở trạng thái X: chi tiêu vượt…") không phải cặp
 * nhãn–giá trị, in đậm nửa đầu câu chỉ làm rối.
 */
function LabelledText({ text, maxRest }: { text: string; maxRest?: number }) {
  const kv = splitLabel(text)
  if (!kv || (maxRest !== undefined && kv.rest.length > maxRest)) {
    return <Inline text={text} />
  }
  return (
    <>
      <span className="font-semibold text-slate-800">
        <Inline text={kv.label} />
      </span>
      {': '}
      <Inline text={kv.rest} />
    </>
  )
}

function Item({ item }: { item: ListItem }) {
  const chip = item.priority ? PRIORITY_CHIP[item.priority] : null

  return (
    <li className="flex items-start gap-2.5">
      <span className="mt-[9px] h-1.5 w-1.5 shrink-0 rounded-full bg-brand-500" />
      <div className="min-w-0 flex-1">
        <div>
          <LabelledText text={item.text} />
          {chip && (
            <span
              className={`ml-1.5 inline-block rounded-full px-2 py-0.5 align-[2px] text-[11px] font-semibold leading-4 ${chip.className}`}
            >
              {chip.label}
            </span>
          )}
        </div>
        {item.reason && (
          <div className="mt-1 border-l-2 border-slate-300 pl-2.5 text-[13px] leading-relaxed text-slate-600">
            <span className="font-semibold text-slate-700">💡 Lý do:</span>{' '}
            <Inline text={item.reason} />
          </div>
        )}
      </div>
    </li>
  )
}

function BlockView({ block, first }: { block: Block; first: boolean }) {
  switch (block.kind) {
    case 'title':
      // Tiêu đề mở đầu là tên câu trả lời — to hơn, có kẻ chân. Tiêu đề emoji
      // xuất hiện giữa chừng (nhánh rule có "💡 Gói tiết kiệm:" rồi "🐖 Gói
      // tư vấn…:") chỉ là tên mục, cùng cấp với các dòng "Việc nên làm:".
      return (
        <div
          className={`flex flex-wrap items-center gap-x-2 gap-y-1 font-bold text-slate-900 ${
            first
              ? 'border-b border-slate-200 pb-2.5 text-[15px]'
              : 'pt-2 text-sm'
          }`}
        >
          <Icon>{block.icon}</Icon>
          <span>
            <Inline text={block.label} />
            {block.value && ':'}
          </span>
          {block.value && (
            <span>
              <Inline text={block.value} />
            </span>
          )}
        </div>
      )

    case 'heading':
      return (
        <div className="flex items-center gap-1.5 pt-2 text-sm font-bold text-slate-900">
          <Icon>{block.icon}</Icon>
          {block.label}
        </div>
      )

    case 'list':
      return (
        <ul className="space-y-1.5 pl-0.5">
          {block.items.map((item, index) => (
            <Item key={index} item={item} />
          ))}
        </ul>
      )

    case 'warning':
      return (
        <div className="flex items-start gap-2 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-amber-900">
          <Icon>⚠️</Icon>
          <p>
            <Inline text={block.text} />
          </p>
        </div>
      )

    case 'disclaimer':
      return (
        <p className="flex items-start gap-2 rounded-lg bg-slate-100 px-3 py-2 text-xs italic leading-relaxed text-slate-500">
          <Icon>ℹ️</Icon>
          <span>{block.text}</span>
        </p>
      )

    default:
      return (
        <p>
          <LabelledText text={block.text} maxRest={60} />
        </p>
      )
  }
}

export default function ChatAnswer({ content }: { content: string }) {
  const blocks = useMemo(() => parseAnswer(content), [content])

  return (
    <div className="space-y-2.5 text-sm leading-relaxed text-slate-700">
      {blocks.map((block, index) => (
        <BlockView key={index} block={block} first={index === 0} />
      ))}
    </div>
  )
}
