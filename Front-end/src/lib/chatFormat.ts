/**
 * Phân tích câu trả lời CHỮ THUẦN của chatbot thành các khối để hiển thị.
 *
 * Server cố ý trả chữ thuần, không Markdown (xem `hfml.llm.presentation.
 * to_plain_text` và `Answer.as_text()` phía Python). Cấu trúc của câu trả lời
 * chỉ nằm ở quy ước dòng:
 *
 *     🧭 Tiêu đề: giá trị          dòng mở đầu bằng emoji
 *     Việc nên làm:                dòng ngắn kết thúc bằng dấu hai chấm
 *     • hành động (ưu tiên cao)    gạch đầu dòng, có thể kèm mức ưu tiên
 *       Lý do: ...                 dòng giải thích thụt lề dưới gạch đầu dòng
 *     ⚠️ ...                       đoạn cảnh báo
 *     ... tham khảo ... không phải  câu miễn trừ trách nhiệm
 *
 * Module này KHÔNG đổi nội dung — chỉ nhận dạng cấu trúc và đánh dấu chỗ cần
 * nhấn. Nút "Đọc bằng giọng nói" vẫn đọc nguyên văn `content`, không đi qua
 * đây. Tách khỏi component để kiểm được bằng Node thuần, không cần DOM.
 */

export type Tone = 'danger' | 'caution' | 'positive'
export type Priority = 'high' | 'medium' | 'low'

export type Token =
  | { kind: 'text'; value: string }
  /** `**đậm**` — phòng khi còn sót Markdown ở nguồn nào đó. */
  | { kind: 'strong'; value: string }
  /** Số tiền, tỉ lệ phần trăm, thời hạn — thứ người dùng quét mắt tìm. */
  | { kind: 'number'; value: string }
  /** Cụm kết luận có sắc thái: "mức cần xử lý ngay", "rủi ro thấp"… */
  | { kind: 'tone'; value: string; tone: Tone }

export interface ListItem {
  text: string
  priority?: Priority
  reason?: string
}

export type Block =
  | { kind: 'title'; icon: string; label: string; value: string }
  | { kind: 'heading'; icon: string; label: string }
  | { kind: 'list'; items: ListItem[] }
  | { kind: 'paragraph'; text: string }
  | { kind: 'warning'; text: string }
  | { kind: 'disclaimer'; text: string }

// --------------------------------------------------------------- tiêu đề mục

/**
 * Icon cho dòng mục theo từ khoá. Thứ tự có ý nghĩa: "Lý do" xét trước
 * "khuyến nghị" vì một dòng có thể chứa cả hai.
 */
const HEADING_ICONS: ReadonlyArray<readonly [RegExp, string]> = [
  [/lý do/i, '💡'],
  [/lưu ý|cảnh báo/i, '📝'],
  [/bổ sung/i, '🎯'],
  [/nên làm|hành động|khuyến nghị|ưu tiên/i, '⚠️'],
  [/yếu tố/i, '🔍'],
  [/xác suất|kết quả|đánh giá/i, '📊'],
]
const DEFAULT_HEADING_ICON = '📌'

/**
 * Nhãn nhóm cho phần giải thích mở đầu khi câu trả lời có các mục bên dưới
 * nhưng không tự có tiêu đề (bản do LLM viết). Chỉ là nhãn hiển thị, không
 * phải nội dung server trả về.
 */
export const RESULT_HEADING = { icon: '📊', label: 'Kết quả đánh giá' } as const

export function headingIcon(label: string): string {
  const hit = HEADING_ICONS.find(([pattern]) => pattern.test(label))
  return hit ? hit[1] : DEFAULT_HEADING_ICON
}

// ------------------------------------------------------------ nhận dạng dòng

// Emoji đầu dòng, kể cả biến thể có VS16 (U+FE0F, như ⚠️ ⚖️), keycap (U+20E3)
// và chuỗi ghép bằng ZWJ (U+200D).
const EMOJI_LEAD =
  /^(\p{Extended_Pictographic}(?:\uFE0F|\u20E3)?(?:\u200D\p{Extended_Pictographic}\uFE0F?)*)\s*/u
const BULLET = /^[•\-–*]\s+/
const REASON = /^lý do\s*:\s*/iu
const PRIORITY = /\s*\(\s*ưu tiên\s+(cao|vừa|thấp)\s*\)\s*[.。]?$/iu
const DISCLAIMER = /tham khảo/iu
const DISCLAIMER_TAIL = /không phải|không thay thế/iu

const PRIORITY_BY_WORD: Record<string, Priority> = {
  cao: 'high',
  vừa: 'medium',
  thấp: 'low',
}

/** Dòng ngắn kết thúc bằng dấu hai chấm là tiêu đề của một nhóm nội dung. */
function isHeading(line: string): boolean {
  return line.endsWith(':') && line.length <= 72 && !BULLET.test(line)
}

function isDisclaimer(line: string): boolean {
  return DISCLAIMER.test(line) && DISCLAIMER_TAIL.test(line)
}

function parseItem(raw: string): ListItem {
  const match = PRIORITY.exec(raw)
  if (!match) return { text: raw }
  return {
    text: raw.slice(0, match.index).trim(),
    priority: PRIORITY_BY_WORD[match[1].toLowerCase()],
  }
}

export function parseAnswer(content: string): Block[] {
  const blocks: Block[] = []
  const lines = content.replace(/\r\n?/g, '\n').split('\n')

  const last = () => blocks[blocks.length - 1]

  for (const rawLine of lines) {
    const line = rawLine.trim()
    if (!line) continue

    // Gạch đầu dòng — gom liên tiếp vào một danh sách.
    if (BULLET.test(line)) {
      const item = parseItem(line.replace(BULLET, ''))
      const prev = last()
      if (prev?.kind === 'list') prev.items.push(item)
      else blocks.push({ kind: 'list', items: [item] })
      continue
    }

    // "Lý do: …" thụt lề dưới gạch đầu dòng → gắn vào mục ngay trên.
    if (REASON.test(line)) {
      const prev = last()
      if (prev?.kind === 'list') {
        const item = prev.items[prev.items.length - 1]
        const reason = line.replace(REASON, '')
        item.reason = item.reason ? `${item.reason} ${reason}` : reason
        continue
      }
    }

    const emoji = EMOJI_LEAD.exec(line)
    if (emoji) {
      const rest = line.slice(emoji[0].length)
      if (emoji[1] === '⚠️' || emoji[1] === '⚠') {
        blocks.push({ kind: 'warning', text: rest })
        continue
      }
      if (rest.length <= 120) {
        const colon = rest.indexOf(':')
        blocks.push({
          kind: 'title',
          icon: emoji[1],
          label: colon === -1 ? rest : rest.slice(0, colon).trim(),
          value: colon === -1 ? '' : rest.slice(colon + 1).trim(),
        })
        continue
      }
    }

    if (isHeading(line)) {
      const label = line.slice(0, -1).trim()
      blocks.push({ kind: 'heading', icon: headingIcon(label), label })
      continue
    }

    if (isDisclaimer(line)) {
      blocks.push({ kind: 'disclaimer', text: line })
      continue
    }

    blocks.push({ kind: 'paragraph', text: line })
  }

  // Phần giải thích mở đầu không có tiêu đề riêng nhưng phía dưới có các mục
  // → gắn nhãn nhóm, để cả câu trả lời cùng một cấp phân đoạn.
  if (blocks[0]?.kind === 'paragraph' && blocks.some((b) => b.kind === 'heading')) {
    blocks.unshift({ kind: 'heading', ...RESULT_HEADING })
  }

  return blocks
}

// ------------------------------------------------------------ nhấn trong dòng

/**
 * Cụm kết luận cần tô sắc thái. Chỉ lấy các cụm mang tính KẾT LUẬN — nhãn
 * trạng thái của rule (`STATUS_VI`), nhãn nhóm ML01/ML02, và vài cách LLM hay
 * diễn đạt lại. Không lấy từ đơn như "an toàn" hay "khẩn cấp": chúng xuất
 * hiện trong cả cụm trung tính ("ngưỡng an toàn", "quỹ dự phòng khẩn cấp"),
 * tô hết thì đoạn văn loang lổ mà không nói thêm được gì.
 */
const TONE_PHRASES: ReadonlyArray<readonly [Tone, ReadonlyArray<string>]> = [
  ['danger', [
    'mức cần xử lý ngay',
    'cần can thiệp xử lý ngay',
    'cần xử lý ngay',
    'cần xử lý khẩn cấp dòng tiền',
    'rủi ro cao',
    'thâm hụt',
    'đang thiếu hụt',
    'cao bất thường',
    'vượt ngưỡng an toàn',
    'vượt quá thu nhập',
    'vượt khả năng trả',
    'chưa nên vay thêm lúc này',
    'mất cân đối',
  ]],
  ['caution', [
    'cần lưu ý',
    'cần tập trung xử lý nợ',
    'cần xây dựng quỹ dự phòng',
    'chi vượt mức khuyến nghị',
    'phần để dành còn thấp hơn khuyến nghị',
    'khá chật vật',
    'chưa khả thi',
    'chưa an toàn',
  ]],
  ['positive', [
    'rủi ro thấp',
    'rất lành mạnh',
    'lành mạnh',
    'rất tốt',
    'ổn định',
    'trong tầm kiểm soát',
    'đạt chuẩn',
    'có thể hướng tới tăng trưởng',
    'khoản vay nằm trong khả năng trả',
    'đủ điều kiện vay',
    'đã đạt mục tiêu',
    'trong tầm với',
    'chia hợp lý',
  ]],
]

const TONE_BY_PHRASE = new Map<string, Tone>(
  TONE_PHRASES.flatMap(([tone, phrases]) => phrases.map((p) => [p, tone] as const)),
)

const escapeRegExp = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')

// Cụm dài xếp trước để "mức cần xử lý ngay" thắng "cần xử lý ngay".
const TONE_ALTERNATION = [...TONE_BY_PHRASE.keys()]
  .sort((a, b) => b.length - a.length)
  .map(escapeRegExp)
  .join('|')

/**
 * Số có đơn vị. Không bắt số trần ("năm 2026", "3 đến 6 tháng" chỉ bắt "6
 * tháng"): số không đơn vị hiếm khi là thứ người dùng cần quét, mà bắt nhầm
 * thì năm tháng và mã số cũng bị tô.
 */
const NUMBER = [
  // 50.000.000 đồng · 3.000.000đ · 1.000.000.000
  String.raw`\d{1,3}(?:\.\d{3})+(?:,\d+)?(?:\s?(?:đồng|VNĐ|VND|đ)(?![\p{L}]))?`,
  // 3 tỷ · 500 triệu đồng · 1,5 tỷ
  String.raw`\d+(?:[.,]\d+)?\s?(?:tỷ|triệu|nghìn|ngàn)(?:\s?đồng)?(?![\p{L}])`,
  // 60% · 97,78 %
  String.raw`\d+(?:[.,]\d+)?\s?%`,
  // 100 tháng · 20 năm · 2,5 tháng
  String.raw`\d+(?:[.,]\d+)?\s?(?:tháng|năm|ngày|tuần)(?![\p{L}])`,
  // 500đ · 2000 VNĐ (không có dấu chấm ngăn nghìn)
  String.raw`\d+\s?(?:đồng|VNĐ|VND|đ)(?![\p{L}])`,
].join('|')

const INLINE = new RegExp(
  [
    String.raw`(?<strong>\*\*[^*\n]+\*\*)`,
    String.raw`(?<number>${NUMBER})`,
    String.raw`(?<![\p{L}])(?<tone>${TONE_ALTERNATION})(?![\p{L}])`,
  ].join('|'),
  'giu',
)

export function tokenize(text: string): Token[] {
  const tokens: Token[] = []
  let cursor = 0

  for (const match of text.matchAll(INLINE)) {
    const index = match.index ?? 0
    if (index > cursor) tokens.push({ kind: 'text', value: text.slice(cursor, index) })

    const groups = match.groups ?? {}
    if (groups.strong) {
      tokens.push({ kind: 'strong', value: groups.strong.slice(2, -2) })
    } else if (groups.number) {
      tokens.push({ kind: 'number', value: groups.number })
    } else if (groups.tone) {
      const tone = TONE_BY_PHRASE.get(groups.tone.toLowerCase()) ?? 'caution'
      tokens.push({ kind: 'tone', value: groups.tone, tone })
    }
    cursor = index + match[0].length
  }

  if (cursor < text.length) tokens.push({ kind: 'text', value: text.slice(cursor) })
  return tokens
}

/**
 * Tách "Nhãn: phần còn lại" ở đầu một gạch đầu dòng để in đậm nhãn. Nhãn
 * phải ngắn và không chứa số — "Dòng tiền hằng tháng: Dư khoảng 3.000.000đ"
 * có nhãn, còn "Chi 10.000.000 đồng: quá nhiều" thì không.
 */
const KEY_VALUE = /^([^:\d]{2,48}?):\s+(.+)$/su

export function splitLabel(text: string): { label: string; rest: string } | null {
  const match = KEY_VALUE.exec(text)
  return match ? { label: match[1], rest: match[2] } : null
}
