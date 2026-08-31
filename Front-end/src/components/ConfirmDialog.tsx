import { useEffect, useRef } from 'react'
import { AlertTriangle, Loader2 } from 'lucide-react'

/**
 * Hộp xác nhận cho hành động không hoàn tác được.
 *
 * Vì sao là modal tự viết chứ không phải `window.confirm`
 * ---------------------------------------------------------
 * `window.confirm` chặn luồng JS của cả tab, không cho hiện trạng thái "đang
 * xoá", và không nói được hậu quả cụ thể — mà đúng chỗ này thì hậu quả là thứ
 * người dùng cần đọc trước khi bấm: xoá hồ sơ kéo theo cả lịch sử hội thoại.
 *
 * Nút xác nhận KHÔNG được autofocus. Người dùng bấm "Xóa dữ liệu" xong quen tay
 * gõ Enter là mất sạch dữ liệu — nên phím Enter mặc định không rơi vào nút huỷ
 * diệt. Escape thì đóng, vì đóng nhầm không mất gì.
 */
export default function ConfirmDialog({
  open,
  title,
  description,
  confirmLabel = 'Xác nhận',
  cancelLabel = 'Hủy',
  busy = false,
  onConfirm,
  onCancel,
}: {
  open: boolean
  title: string
  /** Nói rõ MẤT GÌ, không chỉ hỏi "bạn có chắc không". */
  description: React.ReactNode
  confirmLabel?: string
  cancelLabel?: string
  /** Đang chạy request xoá — khoá cả hai nút để không gửi hai lần. */
  busy?: boolean
  onConfirm: () => void
  onCancel: () => void
}) {
  const cancelRef = useRef<HTMLButtonElement>(null)

  // Focus vào nút HỦY khi mở: bàn phím vào chỗ an toàn, không vào chỗ xoá.
  useEffect(() => {
    if (open) cancelRef.current?.focus()
  }, [open])

  useEffect(() => {
    if (!open) return

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape' && !busy) onCancel()
    }
    window.addEventListener('keydown', onKeyDown)

    // Khoá cuộn nền: modal mở mà nền vẫn cuộn được thì người dùng cuộn mất hộp
    // thoại đi và tưởng trang bị treo.
    const previousOverflow = document.body.style.overflow
    document.body.style.overflow = 'hidden'

    return () => {
      window.removeEventListener('keydown', onKeyDown)
      document.body.style.overflow = previousOverflow
    }
  }, [open, busy, onCancel])

  if (!open) return null

  return (
    <div
      className="fixed inset-0 z-50 grid place-items-center bg-slate-900/40 p-4 backdrop-blur-sm"
      // Bấm ra nền để đóng — nhưng không đóng khi đang xoá dở.
      onClick={() => !busy && onCancel()}
    >
      <div
        role="alertdialog"
        aria-modal="true"
        aria-labelledby="confirm-dialog-title"
        className="w-full max-w-md rounded-2xl border border-slate-100 bg-white p-6 shadow-xl"
        onClick={(event) => event.stopPropagation()}
      >
        <div className="flex items-start gap-3">
          <span className="grid h-10 w-10 shrink-0 place-items-center rounded-full bg-rose-50 text-rose-600">
            <AlertTriangle size={20} />
          </span>
          <div className="min-w-0">
            <h2
              id="confirm-dialog-title"
              className="text-base font-bold text-slate-800"
            >
              {title}
            </h2>
            <div className="mt-1.5 text-sm leading-relaxed text-slate-500">
              {description}
            </div>
          </div>
        </div>

        <div className="mt-6 flex justify-end gap-3">
          <button
            ref={cancelRef}
            type="button"
            onClick={onCancel}
            disabled={busy}
            className="rounded-xl bg-slate-100 px-4 py-2 text-sm font-medium text-slate-600 hover:bg-slate-200 disabled:opacity-60"
          >
            {cancelLabel}
          </button>
          <button
            type="button"
            onClick={onConfirm}
            disabled={busy}
            className="inline-flex items-center gap-2 rounded-xl bg-rose-600 px-4 py-2 text-sm font-semibold text-white hover:bg-rose-700 disabled:opacity-60"
          >
            {busy && <Loader2 size={16} className="animate-spin" />}
            {confirmLabel}
          </button>
        </div>
      </div>
    </div>
  )
}
