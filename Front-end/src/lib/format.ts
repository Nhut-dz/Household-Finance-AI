/** Format a number as Vietnamese đồng, e.g. 35000000 -> "35.000.000 ₫". */
export const currency = (n: number) =>
  new Intl.NumberFormat('vi-VN', {
    style: 'currency',
    currency: 'VND',
    maximumFractionDigits: 0,
  }).format(n)

/** Compact "35.000.000đ" style used inside cards (no currency symbol spacing). */
export const dong = (n: number) => `${new Intl.NumberFormat('vi-VN').format(n)}đ`
