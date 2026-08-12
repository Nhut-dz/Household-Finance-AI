/**
 * 34 tỉnh, thành phố trực thuộc trung ương theo sắp xếp đơn vị hành chính có
 * hiệu lực từ 01/07/2025 (6 thành phố + 28 tỉnh). Thành phố xếp trước, phần
 * còn lại theo thứ tự Bắc vào Nam.
 */
export const PROVINCES: string[] = [
  'Hà Nội',
  'Hải Phòng',
  'Huế',
  'Đà Nẵng',
  'TP. Hồ Chí Minh',
  'Cần Thơ',

  'Cao Bằng',
  'Lạng Sơn',
  'Lai Châu',
  'Điện Biên',
  'Sơn La',
  'Lào Cai',
  'Tuyên Quang',
  'Thái Nguyên',
  'Phú Thọ',
  'Bắc Ninh',
  'Quảng Ninh',
  'Hưng Yên',
  'Ninh Bình',
  'Thanh Hóa',
  'Nghệ An',
  'Hà Tĩnh',
  'Quảng Trị',
  'Quảng Ngãi',
  'Gia Lai',
  'Đắk Lắk',
  'Khánh Hòa',
  'Lâm Đồng',
  'Đồng Nai',
  'Tây Ninh',
  'Vĩnh Long',
  'Đồng Tháp',
  'An Giang',
  'Cà Mau',
]

/**
 * Các năm sinh cho phép chọn, mới nhất trước. Backend giới hạn 1900 đến năm
 * hiện tại nên danh sách bám đúng khoảng đó.
 */
export const BIRTH_YEARS: number[] = Array.from(
  { length: new Date().getFullYear() - 1900 + 1 },
  (_, i) => new Date().getFullYear() - i,
)
