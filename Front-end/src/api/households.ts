import { apiDelete, apiGet, apiPost, apiPut } from '../lib/api'
import { getGuestSessionId, guestQuery } from '../lib/guestSession'
import { emptyProfile, type AssetKey, type HouseholdProfile, type NeedKey } from '../data/profile'

/** Nhãn tài sản của form ↔ AssetTypeEnum phía backend. */
const ASSET_TO_API: Record<AssetKey, string> = {
  nha: 'house',
  xe: 'car',
  dat: 'land',
}

/** Nhu cầu tài chính của form ↔ GoalTypeEnum phía backend. */
const NEED_TO_API: Record<NeedKey, string> = {
  muaNha: 'buy_house',
  muaXe: 'buy_car',
  muaDat: 'buy_land',
  vayVon: 'loan',
}

const ASSET_FROM_API: Record<string, AssetKey> = {
  house: 'nha',
  car: 'xe',
  land: 'dat',
}

const NEED_FROM_API: Record<string, NeedKey> = {
  buy_house: 'muaNha',
  buy_car: 'muaXe',
  buy_land: 'muaDat',
  loan: 'vayVon',
}

/** Body của POST /households, đúng tên field mà StoreHouseholdRequest chờ nhận. */
export interface StoreHouseholdPayload {
  representative_name: string
  birth_year: number | null
  household_size: number
  children_count: number
  residence: string | null
  average_monthly_income: number
  average_monthly_expense: number | null
  has_debt: boolean
  total_current_debt: number | null
  monthly_debt_payment: number | null
  has_savings: boolean
  savings_amount: number | null
  has_dependents: boolean
  assets: string[]
  financial_needs: string[]
  guest_session_id: string
}

/** Bản ghi trả về từ HouseholdResource. */
export interface HouseholdResponse extends StoreHouseholdPayload {
  id: number
  user_id: number | null
  created_at: string
  updated_at: string
}

/**
 * Response của PUT /households/{id}, có thêm trạng thái phiên trò chuyện.
 *
 * `conversation_rotated = true` nghĩa là lần sửa này chạm vào dữ liệu tài chính
 * nên backend đã đóng phiên cũ và mở phiên mới. FE phải xoá hội thoại đang hiển
 * thị — những câu trả lời đó được sinh trên số liệu không còn đúng.
 *
 * Quyết định do backend đưa ra chứ FE không tự so số liệu: chép luật nghiệp vụ
 * sang hai nơi thì sớm muộn hai nơi lệch nhau.
 */
export interface UpdateHouseholdResponse extends HouseholdResponse {
  /** Null khi hộ chưa từng trò chuyện và lần sửa này không xoay phiên. */
  conversation_id: number | null
  conversation_rotated: boolean
}

export function toStoreHouseholdPayload(
  profile: HouseholdProfile,
): StoreHouseholdPayload {
  const birthYear = Number.parseInt(profile.birthYear, 10)
  // Backend chỉ lưu số nợ khi có nợ; gửi null cho phần còn lại để tránh số mồ côi.
  const debt = profile.hasDebt ? profile.debt : null

  return {
    representative_name: profile.name.trim(),
    birth_year: Number.isNaN(birthYear) ? null : birthYear,
    household_size: profile.members,
    children_count: profile.children,
    residence: profile.location.trim() || null,
    average_monthly_income: profile.income,
    average_monthly_expense: profile.spending,
    has_debt: profile.hasDebt,
    total_current_debt: debt,
    // Cùng công thức ước tính đang hiển thị trên form.
    monthly_debt_payment: debt === null ? null : Math.round(debt / 100),
    has_savings: profile.hasSavings,
    savings_amount: profile.hasSavings ? profile.savings : null,
    has_dependents: profile.supportingElderly,
    assets: profile.assets.map((key) => ASSET_TO_API[key]),
    financial_needs: profile.needs.map((key) => NEED_TO_API[key]),
    guest_session_id: getGuestSessionId(),
  }
}

/** Chuyển bản ghi của backend về đúng shape mà form đang dùng. */
export function fromHouseholdResponse(data: HouseholdResponse): HouseholdProfile {
  return {
    ...emptyProfile,
    name: data.representative_name ?? '',
    birthYear: data.birth_year === null ? '' : String(data.birth_year),
    members: data.household_size,
    children: data.children_count,
    location: data.residence ?? '',
    income: Number(data.average_monthly_income ?? 0),
    spending: Number(data.average_monthly_expense ?? 0),
    hasDebt: Boolean(data.has_debt),
    debt: Number(data.total_current_debt ?? 0),
    hasSavings: Boolean(data.has_savings),
    savings: Number(data.savings_amount ?? 0),
    supportingElderly: Boolean(data.has_dependents),
    assets: (data.assets ?? [])
      .map((key) => ASSET_FROM_API[key])
      .filter((key): key is AssetKey => Boolean(key)),
    needs: (data.financial_needs ?? [])
      .map((key) => NEED_FROM_API[key])
      .filter((key): key is NeedKey => Boolean(key)),
  }
}

/** Lưu hồ sơ của màn "Nhập thông tin". Ném ApiError khi thất bại. */
export const createHousehold = (profile: HouseholdProfile) =>
  apiPost<HouseholdResponse>('/households', toStoreHouseholdPayload(profile))

/** Cập nhật hồ sơ đã có (nút "Sửa hồ sơ"), thay vì tạo thêm bản ghi mới. */
export const updateHousehold = (id: number, profile: HouseholdProfile) =>
  apiPut<UpdateHouseholdResponse>(
    `/households/${id}`,
    toStoreHouseholdPayload(profile),
  )

/** Hồ sơ gần nhất của phiên hiện tại. Backend trả 404 khi chưa có hồ sơ nào. */
export const getLatestHousehold = () =>
  apiGet<HouseholdResponse>(`/households/latest${guestQuery()}`)

export const getHousehold = (id: number) =>
  apiGet<HouseholdResponse>(`/households/${id}${guestQuery()}`)

/** Xoá hồ sơ kèm tài sản, nhu cầu và lịch sử hội thoại. */
export const deleteHousehold = (id: number) =>
  apiDelete<null>(`/households/${id}${guestQuery()}`)
