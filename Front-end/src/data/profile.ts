/** The four top-level pages, in nav order. */
export type PageKey = 'home' | 'info' | 'chatbot' | 'proposal'

export type AssetKey = 'nha' | 'xe' | 'dat'
export type NeedKey = 'muaNha' | 'muaDat' | 'muaXe' | 'vayVon'

/** Household profile collected on the "Nhập thông tin" form. */
export interface HouseholdProfile {
  name: string
  birthYear: string
  members: number
  children: number
  location: string
  income: number
  spending: number
  hasDebt: boolean
  debt: number
  hasSavings: boolean
  savings: number
  supportingElderly: boolean
  assets: AssetKey[]
  needs: NeedKey[]
}

export const emptyProfile: HouseholdProfile = {
  name: '',
  birthYear: '',
  members: 1,
  // Backend bắt buộc children_count < household_size, nên mặc định phải là 0.
  children: 0,
  location: '',
  income: 0,
  spending: 0,
  hasDebt: true,
  debt: 0,
  hasSavings: true,
  savings: 0,
  supportingElderly: true,
  assets: [],
  needs: [],
}

/** Sample household matching the design mockups (Nguyễn Văn A). */
export const sampleProfile: HouseholdProfile = {
  name: 'Nguyễn Văn A',
  birthYear: '1991',
  members: 5,
  children: 2,
  location: 'TP.Hồ Chí Minh',
  income: 35_000_000,
  spending: 17_000_000,
  hasDebt: true,
  debt: 500_000_000,
  hasSavings: true,
  savings: 150_000_000,
  supportingElderly: true,
  assets: [],
  needs: ['muaNha'],
}

export const ASSET_LABELS: Record<AssetKey, string> = {
  nha: 'Nhà',
  xe: 'Xe',
  dat: 'Đất',
}

export const NEED_LABELS: Record<NeedKey, string> = {
  muaNha: 'Mua nhà',
  muaDat: 'Mua Đất',
  muaXe: 'Mua Xe',
  vayVon: 'Vay vốn',
}
