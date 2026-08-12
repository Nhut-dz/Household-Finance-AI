import { apiGet } from '../lib/api'
import { guestQuery } from '../lib/guestSession'

/** Ba khối gói có thể null khi không áp dụng cho hộ đó — FE ẩn thẻ tương ứng. */
export interface SavingsPlan {
  monthly_contribution: number | null
  term_months: number | null
  interest_rate: number | null
  note: string | null
}

export interface InvestmentPlan {
  allocation: string | null
  risk_level: string | null
  note: string | null
}

export interface LoanPlan {
  product: string | null
  credit_limit: number | null
  monthly_payment: number | null
  note: string | null
}

export interface Proposal {
  household_id: number
  summary: string | null
  overview: {
    net_income: number | null
    current_debt: number | null
    current_savings: number | null
    target_accumulation: number | null
  }
  savings_plan: SavingsPlan | null
  investment_plan: InvestmentPlan | null
  loan_plan: LoanPlan | null
  roadmap: { step: number; text: string }[]
  generated_at: string | null
}

/** Backend trả 404 khi hồ sơ chưa được nhóm Python tính phương án. */
export const getProposal = (householdId: number) =>
  apiGet<Proposal>(`/households/${householdId}/proposal${guestQuery()}`)
