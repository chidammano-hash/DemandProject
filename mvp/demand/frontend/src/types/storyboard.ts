export type ExceptionType = 'forecast_bias' | 'stockout_risk' | 'accuracy_drop' | 'excess_risk' | 'model_drift' | 'new_item'
export type ExceptionStatus = 'open' | 'investigating' | 'resolved' | 'dismissed'
export type DecisionType = 'override_forecast' | 'accept_exception' | 'escalate' | 'dismiss' | 'request_info'

export interface StoryboardException {
  exception_id: string
  exception_type: ExceptionType
  item_no: string
  loc: string
  severity: number
  financial_impact: number | null
  headline: string | null
  supporting_data: Record<string, unknown> | null
  status: ExceptionStatus
  assigned_to: string | null
  generated_at: string
  expires_at: string | null
  month_start: string | null
}

export interface StoryboardSummary {
  total_open: number
  total_investigating: number
  avg_severity: number
  by_type: Array<{ exception_type: string; open_count: number; avg_severity: number }>
  top_items: Array<{ item_no: string; loc: string; exception_count: number }>
}

export interface PlannerDecision {
  decision_id: string
  exception_id: string
  item_no: string
  loc: string
  decision_type: DecisionType
  decision_value: Record<string, unknown> | null
  rationale: string | null
  decided_by: string
  decided_at: string
}
