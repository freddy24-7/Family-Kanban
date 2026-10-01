// Mirrors backend/app/schemas.py and backend/app/auth.py.
export type Category =
  'chores' | 'groceries' | 'kids' | 'home_maintenance' | 'finance' | 'social' | 'other'
export type Effort = 'S' | 'M' | 'L'
export type ItemStatus = 'todo' | 'in_progress' | 'done'
export type SprintStatus = 'planned' | 'active' | 'completed'

export const CATEGORIES: Category[] = [
  'chores',
  'groceries',
  'kids',
  'home_maintenance',
  'finance',
  'social',
  'other',
]
export const EFFORTS: Effort[] = ['S', 'M', 'L']

export interface User {
  id: string
  email: string
  display_name: string
  is_active: boolean
  is_superuser: boolean
  is_verified: boolean
}

export interface Household {
  id: string
  name: string
  kind: 'real' | 'simulated'
  training_eligible: boolean
  created_at: string
}

export interface Member {
  user_id: string
  display_name: string
  email: string
  is_planner: boolean
  is_reviewer: boolean
  is_child: boolean
  is_simulated: boolean
}

export interface HouseholdDetail extends Household {
  members: Member[]
}

export interface Invite {
  id: string
  household_id: string
  email: string
  is_planner: boolean
  is_reviewer: boolean
  is_child: boolean
  expires_at: string
  accepted_at: string | null
}

export interface Prediction {
  category: Category | null
  category_confidence: number | null
  effort: Effort | null
  effort_confidence: number | null
  needs_review: boolean
}

export interface Labels {
  category: Category | null
  effort: Effort | null
  source: 'planner' | 'review' | 'generator' | null
  labeled_at: string | null
}

export interface Topic {
  id: string
  household_id: string
  text: string
  due_by: string | null
  created_by: string | null
  occurred_at: string
  source: 'real' | 'simulated'
  prediction: Prediction
  labels: Labels
}

export interface SprintItem {
  id: string
  topic: Topic
  assignee_id: string | null
  assignee_name: string | null
  status: ItemStatus
  position: number
  completed: boolean | null
  effort_actual: Effort | null
  review_note: string | null
  reviewed_at: string | null
}

export interface Sprint {
  id: string
  name: string
  start_date: string
  end_date: string
  status: SprintStatus
  started_at: string | null
  completed_at: string | null
  item_count: number
  done_count: number
}

export interface SprintDetail extends Sprint {
  items: SprintItem[]
  review: {
    notes: string | null
    completed_count: number
    total_count: number
    created_at: string
  } | null
}

export interface WeekStats {
  week: number
  monday: string
  events: string[]
  n: number
  true_category_acc: number | null
  true_effort_acc: number | null
  effort_mae: number | null
  measured_category_acc: number | null
  mean_category_conf: number | null
  mean_effort_conf: number | null
  flagged_rate: number | null
  mean_words: number | null
  true_mix: Record<string, number>
  predicted_mix: Record<string, number>
  labelled: number
  sprint_items: number
  sprint_done: number
}

export interface SimulationRun {
  id: string
  household_id: string
  pool_run_id: string | null
  scenario: {
    name: string
    description: string
    events: { week: number; kind: string; note: string }[]
  }
  planner: {
    check_rate: number
    rubber_stamp_rate: number
    label_error_rate: number
    capacity_per_person: number
  }
  start_date: string
  weeks: number
  random_seed: number
  model_versions: Record<string, string> | null
  adaptation: boolean
  status: 'pool_pending' | 'pool_ready' | 'running' | 'completed' | 'failed'
  current_week: number
  tokens_in: number
  tokens_out: number
  error: string | null
  created_at: string
  finished_at: string | null
}

export interface SimulationDetail extends SimulationRun {
  weekly_stats: WeekStats[]
}

export interface Accuracy {
  n: number
  value: number | null
  ci95: [number, number] | null
}

export interface MonitoringWeek {
  week: string
  n_week: number
  n_window: number
  mean_confidence: number | null
  flagged_rate: number | null
  psi_category: number | null
  p_category: number | null
  psi_confidence: number | null
  p_confidence: number | null
  psi_length: number | null
  p_length: number | null
  category_mix: Record<string, number>
  category_accuracy: Accuracy
  effort_accuracy: Accuracy
  true_category_accuracy?: Accuracy
  true_effort_accuracy?: Accuracy
  correction_rate: number | null
  effort_by_category: Record<
    string,
    {
      n: number
      value: number
      prior_n: number
      prior_value: number | null
      p_worse: number | null
    }
  >
  alarms: string[]
}

export interface MonitoringReport {
  reference_kind: 'holdout' | 'first_weeks'
  reference: { category_accuracy: number | null; effort_accuracy: number | null; n: number } | null
  n_tickets?: number
  weekly: MonitoringWeek[]
  detection: {
    false_alarms: { week: string; alarms: string[] }[]
    detections: {
      event: string
      note: string
      first_alarm_week: string | null
      delay_weeks: number | null
      detectors: string[]
    }[]
  } | null
  run?: { id: string; scenario: SimulationRun['scenario']; start_date: string }
}

export interface ModelLogEntry {
  name: string
  task: string
  status: string
  created_at: string
  training_set_size: number
  holdout_macro_f1: number | null
  holdout_macro_f1_ci95: [number, number] | null
  cv_macro_f1: number | null
  notes: string | null
}

/** An earlier, reviewed task of the same household that resembles a backlog topic. */
export interface SimilarTask {
  topic_id: string
  text: string
  similarity: number
  assignee_name: string | null
  completed: boolean | null
  effort_actual: Effort | null
  reviewed_at: string | null
}

export interface TopicSuggestion {
  topic_id: string
  similar: SimilarTask[] // most similar first; empty: nothing comparable done before
}
