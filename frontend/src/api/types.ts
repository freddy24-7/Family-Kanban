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
