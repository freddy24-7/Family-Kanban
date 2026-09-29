import { createContext, useContext, useEffect, useMemo, useState, type ReactNode } from 'react'
import { useHousehold, useHouseholds } from '../api/hooks'
import type { HouseholdDetail, Member } from '../api/types'
import { useAuth } from './auth'

const CURRENT_KEY = 'gezinsbord.household'

interface HouseholdState {
  household: HouseholdDetail
  me: Member | undefined
  isPlanner: boolean
  isReviewer: boolean
  members: Member[]
}

const HouseholdContext = createContext<HouseholdState | null>(null)

function readCurrent(): string | null {
  try {
    return localStorage.getItem(CURRENT_KEY)
  } catch {
    return null
  }
}

/** Resolves which household the user works in. Renders `fallback` while loading
 *  and `empty` when the user has no household yet. Roles here only shape the UI;
 *  the API enforces them. */
export function HouseholdProvider({
  children,
  fallback,
  empty,
}: {
  children: ReactNode
  fallback: ReactNode
  empty: ReactNode
}) {
  const { user } = useAuth()
  const households = useHouseholds()
  const [current] = useState(readCurrent)
  const list = households.data ?? []
  const id = list.find((h) => h.id === current)?.id ?? list[0]?.id

  // Remember the choice for the next visit (external storage only, no state update).
  useEffect(() => {
    if (!id) return
    try {
      localStorage.setItem(CURRENT_KEY, id)
    } catch {
      /* ignore */
    }
  }, [id])

  if (households.isLoading) return <>{fallback}</>
  if (!id) return <>{empty}</>
  return (
    <Resolved id={id} userId={user?.id} fallback={fallback}>
      {children}
    </Resolved>
  )
}

function Resolved({
  id,
  userId,
  fallback,
  children,
}: {
  id: string
  userId: string | undefined
  fallback: ReactNode
  children: ReactNode
}) {
  const detail = useHousehold(id)
  const value = useMemo(() => {
    if (!detail.data) return null
    const me = detail.data.members.find((m) => m.user_id === userId)
    return {
      household: detail.data,
      me,
      isPlanner: !!me?.is_planner,
      isReviewer: !!me?.is_reviewer,
      members: detail.data.members,
    }
  }, [detail.data, userId])
  if (!value) return <>{fallback}</>
  return <HouseholdContext.Provider value={value}>{children}</HouseholdContext.Provider>
}

export function useCurrentHousehold(): HouseholdState {
  const ctx = useContext(HouseholdContext)
  if (!ctx) throw new Error('useCurrentHousehold outside HouseholdProvider')
  return ctx
}
