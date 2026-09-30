import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { api } from './client'
import type {
  Category,
  Effort,
  Household,
  HouseholdDetail,
  Invite,
  ItemStatus,
  SimulationDetail,
  SimulationRun,
  Sprint,
  SprintDetail,
  SprintItem,
  Topic,
  User,
} from './types'

export const keys = {
  me: ['me'] as const,
  households: ['households'] as const,
  household: (h: string) => ['household', h] as const,
  backlog: (h: string) => ['backlog', h] as const,
  sprints: (h: string) => ['sprints', h] as const,
  sprint: (h: string, s: string) => ['sprint', h, s] as const,
  invites: (h: string) => ['invites', h] as const,
}

export const useMe = (enabled = true) =>
  useQuery({ queryKey: keys.me, queryFn: () => api<User>('/users/me'), enabled, retry: false })

export const useHouseholds = () =>
  useQuery({ queryKey: keys.households, queryFn: () => api<Household[]>('/households') })

export const useHousehold = (h: string) =>
  useQuery({ queryKey: keys.household(h), queryFn: () => api<HouseholdDetail>(`/households/${h}`) })

export const useBacklog = (h: string) =>
  useQuery({ queryKey: keys.backlog(h), queryFn: () => api<Topic[]>(`/households/${h}/backlog`) })

export const useSprints = (h: string) =>
  useQuery({ queryKey: keys.sprints(h), queryFn: () => api<Sprint[]>(`/households/${h}/sprints`) })

export const useSprint = (h: string, s: string | undefined) =>
  useQuery({
    queryKey: keys.sprint(h, s ?? ''),
    queryFn: () => api<SprintDetail>(`/households/${h}/sprints/${s}`),
    enabled: !!s,
  })

export const useInvites = (h: string, enabled: boolean) =>
  useQuery({
    queryKey: keys.invites(h),
    queryFn: () => api<Invite[]>(`/households/${h}/invites`),
    enabled,
  })

function useInvalidate() {
  const qc = useQueryClient()
  return (...queryKeys: readonly unknown[][]) =>
    Promise.all(queryKeys.map((queryKey) => qc.invalidateQueries({ queryKey })))
}

export function useCreateHousehold() {
  const invalidate = useInvalidate()
  return useMutation({
    mutationFn: (name: string) => api<Household>('/households', { method: 'POST', body: { name } }),
    onSuccess: () => invalidate([...keys.households]),
  })
}

export function useCreateTopic(h: string) {
  const invalidate = useInvalidate()
  return useMutation({
    mutationFn: (body: { text: string; due_by: string | null }) =>
      api<Topic>(`/households/${h}/topics`, { method: 'POST', body }),
    onSuccess: () => invalidate([...keys.backlog(h)]),
  })
}

export function useSetLabels(h: string) {
  const invalidate = useInvalidate()
  return useMutation({
    mutationFn: ({ topicId, ...body }: { topicId: string; category: Category; effort: Effort }) =>
      api<Topic>(`/households/${h}/topics/${topicId}/labels`, { method: 'PUT', body }),
    onSuccess: () => invalidate([...keys.backlog(h)], ['sprint', h]),
  })
}

export function useCreateSprint(h: string) {
  const invalidate = useInvalidate()
  return useMutation({
    mutationFn: (body: { name: string; start_date: string; end_date: string }) =>
      api<SprintDetail>(`/households/${h}/sprints`, { method: 'POST', body }),
    onSuccess: () => invalidate([...keys.sprints(h)]),
  })
}

function useSprintMutation<TVars>(h: string, s: string, fn: (vars: TVars) => Promise<unknown>) {
  const invalidate = useInvalidate()
  return useMutation({
    mutationFn: fn,
    onSuccess: () => invalidate([...keys.sprint(h, s)], [...keys.sprints(h)], [...keys.backlog(h)]),
  })
}

export const useAddItem = (h: string, s: string) =>
  useSprintMutation(h, s, (body: { topic_id: string; assignee_id: string | null }) =>
    api<SprintItem>(`/households/${h}/sprints/${s}/items`, { method: 'POST', body }),
  )

export const useAssignItem = (h: string, s: string) =>
  useSprintMutation(
    h,
    s,
    ({ itemId, assignee_id }: { itemId: string; assignee_id: string | null }) =>
      api<SprintItem>(`/households/${h}/sprints/${s}/items/${itemId}`, {
        method: 'PATCH',
        body: { assignee_id },
      }),
  )

export const useRemoveItem = (h: string, s: string) =>
  useSprintMutation(h, s, (itemId: string) =>
    api<void>(`/households/${h}/sprints/${s}/items/${itemId}`, { method: 'DELETE' }),
  )

export const useStartSprint = (h: string, s: string) =>
  useSprintMutation(h, s, () =>
    api<SprintDetail>(`/households/${h}/sprints/${s}/start`, { method: 'POST' }),
  )

export function useMoveItem(h: string, s: string) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ itemId, status }: { itemId: string; status: ItemStatus }) =>
      api<SprintItem>(`/households/${h}/sprints/${s}/items/${itemId}/status`, {
        method: 'PUT',
        body: { status, position: 0 },
      }),
    // Optimistic: the card moves immediately; rolled back if the server refuses.
    onMutate: async ({ itemId, status }) => {
      await qc.cancelQueries({ queryKey: keys.sprint(h, s) })
      const previous = qc.getQueryData<SprintDetail>(keys.sprint(h, s))
      if (previous) {
        qc.setQueryData<SprintDetail>(keys.sprint(h, s), {
          ...previous,
          items: previous.items.map((i) => (i.id === itemId ? { ...i, status } : i)),
        })
      }
      return { previous }
    },
    onError: (_err, _vars, context) => {
      if (context?.previous) qc.setQueryData(keys.sprint(h, s), context.previous)
    },
    onSettled: () => qc.invalidateQueries({ queryKey: keys.sprint(h, s) }),
  })
}

export const useReviewItem = (h: string, s: string) =>
  useSprintMutation(
    h,
    s,
    ({
      itemId,
      ...body
    }: {
      itemId: string
      completed: boolean
      effort_actual: Effort | null
      note: string | null
    }) =>
      api<SprintItem>(`/households/${h}/sprints/${s}/items/${itemId}/review`, {
        method: 'PUT',
        body,
      }),
  )

export const useCompleteSprint = (h: string, s: string) =>
  useSprintMutation(h, s, (notes: string | null) =>
    api<SprintDetail>(`/households/${h}/sprints/${s}/complete`, {
      method: 'POST',
      body: { notes },
    }),
  )

export function useInvite(h: string) {
  const invalidate = useInvalidate()
  return useMutation({
    mutationFn: (body: {
      email: string
      is_child: boolean
      is_planner: boolean
      is_reviewer: boolean
    }) => api<Invite>(`/households/${h}/invites`, { method: 'POST', body }),
    onSuccess: () => invalidate([...keys.invites(h)]),
  })
}

export function useUpdateTopic(h: string) {
  const invalidate = useInvalidate()
  return useMutation({
    mutationFn: ({
      topicId,
      ...body
    }: {
      topicId: string
      text?: string
      due_by?: string | null
    }) => api<Topic>(`/households/${h}/topics/${topicId}`, { method: 'PATCH', body }),
    onSuccess: () => invalidate([...keys.backlog(h)], ['sprint', h]),
  })
}

export function useDeleteTopic(h: string) {
  const invalidate = useInvalidate()
  return useMutation({
    mutationFn: (topicId: string) =>
      api<void>(`/households/${h}/topics/${topicId}`, { method: 'DELETE' }),
    onSuccess: () => invalidate([...keys.backlog(h)]),
  })
}

const isRunning = (status?: string) =>
  status === 'pool_pending' || status === 'pool_ready' || status === 'running'

export const useSimulations = () =>
  useQuery({
    queryKey: ['simulations'],
    queryFn: () => api<SimulationRun[]>('/admin/simulations'),
    refetchInterval: (q) => ((q.state.data ?? []).some((r) => isRunning(r.status)) ? 5000 : false),
  })

export const useSimulation = (id: string | null) =>
  useQuery({
    queryKey: ['simulation', id],
    queryFn: () => api<SimulationDetail>(`/admin/simulations/${id}`),
    enabled: !!id,
    refetchInterval: (q) => (isRunning(q.state.data?.status) ? 5000 : false),
  })

export function useStartSimulation() {
  const invalidate = useInvalidate()
  return useMutation({
    mutationFn: (body: {
      scenario: string
      weeks: number
      start_date: string
      planner: { rubber_stamp_rate: number }
    }) => api<SimulationRun>('/admin/simulations', { method: 'POST', body }),
    onSuccess: () => invalidate(['simulations']),
  })
}

export function useReplaySimulation() {
  const invalidate = useInvalidate()
  return useMutation({
    mutationFn: ({ id, rubber_stamp_rate }: { id: string; rubber_stamp_rate: number }) =>
      api<SimulationRun>(`/admin/simulations/${id}/replay`, {
        method: 'POST',
        body: { planner: { rubber_stamp_rate } },
      }),
    onSuccess: () => invalidate(['simulations']),
  })
}
