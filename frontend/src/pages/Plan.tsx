import { useState, type FormEvent } from 'react'
import {
  useAddItem,
  useAssignItem,
  useBacklog,
  useBacklogSuggestions,
  useCreateSprint,
  useRemoveItem,
  useSprint,
  useSprints,
  useStartSprint,
} from '../api/hooks'
import type { Member, SimilarTask } from '../api/types'
import { TopicLabels } from '../components/badges'
import { Button, Card, Empty, ErrorText, Field, Input, PageTitle, Spinner } from '../components/ui'
import { effortLabels, nl } from '../i18n/nl'
import { nextWeek, shortDate } from '../lib/dates'
import { useCurrentHousehold } from '../state/household'

export function PlanPage() {
  const { household, isPlanner } = useCurrentHousehold()
  const sprints = useSprints(household.id)
  if (sprints.isLoading) return <Spinner />
  if (!isPlanner) return <Empty>{nl.plan.plannerOnly}</Empty>
  const planned = sprints.data?.find((s) => s.status === 'planned')
  const active = sprints.data?.find((s) => s.status === 'active')
  return (
    <div>
      <PageTitle>{nl.plan.title}</PageTitle>
      {planned ? <PlannedSprint sprintId={planned.id} blocked={!!active} /> : <CreateSprint />}
    </div>
  )
}

function CreateSprint() {
  const { household } = useCurrentHousehold()
  const create = useCreateSprint(household.id)
  const week = nextWeek()
  const [name, setName] = useState(`Week ${week.week}`)
  const [start, setStart] = useState(week.start)
  const [end, setEnd] = useState(week.end)
  function submit(e: FormEvent) {
    e.preventDefault()
    create.mutate({ name, start_date: start, end_date: end })
  }
  return (
    <Card>
      <p className="mb-4 text-sm text-ink-3">{nl.plan.noPlanned}</p>
      <form onSubmit={submit} className="space-y-4">
        <Field label={nl.plan.name}>
          <Input required value={name} onChange={(e) => setName(e.target.value)} />
        </Field>
        <div className="grid grid-cols-2 gap-3">
          <Field label={nl.plan.start}>
            <Input type="date" required value={start} onChange={(e) => setStart(e.target.value)} />
          </Field>
          <Field label={nl.plan.end}>
            <Input type="date" required value={end} onChange={(e) => setEnd(e.target.value)} />
          </Field>
        </div>
        <ErrorText error={create.error} />
        <Button type="submit" disabled={create.isPending} className="w-full">
          {nl.plan.create}
        </Button>
      </form>
    </Card>
  )
}

function AssigneeSelect({
  members,
  value,
  onChange,
}: {
  members: Member[]
  value: string | null
  onChange: (id: string | null) => void
}) {
  return (
    <select
      aria-label={nl.plan.assignee}
      value={value ?? ''}
      onChange={(e) => onChange(e.target.value || null)}
      className="min-h-10 rounded-xl border border-line bg-surface px-2 text-sm"
    >
      <option value="">{nl.common.nobody}</option>
      {members.map((m) => (
        <option key={m.user_id} value={m.user_id}>
          {m.display_name}
        </option>
      ))}
    </select>
  )
}

function PlannedSprint({ sprintId, blocked }: { sprintId: string; blocked: boolean }) {
  const { household, members } = useCurrentHousehold()
  const h = household.id
  const sprint = useSprint(h, sprintId)
  const backlog = useBacklog(h)
  const suggestions = useBacklogSuggestions(h)
  const similarByTopic = new Map((suggestions.data ?? []).map((t) => [t.topic_id, t.similar]))
  const add = useAddItem(h, sprintId)
  const assign = useAssignItem(h, sprintId)
  const remove = useRemoveItem(h, sprintId)
  const start = useStartSprint(h, sprintId)
  const [assignees, setAssignees] = useState<Record<string, string | null>>({})

  if (sprint.isLoading || backlog.isLoading) return <Spinner />
  const s = sprint.data
  if (!s) return <ErrorText error={sprint.error} />

  return (
    <div className="space-y-6">
      <Card className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <div className="font-semibold">{s.name}</div>
          <div className="text-sm text-ink-3">
            {shortDate(s.start_date)} – {shortDate(s.end_date)} · {s.item_count}
          </div>
        </div>
        <Button
          onClick={() => start.mutate()}
          disabled={blocked || s.item_count === 0 || start.isPending}
        >
          {nl.plan.start_sprint}
        </Button>
        {blocked && <p className="w-full text-sm text-warn">{nl.plan.activeExists}</p>}
        <ErrorText error={start.error} />
      </Card>

      <section>
        <h2 className="mb-2 font-semibold">{nl.plan.inSprint}</h2>
        {s.items.length === 0 ? (
          <Empty>{nl.plan.sprintEmpty}</Empty>
        ) : (
          <ul className="space-y-2">
            {s.items.map((item) => (
              <li key={item.id}>
                <Card className="flex flex-wrap items-center gap-3">
                  <div className="min-w-0 flex-1 space-y-1.5">
                    <div>{item.topic.text}</div>
                    <TopicLabels topic={item.topic} />
                  </div>
                  <AssigneeSelect
                    members={members}
                    value={item.assignee_id}
                    onChange={(id) => assign.mutate({ itemId: item.id, assignee_id: id })}
                  />
                  <Button variant="danger" onClick={() => remove.mutate(item.id)}>
                    {nl.plan.remove}
                  </Button>
                </Card>
              </li>
            ))}
          </ul>
        )}
      </section>

      <section>
        <h2 className="mb-2 font-semibold">{nl.plan.fromBacklog}</h2>
        <ErrorText error={add.error} />
        {(backlog.data ?? []).length === 0 ? (
          <Empty>{nl.plan.backlogEmpty}</Empty>
        ) : (
          <ul className="space-y-2">
            {(backlog.data ?? []).map((t) => (
              <li key={t.id}>
                <Card className="flex flex-wrap items-center gap-3">
                  <div className="min-w-0 flex-1 space-y-1.5">
                    <div>{t.text}</div>
                    <TopicLabels topic={t} />
                  </div>
                  <AssigneeSelect
                    members={members}
                    value={assignees[t.id] ?? null}
                    onChange={(id) => setAssignees({ ...assignees, [t.id]: id })}
                  />
                  <Button
                    variant="secondary"
                    disabled={add.isPending}
                    onClick={() =>
                      add.mutate({ topic_id: t.id, assignee_id: assignees[t.id] ?? null })
                    }
                  >
                    {nl.plan.add}
                  </Button>
                  <SimilarTasks tasks={similarByTopic.get(t.id) ?? []} />
                </Card>
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  )
}

/** Earlier, similar tasks of this family: who did it and how much work it really was.
 * Shown as history for the planner to judge, not as a prediction: as an effort
 * predictor, the vote of similar tasks lost to the model (docs/learning/07-embeddings.md). */
function SimilarTasks({ tasks }: { tasks: SimilarTask[] }) {
  if (tasks.length === 0) return null
  return (
    <div
      className="w-full border-t border-line pt-2 text-xs text-ink-3"
      title={nl.plan.similarHint}
    >
      <span className="font-medium">{nl.plan.similarTitle}:</span>
      <ul className="mt-0.5 space-y-0.5">
        {tasks.map((s) => (
          <li key={s.topic_id} className="truncate">
            ‘{s.text}’ · {s.assignee_name ?? nl.plan.nobody} ·{' '}
            {s.completed === false
              ? nl.plan.notFinished
              : s.effort_actual
                ? effortLabels[s.effort_actual]
                : '–'}
            {s.reviewed_at && <> · {shortDate(s.reviewed_at.slice(0, 10))}</>}
          </li>
        ))}
      </ul>
    </div>
  )
}
