import { useState } from 'react'
import { useCompleteSprint, useReviewItem, useSprint, useSprints } from '../api/hooks'
import { EFFORTS, type Effort, type SprintItem } from '../api/types'
import { TopicLabels } from '../components/badges'
import {
  Button,
  Card,
  Empty,
  ErrorText,
  Input,
  PageTitle,
  Segmented,
  Spinner,
  TextArea,
} from '../components/ui'
import { effortHints, effortLabels, nl } from '../i18n/nl'
import { shortDate } from '../lib/dates'
import { useCurrentHousehold } from '../state/household'

export function ReviewPage() {
  const { household, isReviewer } = useCurrentHousehold()
  const sprints = useSprints(household.id)
  if (sprints.isLoading) return <Spinner />
  const active = sprints.data?.find((s) => s.status === 'active')
  const past = (sprints.data ?? []).filter((s) => s.status === 'completed')
  return (
    <div className="space-y-6">
      <PageTitle>{nl.review.title}</PageTitle>
      {!isReviewer ? (
        <Empty>{nl.review.reviewerOnly}</Empty>
      ) : active ? (
        <ActiveReview sprintId={active.id} />
      ) : (
        <Empty>{nl.review.noActive}</Empty>
      )}
      {past.length > 0 && (
        <section>
          <h2 className="mb-2 font-semibold">{nl.review.history}</h2>
          <ul className="space-y-2">
            {past.map((s) => (
              <li key={s.id}>
                <Card className="flex justify-between gap-3 text-sm">
                  <span className="font-medium">{s.name}</span>
                  <span className="text-ink-3">
                    {shortDate(s.start_date)} – {shortDate(s.end_date)} · {s.done_count}/
                    {s.item_count}
                  </span>
                </Card>
              </li>
            ))}
          </ul>
        </section>
      )}
    </div>
  )
}

function ActiveReview({ sprintId }: { sprintId: string }) {
  const { household } = useCurrentHousehold()
  const sprint = useSprint(household.id, sprintId)
  const complete = useCompleteSprint(household.id, sprintId)
  const [notes, setNotes] = useState('')
  if (sprint.isLoading) return <Spinner />
  const s = sprint.data
  if (!s) return <ErrorText error={sprint.error} />
  const reviewed = s.items.filter((i) => i.reviewed_at).length

  return (
    <div className="space-y-4">
      <p className="text-sm text-ink-3">
        {s.name} · {nl.review.progress(reviewed, s.items.length)}
      </p>
      <ul className="space-y-3">
        {s.items.map((item) => (
          <li key={item.id}>
            <ItemReview item={item} sprintId={sprintId} />
          </li>
        ))}
      </ul>
      <Card className="space-y-3">
        <TextArea
          placeholder={nl.review.sprintNotes}
          aria-label={nl.review.sprintNotes}
          value={notes}
          onChange={(e) => setNotes(e.target.value)}
        />
        <ErrorText error={complete.error} />
        <Button
          className="w-full"
          disabled={reviewed < s.items.length || complete.isPending}
          onClick={() => complete.mutate(notes.trim() || null)}
        >
          {nl.review.complete}
        </Button>
      </Card>
    </div>
  )
}

function ItemReview({ item, sprintId }: { item: SprintItem; sprintId: string }) {
  const { household } = useCurrentHousehold()
  const review = useReviewItem(household.id, sprintId)
  const [completed, setCompleted] = useState<'yes' | 'no' | null>(
    item.completed === null
      ? item.status === 'done'
        ? 'yes'
        : null
      : item.completed
        ? 'yes'
        : 'no',
  )
  // Actual effort starts empty unless already reviewed: it's a measurement, not a confirmation.
  const [effort, setEffort] = useState<Effort | null>(item.effort_actual)
  const [note, setNote] = useState(item.review_note ?? '')
  const ready = completed === 'no' || (completed === 'yes' && effort)

  return (
    <Card className="space-y-3">
      <div className="flex items-start justify-between gap-3">
        <div className="space-y-1.5">
          <div>{item.topic.text}</div>
          <TopicLabels topic={item.topic} />
        </div>
        <span className="shrink-0 text-xs text-ink-3">
          {item.assignee_name ?? nl.common.nobody}
        </span>
      </div>
      <div className="flex flex-wrap items-center gap-3">
        <span className="text-sm text-ink-2">{nl.review.done}</span>
        <Segmented
          label={nl.review.done}
          value={completed}
          onChange={setCompleted}
          options={[
            { value: 'yes', label: nl.review.yes },
            { value: 'no', label: nl.review.no },
          ]}
        />
      </div>
      {completed === 'yes' && (
        <div className="space-y-1.5">
          <span className="text-sm text-ink-2">{nl.review.actualEffort}</span>
          <Segmented
            label={nl.review.actualEffort}
            value={effort}
            onChange={setEffort}
            options={EFFORTS.map((e) => ({
              value: e,
              label: `${effortLabels[e]} · ${effortHints[e]}`,
            }))}
          />
        </div>
      )}
      <Input
        placeholder={`${nl.review.note} (${nl.common.optional})`}
        aria-label={nl.review.note}
        value={note}
        onChange={(e) => setNote(e.target.value)}
      />
      <ErrorText error={review.error} />
      <div className="flex items-center justify-end gap-3">
        {item.reviewed_at && !review.isPending && (
          <span className="text-sm text-good">✓ {nl.review.saved}</span>
        )}
        <Button
          variant="secondary"
          disabled={!ready || review.isPending}
          onClick={() =>
            review.mutate({
              itemId: item.id,
              completed: completed === 'yes',
              effort_actual: completed === 'yes' ? effort : null,
              note: note.trim() || null,
            })
          }
        >
          {nl.common.save}
        </Button>
      </div>
    </Card>
  )
}
