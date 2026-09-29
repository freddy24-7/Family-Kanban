import { useState } from 'react'
import { useBacklog, useSetLabels } from '../api/hooks'
import { CATEGORIES, EFFORTS, type Category, type Effort, type Topic } from '../api/types'
import { REVIEW_THRESHOLD, TopicLabels } from '../components/badges'
import { Button, Card, Empty, ErrorText, PageTitle, Segmented, Spinner } from '../components/ui'
import { categoryLabels, effortHints, effortLabels, nl } from '../i18n/nl'
import { useCurrentHousehold } from '../state/household'

const needsCheck = (t: Topic) => !t.labels.category || !t.labels.effort

export function BacklogPage() {
  const { household, isPlanner } = useCurrentHousehold()
  const backlog = useBacklog(household.id)
  const [filter, setFilter] = useState<'review' | 'all'>('review')
  const [open, setOpen] = useState<string | null>(null)

  if (backlog.isLoading) return <Spinner />
  const topics = backlog.data ?? []
  const shown = filter === 'review' ? topics.filter(needsCheck) : topics

  return (
    <div>
      <PageTitle>{nl.backlog.title}</PageTitle>
      <div className="mb-4">
        <Segmented
          label={nl.backlog.title}
          value={filter}
          onChange={setFilter}
          options={[
            {
              value: 'review',
              label: `${nl.backlog.toReview} (${topics.filter(needsCheck).length})`,
            },
            { value: 'all', label: `${nl.backlog.all} (${topics.length})` },
          ]}
        />
      </div>
      <ErrorText error={backlog.error} />
      {shown.length === 0 ? (
        <Empty>{filter === 'review' ? nl.backlog.emptyReview : nl.backlog.empty}</Empty>
      ) : (
        <ul className="space-y-2">
          {shown.map((t) => (
            <li key={t.id}>
              <Card className="p-0">
                <button
                  className="w-full space-y-2 p-4 text-left"
                  onClick={() => setOpen(open === t.id ? null : t.id)}
                  aria-expanded={open === t.id}
                >
                  <div className="flex items-start justify-between gap-3">
                    <span>{t.text}</span>
                    {t.due_by && <span className="shrink-0 text-xs text-ink-3">{t.due_by}</span>}
                  </div>
                  <TopicLabels topic={t} />
                </button>
                {open === t.id && (
                  <div className="border-t border-line p-4">
                    {isPlanner ? (
                      <LabelEditor topic={t} onDone={() => setOpen(null)} />
                    ) : (
                      <p className="text-sm text-ink-3">{nl.backlog.plannerOnly}</p>
                    )}
                  </div>
                )}
              </Card>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

/** Confirm or correct. Starts from the confirmed labels, else from the prediction;
 *  an uncertain predicted field starts EMPTY so it must be chosen consciously
 *  (guards against rubber-stamping, which would inflate measured accuracy). */
function LabelEditor({ topic, onDone }: { topic: Topic; onDone: () => void }) {
  const { household } = useCurrentHousehold()
  const setLabels = useSetLabels(household.id)
  const p = topic.prediction
  const initialCategory =
    topic.labels.category ?? ((p.category_confidence ?? 0) >= REVIEW_THRESHOLD ? p.category : null)
  const initialEffort =
    topic.labels.effort ?? ((p.effort_confidence ?? 0) >= REVIEW_THRESHOLD ? p.effort : null)
  const [category, setCategory] = useState<Category | null>(initialCategory)
  const [effort, setEffort] = useState<Effort | null>(initialEffort)

  async function confirm() {
    if (!category || !effort) return
    await setLabels.mutateAsync({ topicId: topic.id, category, effort })
    onDone()
  }

  return (
    <div className="space-y-4">
      <p className="text-sm text-ink-3">{nl.backlog.confirmHelp}</p>
      <div className="space-y-2">
        <div className="text-sm font-medium text-ink-2">{nl.backlog.category}</div>
        <Segmented
          label={nl.backlog.category}
          value={category}
          onChange={setCategory}
          options={CATEGORIES.map((c) => ({ value: c, label: categoryLabels[c] }))}
        />
      </div>
      <div className="space-y-2">
        <div className="text-sm font-medium text-ink-2">{nl.backlog.effort}</div>
        <Segmented
          label={nl.backlog.effort}
          value={effort}
          onChange={setEffort}
          options={EFFORTS.map((e) => ({
            value: e,
            label: `${effortLabels[e]} · ${effortHints[e]}`,
          }))}
        />
      </div>
      <ErrorText error={setLabels.error} />
      <Button
        onClick={confirm}
        disabled={!category || !effort || setLabels.isPending}
        className="w-full"
      >
        {nl.backlog.confirm}
      </Button>
    </div>
  )
}
