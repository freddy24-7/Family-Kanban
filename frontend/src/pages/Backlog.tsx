import { useState } from 'react'
import { useBacklog, useDeleteTopic, useSetLabels, useUpdateTopic } from '../api/hooks'
import { CATEGORIES, EFFORTS, type Category, type Effort, type Topic } from '../api/types'
import { REVIEW_THRESHOLD, TopicLabels } from '../components/badges'
import {
  Button,
  Card,
  Empty,
  ErrorText,
  Field,
  Input,
  PageTitle,
  Segmented,
  Spinner,
  TextArea,
} from '../components/ui'
import { categoryLabels, effortHints, effortLabels, nl } from '../i18n/nl'
import { useCurrentHousehold } from '../state/household'

const needsCheck = (t: Topic) => !t.labels.category || !t.labels.effort

export function BacklogPage() {
  const { household, isPlanner, me } = useCurrentHousehold()
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
                  <div className="space-y-4 border-t border-line p-4">
                    {isPlanner ? (
                      <LabelEditor topic={t} onDone={() => setOpen(null)} />
                    ) : (
                      <p className="text-sm text-ink-3">{nl.backlog.plannerOnly}</p>
                    )}
                    {(isPlanner || t.created_by === me?.user_id) && (
                      <TopicActions topic={t} onDone={() => setOpen(null)} />
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

/** Edit text / due date, or delete (creator or planner). Editing the text makes the
 *  model predict again; confirmed labels stay. */
function TopicActions({ topic, onDone }: { topic: Topic; onDone: () => void }) {
  const { household } = useCurrentHousehold()
  const update = useUpdateTopic(household.id)
  const remove = useDeleteTopic(household.id)
  const [mode, setMode] = useState<'idle' | 'edit' | 'delete'>('idle')
  const [text, setText] = useState(topic.text)
  const [dueBy, setDueBy] = useState(topic.due_by ?? '')

  async function save() {
    await update.mutateAsync({ topicId: topic.id, text: text.trim(), due_by: dueBy || null })
    setMode('idle')
  }

  if (mode === 'edit') {
    return (
      <div className="space-y-3 border-t border-line pt-4">
        <div className="text-sm font-medium text-ink-2">{nl.backlog.editTitle}</div>
        <TextArea
          aria-label={nl.backlog.editTitle}
          value={text}
          maxLength={2000}
          onChange={(e) => setText(e.target.value)}
        />
        <Field label={`${nl.intake.dueBy} (${nl.common.optional})`}>
          <Input type="date" value={dueBy} onChange={(e) => setDueBy(e.target.value)} />
        </Field>
        <ErrorText error={update.error} />
        <div className="flex gap-2">
          <Button onClick={save} disabled={!text.trim() || update.isPending} className="flex-1">
            {nl.common.save}
          </Button>
          <Button variant="ghost" onClick={() => setMode('idle')}>
            {nl.common.cancel}
          </Button>
        </div>
      </div>
    )
  }

  if (mode === 'delete') {
    return (
      <div className="space-y-3 border-t border-line pt-4">
        <p className="text-sm">{nl.backlog.confirmDelete}</p>
        <ErrorText error={remove.error} />
        <div className="flex gap-2">
          <Button
            variant="danger"
            className="flex-1 border border-danger"
            disabled={remove.isPending}
            onClick={async () => {
              await remove.mutateAsync(topic.id)
              onDone()
            }}
          >
            {nl.backlog.yesDelete}
          </Button>
          <Button variant="ghost" onClick={() => setMode('idle')}>
            {nl.common.cancel}
          </Button>
        </div>
      </div>
    )
  }

  return (
    <div className="flex gap-2 border-t border-line pt-4">
      <Button variant="secondary" onClick={() => setMode('edit')}>
        {nl.backlog.edit}
      </Button>
      <Button variant="danger" onClick={() => setMode('delete')}>
        {nl.backlog.delete}
      </Button>
    </div>
  )
}
