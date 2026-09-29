import { useState, type FormEvent } from 'react'
import { useCreateTopic } from '../api/hooks'
import type { Topic } from '../api/types'
import { TopicLabels } from '../components/badges'
import { Button, Card, ErrorText, Field, Input, PageTitle, TextArea } from '../components/ui'
import { nl } from '../i18n/nl'
import { useCurrentHousehold } from '../state/household'

export function NewTopicPage() {
  const { household } = useCurrentHousehold()
  const create = useCreateTopic(household.id)
  const [text, setText] = useState('')
  const [dueBy, setDueBy] = useState('')
  const [added, setAdded] = useState<Topic | null>(null)

  async function submit(e: FormEvent) {
    e.preventDefault()
    if (!text.trim()) return
    const topic = await create.mutateAsync({ text: text.trim(), due_by: dueBy || null })
    setAdded(topic)
    setText('')
    setDueBy('')
  }

  if (added) {
    return (
      <div className="mx-auto max-w-lg">
        <PageTitle>{nl.intake.added} ✓</PageTitle>
        <Card className="space-y-3">
          <p className="text-lg">{added.text}</p>
          <div className="space-y-1.5">
            <p className="text-sm text-ink-3">{nl.intake.modelThinks}</p>
            <TopicLabels topic={added} />
          </div>
          {added.prediction.needs_review && (
            <p className="text-sm text-warn">{nl.intake.willBeChecked}</p>
          )}
        </Card>
        <Button className="mt-4 w-full" onClick={() => setAdded(null)}>
          {nl.intake.another}
        </Button>
      </div>
    )
  }

  return (
    <form onSubmit={submit} className="mx-auto max-w-lg space-y-4">
      <PageTitle>{nl.intake.title}</PageTitle>
      <TextArea
        autoFocus
        required
        maxLength={2000}
        placeholder={nl.intake.placeholder}
        aria-label={nl.intake.placeholder}
        value={text}
        onChange={(e) => setText(e.target.value)}
      />
      <Field label={`${nl.intake.dueBy} (${nl.common.optional})`}>
        <Input type="date" value={dueBy} onChange={(e) => setDueBy(e.target.value)} />
      </Field>
      <ErrorText error={create.error} />
      <Button type="submit" disabled={create.isPending || !text.trim()} className="w-full">
        {nl.intake.submit}
      </Button>
    </form>
  )
}
