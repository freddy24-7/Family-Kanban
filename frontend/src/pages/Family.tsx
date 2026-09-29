import { useState, type FormEvent } from 'react'
import { useInvite, useInvites } from '../api/hooks'
import { Button, Card, ErrorText, Field, Input, PageTitle } from '../components/ui'
import { nl } from '../i18n/nl'
import { useCurrentHousehold } from '../state/household'

export function FamilyPage() {
  const { household, members, isPlanner } = useCurrentHousehold()
  const invites = useInvites(household.id, isPlanner)
  return (
    <div className="space-y-6">
      <PageTitle>{nl.family.title}</PageTitle>
      <section>
        <h2 className="mb-2 font-semibold">{nl.family.members}</h2>
        <ul className="space-y-2">
          {members.map((m) => (
            <li key={m.user_id}>
              <Card className="flex flex-wrap items-center justify-between gap-2">
                <div>
                  <div className="font-medium">{m.display_name}</div>
                  <div className="text-xs text-ink-3">{m.email}</div>
                </div>
                <div className="flex flex-wrap gap-1.5 text-xs text-ink-2">
                  <Tag>{m.is_child ? nl.family.child : nl.family.adult}</Tag>
                  {m.is_planner && <Tag>{nl.family.planner}</Tag>}
                  {m.is_reviewer && <Tag>{nl.family.reviewer}</Tag>}
                </div>
              </Card>
            </li>
          ))}
        </ul>
      </section>
      {isPlanner && <InviteForm />}
      {isPlanner && (invites.data ?? []).length > 0 && (
        <section>
          <h2 className="mb-2 font-semibold">{nl.family.pending}</h2>
          <ul className="space-y-2">
            {invites.data!.map((i) => (
              <li key={i.id}>
                <Card className="flex justify-between gap-3 text-sm">
                  <span>{i.email}</span>
                  <span className="text-ink-3">
                    {nl.family.expires} {new Date(i.expires_at).toLocaleDateString('nl-NL')}
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

function Tag({ children }: { children: string }) {
  return <span className="rounded-full bg-surface-2 px-2 py-0.5">{children}</span>
}

function InviteForm() {
  const { household } = useCurrentHousehold()
  const invite = useInvite(household.id)
  const [email, setEmail] = useState('')
  const [isChild, setIsChild] = useState(false)
  const [canPlan, setCanPlan] = useState(false)
  const [canReview, setCanReview] = useState(false)
  const [sent, setSent] = useState(false)

  async function submit(e: FormEvent) {
    e.preventDefault()
    await invite.mutateAsync({
      email: email.trim(),
      is_child: isChild,
      is_planner: canPlan,
      is_reviewer: canReview,
    })
    setSent(true)
    setEmail('')
  }

  return (
    <section>
      <h2 className="mb-2 font-semibold">{nl.family.invite}</h2>
      <Card>
        <form onSubmit={submit} className="space-y-3">
          <Field label={nl.family.inviteEmail}>
            <Input
              type="email"
              required
              value={email}
              onChange={(e) => {
                setEmail(e.target.value)
                setSent(false)
              }}
            />
          </Field>
          <Check
            checked={isChild}
            onChange={setIsChild}
            label={nl.family.isChild}
            hint={nl.family.isChildHelp}
          />
          <Check checked={canPlan} onChange={setCanPlan} label={nl.family.canPlan} />
          <Check checked={canReview} onChange={setCanReview} label={nl.family.canReview} />
          <ErrorText error={invite.error} />
          {sent && <p className="text-sm text-good">✓ {nl.family.sent}</p>}
          <Button type="submit" disabled={invite.isPending} className="w-full">
            {nl.family.send}
          </Button>
        </form>
      </Card>
    </section>
  )
}

function Check({
  checked,
  onChange,
  label,
  hint,
}: {
  checked: boolean
  onChange: (v: boolean) => void
  label: string
  hint?: string
}) {
  return (
    <label className="flex items-start gap-3">
      <input
        type="checkbox"
        className="mt-1 size-4 accent-[var(--accent)]"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
      />
      <span>
        <span className="block text-sm">{label}</span>
        {hint && <span className="block text-xs text-ink-3">{hint}</span>}
      </span>
    </label>
  )
}
