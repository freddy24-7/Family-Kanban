import { useState, type FormEvent } from 'react'
import { useCreateHousehold } from '../api/hooks'
import { Button, ErrorText, Field, Input } from '../components/ui'
import { nl } from '../i18n/nl'
import { useAuth } from '../state/auth'
import { AuthShell } from './Auth'

export function OnboardingPage() {
  const { logout } = useAuth()
  const create = useCreateHousehold()
  const [name, setName] = useState('')
  function submit(e: FormEvent) {
    e.preventDefault()
    if (name.trim()) create.mutate(name.trim())
  }
  return (
    <AuthShell title={nl.onboarding.title}>
      <p className="mb-4 text-sm text-ink-2">{nl.onboarding.intro}</p>
      <form onSubmit={submit} className="space-y-4">
        <Field label={nl.onboarding.householdName}>
          <Input
            required
            placeholder={nl.onboarding.householdPlaceholder}
            value={name}
            onChange={(e) => setName(e.target.value)}
          />
        </Field>
        <ErrorText error={create.error} />
        <Button type="submit" disabled={create.isPending} className="w-full">
          {nl.onboarding.create}
        </Button>
      </form>
      <button onClick={logout} className="mt-4 text-sm text-ink-3">
        {nl.auth.logout}
      </button>
    </AuthShell>
  )
}
