import { useQueryClient } from '@tanstack/react-query'
import { useState } from 'react'
import { Link, useNavigate, useSearchParams } from 'react-router-dom'
import { api, ApiError } from '../api/client'
import { keys } from '../api/hooks'
import { Button } from '../components/ui'
import { nl } from '../i18n/nl'
import { useAuth } from '../state/auth'
import { AuthShell } from './Auth'

const errorText: Record<number, string> = {
  403: nl.invite.wrongEmail,
  404: nl.invite.invalid,
  409: nl.invite.used,
  410: nl.invite.expired,
}

export function InvitePage() {
  const { user } = useAuth()
  const [params] = useSearchParams()
  const navigate = useNavigate()
  const qc = useQueryClient()
  const token = params.get('token') ?? ''
  const [error, setError] = useState<string | null>(null)
  const next = encodeURIComponent(`/invite?token=${token}`)

  async function accept() {
    setError(null)
    try {
      await api('/invites/accept', { method: 'POST', body: { token } })
      await qc.invalidateQueries({ queryKey: keys.households })
      navigate('/', { replace: true })
    } catch (err) {
      setError((err instanceof ApiError && errorText[err.status]) || nl.common.error)
    }
  }

  return (
    <AuthShell title={nl.invite.title}>
      {!user ? (
        <div className="space-y-4">
          <p className="text-sm text-ink-2">{nl.invite.loginFirst}</p>
          <div className="flex gap-2">
            <Link to={`/login?next=${next}`} className="flex-1">
              <Button className="w-full">{nl.auth.login}</Button>
            </Link>
            <Link to={`/register?next=${next}`} className="flex-1">
              <Button variant="secondary" className="w-full">
                {nl.auth.register}
              </Button>
            </Link>
          </div>
        </div>
      ) : (
        <div className="space-y-4">
          <Button onClick={accept} className="w-full">
            {nl.invite.accept}
          </Button>
          {error && <p className="rounded-xl bg-warn-bg px-3 py-2 text-sm text-warn">{error}</p>}
        </div>
      )}
    </AuthShell>
  )
}
