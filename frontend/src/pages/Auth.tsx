import { useState, type FormEvent, type ReactNode } from 'react'
import { Link, useNavigate, useSearchParams } from 'react-router-dom'
import { api, ApiError } from '../api/client'
import { Button, Card, ErrorText, Field, Input } from '../components/ui'
import { nl } from '../i18n/nl'
import { useAuth } from '../state/auth'

export function AuthShell({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div className="mx-auto flex min-h-dvh max-w-sm flex-col justify-center px-4 py-10">
      <div className="mb-6 flex items-center gap-3">
        <img src="/favicon.svg" alt="" className="size-10" />
        <div>
          <div className="text-xl font-semibold">{nl.appName}</div>
          <div className="text-sm text-ink-3">{nl.tagline}</div>
        </div>
      </div>
      <Card>
        <h1 className="mb-4 text-lg font-semibold">{title}</h1>
        {children}
      </Card>
    </div>
  )
}

function Message({ children }: { children: ReactNode }) {
  return <p className="rounded-xl bg-surface-2 px-3 py-2 text-sm text-ink-2">{children}</p>
}

/** Where to go after logging in (e.g. back to an invite link). */
function useNext() {
  const [params] = useSearchParams()
  const next = params.get('next')
  return next && next.startsWith('/') ? next : '/'
}

export function LoginPage() {
  const { login } = useAuth()
  const navigate = useNavigate()
  const next = useNext()
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  async function submit(e: FormEvent) {
    e.preventDefault()
    setBusy(true)
    setError(null)
    try {
      await login(email.trim(), password)
      navigate(next, { replace: true })
    } catch (err) {
      setError(
        err instanceof ApiError && err.status === 400 ? nl.auth.badCredentials : nl.common.error,
      )
    } finally {
      setBusy(false)
    }
  }

  return (
    <AuthShell title={nl.auth.login}>
      <form onSubmit={submit} className="space-y-4">
        <Field label={nl.auth.email}>
          <Input
            type="email"
            autoComplete="email"
            required
            value={email}
            onChange={(e) => setEmail(e.target.value)}
          />
        </Field>
        <Field label={nl.auth.password}>
          <Input
            type="password"
            autoComplete="current-password"
            required
            value={password}
            onChange={(e) => setPassword(e.target.value)}
          />
        </Field>
        {error && <ErrorText error={new ApiError(0, error)} />}
        <Button type="submit" disabled={busy} className="w-full">
          {nl.auth.login}
        </Button>
      </form>
      <div className="mt-4 flex justify-between text-sm">
        <Link
          className="text-accent"
          to={`/register${next !== '/' ? `?next=${encodeURIComponent(next)}` : ''}`}
        >
          {nl.auth.register}
        </Link>
        <Link className="text-ink-3" to="/forgot-password">
          {nl.auth.forgot}
        </Link>
      </div>
    </AuthShell>
  )
}

export function RegisterPage() {
  const { register } = useAuth()
  const navigate = useNavigate()
  const next = useNext()
  const [name, setName] = useState('')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  async function submit(e: FormEvent) {
    e.preventDefault()
    if (password.length < 10) return setError(nl.auth.passwordTooShort)
    setBusy(true)
    setError(null)
    try {
      await register(name.trim(), email.trim(), password)
      navigate(next, { replace: true })
    } catch (err) {
      const taken = err instanceof ApiError && err.detail === 'REGISTER_USER_ALREADY_EXISTS'
      setError(taken ? nl.auth.emailTaken : nl.common.error)
    } finally {
      setBusy(false)
    }
  }

  return (
    <AuthShell title={nl.auth.register}>
      <form onSubmit={submit} className="space-y-4">
        <Field label={nl.auth.name}>
          <Input
            autoComplete="given-name"
            required
            value={name}
            onChange={(e) => setName(e.target.value)}
          />
        </Field>
        <Field label={nl.auth.email}>
          <Input
            type="email"
            autoComplete="email"
            required
            value={email}
            onChange={(e) => setEmail(e.target.value)}
          />
        </Field>
        <Field label={nl.auth.password} hint={nl.auth.passwordHint}>
          <Input
            type="password"
            autoComplete="new-password"
            required
            value={password}
            onChange={(e) => setPassword(e.target.value)}
          />
        </Field>
        {error && <ErrorText error={new ApiError(0, error)} />}
        <Button type="submit" disabled={busy} className="w-full">
          {nl.auth.register}
        </Button>
      </form>
      <p className="mt-4 text-sm text-ink-3">
        {nl.auth.haveAccount}{' '}
        <Link
          className="text-accent"
          to={`/login${next !== '/' ? `?next=${encodeURIComponent(next)}` : ''}`}
        >
          {nl.auth.login}
        </Link>
      </p>
    </AuthShell>
  )
}

export function ForgotPasswordPage() {
  const [email, setEmail] = useState('')
  const [sent, setSent] = useState(false)
  async function submit(e: FormEvent) {
    e.preventDefault()
    await api('/auth/forgot-password', { method: 'POST', body: { email: email.trim() } }).catch(
      () => {},
    )
    setSent(true) // same message either way: don't reveal which emails exist
  }
  return (
    <AuthShell title={nl.auth.forgotTitle}>
      {sent ? (
        <Message>{nl.auth.forgotSent}</Message>
      ) : (
        <form onSubmit={submit} className="space-y-4">
          <Field label={nl.auth.email}>
            <Input type="email" required value={email} onChange={(e) => setEmail(e.target.value)} />
          </Field>
          <Button type="submit" className="w-full">
            {nl.auth.forgotSend}
          </Button>
        </form>
      )}
      <Link to="/login" className="mt-4 block text-sm text-accent">
        {nl.auth.login}
      </Link>
    </AuthShell>
  )
}

export function ResetPasswordPage() {
  const [params] = useSearchParams()
  const [password, setPassword] = useState('')
  const [state, setState] = useState<'form' | 'done' | 'invalid' | 'short'>('form')
  async function submit(e: FormEvent) {
    e.preventDefault()
    if (password.length < 10) return setState('short')
    try {
      await api('/auth/reset-password', {
        method: 'POST',
        body: { token: params.get('token'), password },
      })
      setState('done')
    } catch {
      setState('invalid')
    }
  }
  return (
    <AuthShell title={nl.auth.resetTitle}>
      {state === 'done' ? (
        <Message>{nl.auth.resetDone}</Message>
      ) : state === 'invalid' ? (
        <Message>{nl.auth.resetInvalid}</Message>
      ) : (
        <form onSubmit={submit} className="space-y-4">
          <Field label={nl.auth.password} hint={nl.auth.passwordHint}>
            <Input
              type="password"
              autoComplete="new-password"
              required
              value={password}
              onChange={(e) => setPassword(e.target.value)}
            />
          </Field>
          {state === 'short' && <ErrorText error={new ApiError(0, nl.auth.passwordTooShort)} />}
          <Button type="submit" className="w-full">
            {nl.common.save}
          </Button>
        </form>
      )}
      <Link to="/login" className="mt-4 block text-sm text-accent">
        {nl.auth.login}
      </Link>
    </AuthShell>
  )
}

export function VerifyEmailPage() {
  const [params] = useSearchParams()
  const [state, setState] = useState<'idle' | 'done' | 'invalid'>('idle')
  async function verify() {
    try {
      await api('/auth/verify', { method: 'POST', body: { token: params.get('token') } })
      setState('done')
    } catch {
      setState('invalid')
    }
  }
  return (
    <AuthShell title={nl.auth.verifyTitle}>
      {state === 'idle' && (
        // A click (not an automatic request) so email scanners opening the link don't verify it.
        <Button onClick={verify} className="w-full">
          {nl.auth.verifyTitle}
        </Button>
      )}
      {state === 'done' && <Message>{nl.auth.verifyDone}</Message>}
      {state === 'invalid' && <Message>{nl.auth.verifyInvalid}</Message>}
      <Link to="/" className="mt-4 block text-sm text-accent">
        {nl.appName}
      </Link>
    </AuthShell>
  )
}
