import { Component, type ReactNode } from 'react'
import { nl } from '../i18n/nl'

/** A render crash shows a friendly reload card instead of a white screen. */
export class ErrorBoundary extends Component<{ children: ReactNode }, { failed: boolean }> {
  state = { failed: false }

  static getDerivedStateFromError() {
    return { failed: true }
  }

  componentDidCatch(error: unknown) {
    console.error(error)
  }

  render() {
    if (!this.state.failed) return this.props.children
    return (
      <div className="mx-auto max-w-md p-6 text-center">
        <p className="mb-4">{nl.common.error}</p>
        <button
          className="rounded-xl bg-accent px-4 py-2 text-accent-ink"
          onClick={() => location.reload()}
        >
          {nl.common.retry}
        </button>
      </div>
    )
  }
}
