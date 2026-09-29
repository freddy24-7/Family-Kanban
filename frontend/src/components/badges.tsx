import type { Category, Effort, Topic } from '../api/types'
import { categoryLabels, effortHints, effortLabels, nl } from '../i18n/nl'

// Keep in sync with LOW_CONFIDENCE_THRESHOLD in the backend (chosen in Phase 3).
export const REVIEW_THRESHOLD = 0.8

export function CategoryBadge({ category, unsure }: { category: Category; unsure?: boolean }) {
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-xs font-medium ${
        unsure ? 'border-dashed border-warn text-warn' : 'border-line text-ink-2'
      }`}
    >
      <span
        className="size-2 rounded-full"
        style={{ background: `var(--cat-${category})` }}
        aria-hidden
      />
      {categoryLabels[category]}
      {unsure && <span className="sr-only">({nl.prediction.unsure})</span>}
      {unsure && <span aria-hidden>?</span>}
    </span>
  )
}

export function EffortBadge({ effort, unsure }: { effort: Effort; unsure?: boolean }) {
  return (
    <span
      title={effortHints[effort]}
      className={`inline-flex items-center gap-1 rounded-full border px-2.5 py-0.5 text-xs font-medium ${
        unsure ? 'border-dashed border-warn text-warn' : 'border-line text-ink-2'
      }`}
    >
      {effortLabels[effort]}
      {unsure && <span aria-hidden>?</span>}
    </span>
  )
}

/** Confirmed labels if present; otherwise the model's prediction, with each field
 *  marked "?" separately when its confidence is below the review threshold. */
export function TopicLabels({ topic }: { topic: Topic }) {
  const { labels, prediction: p } = topic
  if (labels.category && labels.effort) {
    return (
      <div className="flex flex-wrap items-center gap-1.5">
        <CategoryBadge category={labels.category} />
        <EffortBadge effort={labels.effort} />
        <span className="text-xs text-good">✓ {nl.prediction.confirmed}</span>
      </div>
    )
  }
  return (
    <div className="flex flex-wrap items-center gap-1.5">
      {p.category ? (
        <CategoryBadge
          category={p.category}
          unsure={(p.category_confidence ?? 0) < REVIEW_THRESHOLD}
        />
      ) : (
        <span className="text-xs text-ink-3">{nl.prediction.noPrediction}</span>
      )}
      {p.effort && (
        <EffortBadge effort={p.effort} unsure={(p.effort_confidence ?? 0) < REVIEW_THRESHOLD} />
      )}
    </div>
  )
}
