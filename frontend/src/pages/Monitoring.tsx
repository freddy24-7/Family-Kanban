import { useState } from 'react'
import {
  Area,
  Bar,
  BarChart,
  CartesianGrid,
  ComposedChart,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { useModelLog, useMonitoring, useSimulations, type ReferenceKind } from '../api/hooks'
import { CATEGORIES, type MonitoringReport, type MonitoringWeek } from '../api/types'
import { Card, Empty, ErrorText, Segmented, Spinner } from '../components/ui'
import { categoryLabels, nl } from '../i18n/nl'

const t = nl.monitoring
const MIN_WINDOW = 30 // keep in sync with ml/monitoring.py
const SERIES = { category: 'var(--cat-chores)', effort: 'var(--cat-groceries)' }
const axis = { fontSize: 11, fill: 'var(--ink-3)' }
const shortWeek = (iso: string) =>
  new Date(`${iso}T00:00:00`).toLocaleDateString('nl-NL', { day: 'numeric', month: 'short' })
const pct = (v: number | null | undefined) =>
  v === null || v === undefined ? '–' : `${Math.round(v * 100)}%`

export function MonitoringPage() {
  const sims = useSimulations()
  const [source, setSource] = useState('real')
  const [reference, setReference] = useState<ReferenceKind>('holdout')
  const [window, setWindow] = useState(4)
  const report = useMonitoring(source, reference, window)
  const completed = (sims.data ?? []).filter((r) => r.status === 'completed')

  return (
    <div className="space-y-5">
      <p className="text-sm text-ink-2">{t.intro}</p>
      <Card className="grid gap-3 sm:grid-cols-3">
        <label className="space-y-1 text-sm">
          <span className="text-ink-2">{t.source}</span>
          <select
            value={source}
            onChange={(e) => setSource(e.target.value)}
            className="min-h-10 w-full rounded-xl border border-line bg-surface px-2"
          >
            <option value="real">{t.realFamilies}</option>
            {completed.map((r) => (
              <option key={r.id} value={r.id}>
                {t.simulation}: {r.scenario.name} {r.pool_run_id ? '(replay)' : ''} · {r.start_date}
              </option>
            ))}
          </select>
        </label>
        <div className="space-y-1 text-sm">
          <span className="text-ink-2">{t.reference}</span>
          <Segmented
            label={t.reference}
            value={reference}
            onChange={setReference}
            options={[
              { value: 'holdout', label: t.refHoldout },
              { value: 'first_weeks', label: t.refFirstWeeks },
            ]}
          />
        </div>
        <label className="space-y-1 text-sm">
          <span className="text-ink-2">{t.window}</span>
          <select
            value={window}
            onChange={(e) => setWindow(Number(e.target.value))}
            className="min-h-10 w-full rounded-xl border border-line bg-surface px-2"
          >
            {[2, 4, 6, 8].map((w) => (
              <option key={w} value={w}>
                {w} {t.weeks}
              </option>
            ))}
          </select>
        </label>
      </Card>
      {report.isLoading ? (
        <Spinner />
      ) : report.error ? (
        <ErrorText error={report.error} />
      ) : (
        <Report report={report.data!} />
      )}
      <ModelLog />
    </div>
  )
}

function Report({ report }: { report: MonitoringReport }) {
  if (!report.reference || report.weekly.length === 0) return <Empty>{t.noData}</Empty>
  const weeks = report.weekly
  const last = weeks[weeks.length - 1]
  // Local date arithmetic only: toISOString() converts to UTC and would shift a Monday
  // to Sunday, so the event line would miss its week on the axis.
  const eventWeeks = (report.run?.scenario.events ?? []).map((e) => {
    const [y, m, d] = report.run!.start_date.split('-').map(Number)
    const day = new Date(y, m - 1, d + 7 * e.week)
    const iso = `${day.getFullYear()}-${String(day.getMonth() + 1).padStart(2, '0')}-${String(day.getDate()).padStart(2, '0')}`
    return { week: iso, label: e.kind }
  })
  return (
    <div className="space-y-5">
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <Stat label={t.tickets} value={String(report.n_tickets ?? 0)} />
        <Stat
          label={t.categoryAcc}
          value={pct(last.category_accuracy.value)}
          sub={ci(last.category_accuracy.ci95)}
        />
        <Stat
          label={t.effortAcc}
          value={pct(last.effort_accuracy.value)}
          sub={ci(last.effort_accuracy.ci95)}
        />
        <Stat
          label={t.activeAlarms}
          value={String(last.alarms.length)}
          sub={
            last.alarms.map((a) => t.alarmNames[a as keyof typeof t.alarmNames] ?? a).join(', ') ||
            t.none
          }
          warn={last.alarms.length > 0}
        />
      </div>
      <AccuracyChart weeks={weeks} reference={report.reference} events={eventWeeks} />
      <div className="grid gap-3 md:grid-cols-3">
        <PsiChart
          weeks={weeks}
          psiKey="psi_category"
          alarm="category_mix"
          title={t.psiCategory}
          events={eventWeeks}
        />
        <PsiChart
          weeks={weeks}
          psiKey="psi_confidence"
          alarm="confidence"
          title={t.psiConfidence}
          events={eventWeeks}
        />
        <PsiChart
          weeks={weeks}
          psiKey="psi_length"
          alarm="text_length"
          title={t.psiLength}
          events={eventWeeks}
        />
      </div>
      <MixChart weeks={weeks} />
      {report.detection && <Detection detection={report.detection} />}
    </div>
  )
}

const ci = (c: [number, number] | null) => (c ? `95%: ${pct(c[0])}–${pct(c[1])}` : '')

function Stat({
  label,
  value,
  sub,
  warn,
}: {
  label: string
  value: string
  sub?: string
  warn?: boolean
}) {
  return (
    <Card className="p-3">
      <div className="text-xs text-ink-3">{label}</div>
      <div className={`text-2xl font-semibold tabular-nums ${warn ? 'text-warn' : ''}`}>
        {value}
      </div>
      {sub && <div className="truncate text-xs text-ink-3">{sub}</div>}
    </Card>
  )
}

function ChartCard({
  title,
  legend,
  children,
}: {
  title: string
  legend?: React.ReactNode
  children: React.ReactNode
}) {
  return (
    <Card className="p-3">
      <div className="mb-2 flex flex-wrap items-baseline justify-between gap-2">
        <h3 className="text-sm font-semibold">{title}</h3>
        {legend}
      </div>
      {children}
    </Card>
  )
}

function Swatch({
  color,
  label,
  line = 'solid',
}: {
  color: string
  label: string
  line?: 'solid' | 'dashed' | 'dotted'
}) {
  return (
    <span className="inline-flex items-center gap-1.5 text-xs text-ink-2">
      <span
        className="inline-block h-0 w-4 border-t-2"
        style={{ borderColor: color, borderStyle: line }}
      />
      {label}
    </span>
  )
}

const tooltipStyle = {
  contentStyle: {
    background: 'var(--surface)',
    border: '1px solid var(--border)',
    borderRadius: 12,
    fontSize: 12,
  },
  labelStyle: { color: 'var(--ink)' },
}

/** Planted events as vertical lines. Labels only where there is room (the wide chart);
 *  the small charts show the same lines unlabelled, so text never collides. */
function EventLines({
  events,
  labels = false,
}: {
  events: { week: string; label: string }[]
  labels?: boolean
}) {
  return (
    <>
      {events.map((e) => (
        <ReferenceLine
          key={e.week}
          x={e.week}
          stroke="var(--ink-3)"
          strokeDasharray="2 3"
          label={
            labels
              ? { value: e.label, position: 'insideTopLeft', fontSize: 10, fill: 'var(--ink-3)' }
              : undefined
          }
        />
      ))}
    </>
  )
}

function AccuracyChart({
  weeks,
  reference,
  events,
}: {
  weeks: MonitoringWeek[]
  reference: NonNullable<MonitoringReport['reference']>
  events: { week: string; label: string }[]
}) {
  const hasTruth = weeks.some((w) => w.true_category_accuracy)
  const data = weeks.map((w) => ({
    week: w.week,
    category: w.category_accuracy.value,
    categoryBand: w.category_accuracy.ci95,
    effort: w.effort_accuracy.value,
    effortBand: w.effort_accuracy.ci95,
    trueCategory: w.true_category_accuracy?.value ?? null,
    trueEffort: w.true_effort_accuracy?.value ?? null,
  }))
  return (
    <ChartCard
      title={t.accuracyTitle}
      legend={
        <span className="flex flex-wrap gap-3">
          <Swatch color={SERIES.category} label={t.category} />
          <Swatch color={SERIES.effort} label={t.effort} />
          <Swatch color="var(--ink-3)" label={t.referenceLine} line="dashed" />
          {hasTruth && <Swatch color="var(--ink-2)" label={t.truth} line="dotted" />}
        </span>
      }
    >
      <p className="mb-2 text-xs text-ink-3">{t.accuracyHelp}</p>
      <ResponsiveContainer width="100%" height={240}>
        <ComposedChart data={data} margin={{ top: 8, right: 8, left: -18, bottom: 0 }}>
          <CartesianGrid stroke="var(--border)" vertical={false} />
          <XAxis
            dataKey="week"
            tickFormatter={shortWeek}
            tick={axis}
            tickLine={false}
            axisLine={{ stroke: 'var(--border)' }}
            minTickGap={24}
          />
          <YAxis
            domain={[0, 1]}
            tickFormatter={(v) => `${Math.round(v * 100)}%`}
            tick={axis}
            tickLine={false}
            axisLine={false}
          />
          <Tooltip
            {...tooltipStyle}
            labelFormatter={(l) => `${t.weekOf} ${shortWeek(String(l))}`}
            formatter={(v, name) => [
              Array.isArray(v) ? v.map((x) => pct(x as number)).join('–') : pct(v as number),
              name,
            ]}
          />
          <Area
            dataKey="categoryBand"
            name={`${t.category} 95%`}
            stroke="none"
            fill={SERIES.category}
            fillOpacity={0.12}
            isAnimationActive={false}
          />
          <Area
            dataKey="effortBand"
            name={`${t.effort} 95%`}
            stroke="none"
            fill={SERIES.effort}
            fillOpacity={0.12}
            isAnimationActive={false}
          />
          {reference.category_accuracy !== null && (
            <ReferenceLine
              y={reference.category_accuracy}
              stroke={SERIES.category}
              strokeDasharray="5 4"
              strokeOpacity={0.7}
            />
          )}
          {reference.effort_accuracy !== null && (
            <ReferenceLine
              y={reference.effort_accuracy}
              stroke={SERIES.effort}
              strokeDasharray="5 4"
              strokeOpacity={0.7}
            />
          )}
          <Line
            dataKey="category"
            name={t.category}
            stroke={SERIES.category}
            strokeWidth={2}
            dot={false}
            connectNulls
            isAnimationActive={false}
          />
          <Line
            dataKey="effort"
            name={t.effort}
            stroke={SERIES.effort}
            strokeWidth={2}
            dot={false}
            connectNulls
            isAnimationActive={false}
          />
          {hasTruth && (
            <Line
              dataKey="trueCategory"
              name={`${t.category} (${t.truth})`}
              stroke={SERIES.category}
              strokeWidth={1.5}
              strokeDasharray="1 3"
              dot={false}
              isAnimationActive={false}
            />
          )}
          {hasTruth && (
            <Line
              dataKey="trueEffort"
              name={`${t.effort} (${t.truth})`}
              stroke={SERIES.effort}
              strokeWidth={1.5}
              strokeDasharray="1 3"
              dot={false}
              isAnimationActive={false}
            />
          )}
          <EventLines events={events} labels />
        </ComposedChart>
      </ResponsiveContainer>
    </ChartCard>
  )
}

function PsiChart({
  weeks,
  psiKey,
  alarm,
  title,
  events,
}: {
  weeks: MonitoringWeek[]
  psiKey: 'psi_category' | 'psi_confidence' | 'psi_length'
  alarm: string
  title: string
  events: { week: string; label: string }[]
}) {
  // Windows below the minimum size are left out: PSI on a handful of tickets is noise.
  const data = weeks.map((w) => ({
    week: w.week,
    psi: w.n_window >= MIN_WINDOW ? w[psiKey] : null,
    alarm: w.alarms.includes(alarm),
  }))
  return (
    <ChartCard title={title}>
      <ResponsiveContainer width="100%" height={150}>
        <LineChart data={data} margin={{ top: 8, right: 8, left: -22, bottom: 0 }}>
          <CartesianGrid stroke="var(--border)" vertical={false} />
          <XAxis
            dataKey="week"
            tickFormatter={shortWeek}
            tick={axis}
            tickLine={false}
            axisLine={{ stroke: 'var(--border)' }}
            minTickGap={30}
          />
          <YAxis
            tick={axis}
            tickLine={false}
            axisLine={false}
            domain={[0, (max: number) => Math.max(0.4, Math.ceil(max * 10) / 10)]}
          />
          <Tooltip
            {...tooltipStyle}
            labelFormatter={(l) => `${t.weekOf} ${shortWeek(String(l))}`}
            formatter={(v, _n, item) => [
              `${(v as number)?.toFixed(2)}${item.payload.alarm ? ` · ${t.alarm}` : ''}`,
              'PSI',
            ]}
          />
          <ReferenceLine y={0.25} stroke="var(--warn)" strokeDasharray="5 4" />
          <ReferenceLine y={0.1} stroke="var(--border)" />
          <Line
            dataKey="psi"
            stroke="var(--accent)"
            strokeWidth={2}
            isAnimationActive={false}
            dot={(props: {
              cx?: number
              cy?: number
              payload?: { alarm: boolean }
              index?: number
            }) =>
              props.payload?.alarm ? (
                <circle
                  key={props.index}
                  cx={props.cx}
                  cy={props.cy}
                  r={4.5}
                  fill="var(--warn)"
                  stroke="var(--surface)"
                  strokeWidth={2}
                />
              ) : (
                <g key={props.index} />
              )
            }
          />
          <EventLines events={events} />
        </LineChart>
      </ResponsiveContainer>
    </ChartCard>
  )
}

function MixChart({ weeks }: { weeks: MonitoringWeek[] }) {
  const data = weeks.map((w) => ({ week: w.week, ...w.category_mix }))
  return (
    <ChartCard
      title={t.mixTitle}
      legend={
        <span className="flex flex-wrap gap-x-3 gap-y-1">
          {CATEGORIES.map((c) => (
            <span key={c} className="inline-flex items-center gap-1 text-xs text-ink-2">
              <span className="size-2.5 rounded-sm" style={{ background: `var(--cat-${c})` }} />
              {categoryLabels[c]}
            </span>
          ))}
        </span>
      }
    >
      <ResponsiveContainer width="100%" height={200}>
        <BarChart
          data={data}
          margin={{ top: 4, right: 8, left: -18, bottom: 0 }}
          barCategoryGap={2}
        >
          <XAxis
            dataKey="week"
            tickFormatter={shortWeek}
            tick={axis}
            tickLine={false}
            axisLine={{ stroke: 'var(--border)' }}
            minTickGap={24}
          />
          <YAxis
            tickFormatter={(v) => `${Math.round(v * 100)}%`}
            tick={axis}
            tickLine={false}
            axisLine={false}
            domain={[0, 1]}
          />
          <Tooltip
            {...tooltipStyle}
            labelFormatter={(l) => `${t.weekOf} ${shortWeek(String(l))}`}
            formatter={(v, name) => [
              pct(v as number),
              categoryLabels[name as keyof typeof categoryLabels] ?? name,
            ]}
          />
          {CATEGORIES.map((c) => (
            <Bar
              key={c}
              dataKey={c}
              stackId="mix"
              fill={`var(--cat-${c})`}
              stroke="var(--surface)"
              strokeWidth={1}
              isAnimationActive={false}
            />
          ))}
        </BarChart>
      </ResponsiveContainer>
    </ChartCard>
  )
}

function Detection({ detection }: { detection: NonNullable<MonitoringReport['detection']> }) {
  return (
    <ChartCard title={t.detectionTitle}>
      <p className="mb-2 text-xs text-ink-3">{t.detectionHelp}</p>
      <div className="overflow-x-auto">
        <table className="w-full text-left text-sm">
          <thead className="text-xs text-ink-3">
            <tr>
              <th className="py-1 pr-3 font-medium">{t.event}</th>
              <th className="py-1 pr-3 font-medium">{t.delay}</th>
              <th className="py-1 font-medium">{t.detectors}</th>
            </tr>
          </thead>
          <tbody>
            {detection.detections.map((d) => (
              <tr key={d.event} className="border-t border-line align-top">
                <td className="py-2 pr-3">
                  <div className="font-medium">{d.event}</div>
                  <div className="text-xs text-ink-3">{d.note}</div>
                </td>
                <td className="py-2 pr-3 tabular-nums">
                  {d.delay_weeks === null ? (
                    <span className="text-warn">{t.missed}</span>
                  ) : (
                    `${d.delay_weeks} ${t.weeks}`
                  )}
                </td>
                <td className="py-2 text-xs text-ink-2">{d.detectors.join(', ') || '–'}</td>
              </tr>
            ))}
            <tr className="border-t border-line">
              <td className="py-2 pr-3 font-medium">{t.falseAlarms}</td>
              <td className="py-2 pr-3 tabular-nums">{detection.false_alarms.length}</td>
              <td className="py-2 text-xs text-ink-2">
                {detection.false_alarms
                  .map((f) => `${shortWeek(f.week)}: ${f.alarms.join(', ')}`)
                  .join(' · ') || '–'}
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </ChartCard>
  )
}

function ModelLog() {
  const log = useModelLog()
  if (!log.data) return null
  return (
    <ChartCard title={t.modelLog}>
      <div className="overflow-x-auto">
        <table className="w-full text-left text-sm">
          <thead className="text-xs text-ink-3">
            <tr>
              <th className="py-1 pr-3 font-medium">{t.model}</th>
              <th className="py-1 pr-3 font-medium">{t.status}</th>
              <th className="py-1 pr-3 text-right font-medium">{t.trainingSize}</th>
              <th className="py-1 text-right font-medium">{t.holdoutF1}</th>
            </tr>
          </thead>
          <tbody>
            {log.data.map((m) => (
              <tr key={m.name} className="border-t border-line">
                <td className="py-2 pr-3">
                  <div className="font-medium">{m.name}</div>
                  <div className="text-xs text-ink-3">
                    {new Date(m.created_at).toLocaleDateString('nl-NL')}
                  </div>
                </td>
                <td className={`py-2 pr-3 ${m.status === 'active' ? 'text-good' : 'text-ink-3'}`}>
                  {m.status}
                </td>
                <td className="py-2 pr-3 text-right tabular-nums">{m.training_set_size || '–'}</td>
                <td className="py-2 text-right tabular-nums">
                  {m.holdout_macro_f1 === null ? '–' : m.holdout_macro_f1.toFixed(3)}
                  {m.holdout_macro_f1_ci95 && (
                    <div className="text-xs text-ink-3">
                      {m.holdout_macro_f1_ci95.map((x) => x.toFixed(3)).join('–')}
                    </div>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </ChartCard>
  )
}
