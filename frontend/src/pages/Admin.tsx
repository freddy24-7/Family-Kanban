import { useState } from 'react'
import {
  useReplaySimulation,
  useResumeSimulation,
  useSimulation,
  useSimulations,
  useStartSimulation,
} from '../api/hooks'
import type { WeekStats } from '../api/types'
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
} from '../components/ui'
import { nl } from '../i18n/nl'
import { MonitoringPage } from './Monitoring'

const pct = (v: number | null) => (v === null ? '–' : `${Math.round(v * 100)}%`)

export function AdminPage() {
  const [tab, setTab] = useState<'monitoring' | 'simulations'>('monitoring')
  return (
    <div className="space-y-5">
      <PageTitle>{nl.nav.admin}</PageTitle>
      <Segmented
        label={nl.nav.admin}
        value={tab}
        onChange={setTab}
        options={[
          { value: 'monitoring', label: nl.monitoring.tab },
          { value: 'simulations', label: nl.admin.title },
        ]}
      />
      {tab === 'monitoring' ? <MonitoringPage /> : <SimulationsPage />}
    </div>
  )
}

function SimulationsPage() {
  const runs = useSimulations()
  const start = useStartSimulation()
  const [selected, setSelected] = useState<string | null>(null)
  const [scenario, setScenario] = useState('drift-demo')
  const [weeks, setWeeks] = useState(26)
  const [startDate, setStartDate] = useState('2026-07-06')
  const [rubber, setRubber] = useState(0.2)
  const current = selected ?? runs.data?.[0]?.id ?? null

  return (
    <div className="space-y-6">
      <p className="text-sm text-ink-2">{nl.admin.intro}</p>
      <Card className="space-y-3">
        <div className="grid grid-cols-2 gap-3">
          <Field label={nl.admin.scenario}>
            <select
              value={scenario}
              onChange={(e) => setScenario(e.target.value)}
              className="min-h-11 w-full rounded-xl border border-line bg-surface px-2"
            >
              <option value="drift-demo">drift-demo</option>
              <option value="baseline">baseline</option>
            </select>
          </Field>
          <Field label={nl.admin.weeks}>
            <Input
              type="number"
              min={1}
              max={104}
              value={weeks}
              onChange={(e) => setWeeks(Number(e.target.value))}
            />
          </Field>
          <Field label={nl.admin.start}>
            <Input type="date" value={startDate} onChange={(e) => setStartDate(e.target.value)} />
          </Field>
          <Field label={nl.admin.rubberStamp} hint={nl.admin.rubberStampHelp}>
            <Input
              type="number"
              min={0}
              max={1}
              step={0.1}
              value={rubber}
              onChange={(e) => setRubber(Number(e.target.value))}
            />
          </Field>
        </div>
        <ErrorText error={start.error} />
        <Button
          className="w-full"
          disabled={start.isPending}
          onClick={async () => {
            const run = await start.mutateAsync({
              scenario,
              weeks,
              start_date: startDate,
              planner: { rubber_stamp_rate: rubber },
            })
            setSelected(run.id)
          }}
        >
          {nl.admin.run}
        </Button>
      </Card>

      <section>
        <h2 className="mb-2 font-semibold">{nl.admin.runs}</h2>
        {runs.isLoading ? (
          <Spinner />
        ) : (runs.data ?? []).length === 0 ? (
          <Empty>–</Empty>
        ) : (
          <ul className="space-y-2">
            {runs.data!.map((r) => (
              <li key={r.id}>
                <button
                  onClick={() => setSelected(r.id)}
                  className={`w-full rounded-2xl border p-3 text-left text-sm ${r.id === current ? 'border-accent' : 'border-line'} bg-surface`}
                >
                  <div className="flex justify-between gap-2">
                    <span className="font-medium">
                      {r.scenario.name} {r.pool_run_id ? `(${nl.admin.replay.toLowerCase()})` : ''}
                    </span>
                    <span className="text-ink-3">
                      {nl.admin.status[r.status]} · {nl.admin.progress(r.current_week, r.weeks)}
                    </span>
                  </div>
                  <div className="text-xs text-ink-3">
                    {r.start_date} · rubber-stamp {r.planner.rubber_stamp_rate} · seed{' '}
                    {r.random_seed}
                  </div>
                </button>
              </li>
            ))}
          </ul>
        )}
      </section>

      {current && <RunDetail id={current} onReplay={setSelected} />}
    </div>
  )
}

function RunDetail({ id, onReplay }: { id: string; onReplay: (id: string) => void }) {
  const run = useSimulation(id)
  const replay = useReplaySimulation()
  const resume = useResumeSimulation()
  const [rubber, setRubber] = useState(0.6)
  if (run.isLoading || !run.data) return <Spinner />
  const r = run.data
  return (
    <section className="space-y-3">
      <h2 className="font-semibold">{nl.admin.weekly}</h2>
      {r.error && <ErrorText error={new Error(r.error)} />}
      <p className="text-xs text-ink-3">{nl.admin.legend}</p>
      <div className="overflow-x-auto rounded-2xl border border-line bg-surface">
        <table className="w-full text-right text-xs tabular-nums">
          <thead className="text-ink-3">
            <tr>
              {['week', 'n', 'cat', 'cat*', 'moeite', 'zeker', 'gemarkeerd', 'woorden', ''].map(
                (h) => (
                  <th key={h} className="px-2 py-2 font-medium">
                    {h}
                  </th>
                ),
              )}
            </tr>
          </thead>
          <tbody>
            {r.weekly_stats.map((s: WeekStats) => (
              <tr
                key={s.week}
                className={`border-t border-line ${s.events.length ? 'bg-warn-bg' : ''}`}
              >
                <td className="px-2 py-1.5">{s.week + 1}</td>
                <td className="px-2">{s.n}</td>
                <td className="px-2">{pct(s.true_category_acc)}</td>
                <td className="px-2">{pct(s.measured_category_acc)}</td>
                <td className="px-2">{pct(s.true_effort_acc)}</td>
                <td className="px-2">{pct(s.mean_category_conf)}</td>
                <td className="px-2">{pct(s.flagged_rate)}</td>
                <td className="px-2">{s.mean_words?.toFixed(1) ?? '–'}</td>
                <td className="px-2 text-left text-warn">
                  {s.events.map((e) => e.split(':')[0]).join(', ')}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {r.status === 'failed' && !r.pool_run_id && r.current_week === 0 && (
        <Card className="flex flex-wrap items-center justify-between gap-3">
          <span className="text-sm text-ink-2">{nl.admin.resumeHelp}</span>
          <Button
            variant="secondary"
            disabled={resume.isPending}
            onClick={() => resume.mutate(r.id)}
          >
            {nl.admin.resume}
          </Button>
        </Card>
      )}
      {(r.status === 'completed' || (r.status === 'failed' && r.current_week > 0)) && (
        <Card className="flex flex-wrap items-end gap-3">
          <Field label={nl.admin.rubberStamp}>
            <Input
              type="number"
              min={0}
              max={1}
              step={0.1}
              value={rubber}
              onChange={(e) => setRubber(Number(e.target.value))}
            />
          </Field>
          <Button
            variant="secondary"
            disabled={replay.isPending}
            onClick={async () =>
              onReplay((await replay.mutateAsync({ id: r.id, rubber_stamp_rate: rubber })).id)
            }
          >
            {nl.admin.replay}
          </Button>
        </Card>
      )}
    </section>
  )
}
