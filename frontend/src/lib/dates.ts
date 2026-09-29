const pad = (n: number) => String(n).padStart(2, '0')
export const isoDate = (d: Date) =>
  `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`

/** Next week's Monday..Sunday, or this week's if today is Monday. */
export function nextWeek(today = new Date()): { start: string; end: string; week: number } {
  const start = new Date(today)
  const offset = (8 - start.getDay()) % 7
  start.setDate(start.getDate() + offset)
  const end = new Date(start)
  end.setDate(start.getDate() + 6)
  return { start: isoDate(start), end: isoDate(end), week: isoWeek(start) }
}

export function isoWeek(date: Date): number {
  const d = new Date(Date.UTC(date.getFullYear(), date.getMonth(), date.getDate()))
  const day = d.getUTCDay() || 7
  d.setUTCDate(d.getUTCDate() + 4 - day)
  const yearStart = new Date(Date.UTC(d.getUTCFullYear(), 0, 1))
  return Math.ceil(((d.getTime() - yearStart.getTime()) / 86400000 + 1) / 7)
}

export const shortDate = (iso: string) =>
  new Date(`${iso}T00:00:00`).toLocaleDateString('nl-NL', { day: 'numeric', month: 'short' })
