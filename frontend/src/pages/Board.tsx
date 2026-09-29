import {
  DndContext,
  PointerSensor,
  TouchSensor,
  useDraggable,
  useDroppable,
  useSensor,
  useSensors,
  type DragEndEvent,
} from '@dnd-kit/core'
import { useState } from 'react'
import { useMoveItem, useSprint, useSprints } from '../api/hooks'
import type { ItemStatus, SprintItem } from '../api/types'
import { TopicLabels } from '../components/badges'
import { Empty, ErrorText, PageTitle, Segmented, Spinner } from '../components/ui'
import { nl } from '../i18n/nl'
import { useCurrentHousehold } from '../state/household'

const COLUMNS: ItemStatus[] = ['todo', 'in_progress', 'done']

export function BoardPage() {
  const { household } = useCurrentHousehold()
  const sprints = useSprints(household.id)
  const active = sprints.data?.find((s) => s.status === 'active')
  if (sprints.isLoading) return <Spinner />
  if (!active) return <Empty>{nl.board.noActive}</Empty>
  return <Board sprintId={active.id} />
}

function Board({ sprintId }: { sprintId: string }) {
  const { household, me, isPlanner, members } = useCurrentHousehold()
  const sprint = useSprint(household.id, sprintId)
  const move = useMoveItem(household.id, sprintId)
  const [who, setWho] = useState<string>('all')
  // Touch: a short press on the handle starts a drag, so scrolling still works.
  const sensors = useSensors(
    useSensor(PointerSensor, { activationConstraint: { distance: 4 } }),
    useSensor(TouchSensor, { activationConstraint: { delay: 150, tolerance: 8 } }),
  )

  if (sprint.isLoading) return <Spinner />
  const s = sprint.data
  if (!s) return <ErrorText error={sprint.error} />

  const canMove = (item: SprintItem) => isPlanner || item.assignee_id === me?.user_id
  const items = s.items.filter(
    (i) => who === 'all' || i.assignee_id === (who === 'me' ? me?.user_id : who),
  )

  function onDragEnd(event: DragEndEvent) {
    const item = s!.items.find((i) => i.id === event.active.id)
    const target = event.over?.id as ItemStatus | undefined
    if (item && target && target !== item.status && canMove(item))
      move.mutate({ itemId: item.id, status: target })
  }

  return (
    <div>
      <PageTitle>{s.name}</PageTitle>
      <div className="mb-4">
        <Segmented
          label={nl.board.title}
          value={who}
          onChange={setWho}
          options={[
            { value: 'all', label: nl.common.everyone },
            { value: 'me', label: nl.board.mine },
            ...members
              .filter((m) => m.user_id !== me?.user_id)
              .map((m) => ({ value: m.user_id, label: m.display_name })),
          ]}
        />
      </div>
      <ErrorText error={move.error} />
      <DndContext sensors={sensors} onDragEnd={onDragEnd}>
        <div className="grid gap-3 md:grid-cols-3">
          {COLUMNS.map((status) => (
            <Column
              key={status}
              status={status}
              items={items.filter((i) => i.status === status)}
              canMove={canMove}
              onMove={(itemId, to) => move.mutate({ itemId, status: to })}
            />
          ))}
        </div>
      </DndContext>
    </div>
  )
}

function Column({
  status,
  items,
  canMove,
  onMove,
}: {
  status: ItemStatus
  items: SprintItem[]
  canMove: (i: SprintItem) => boolean
  onMove: (itemId: string, to: ItemStatus) => void
}) {
  const { setNodeRef, isOver } = useDroppable({ id: status })
  return (
    <section
      ref={setNodeRef}
      className={`min-h-32 rounded-2xl border p-3 transition ${isOver ? 'border-accent bg-accent/5' : 'border-line bg-surface-2/60'}`}
    >
      <h2 className="mb-2 flex items-center justify-between text-sm font-semibold text-ink-2">
        {nl.board.columns[status]}
        <span className="text-ink-3">{items.length}</span>
      </h2>
      <ul className="space-y-2">
        {items.map((item) => (
          <BoardCard key={item.id} item={item} movable={canMove(item)} onMove={onMove} />
        ))}
      </ul>
    </section>
  )
}

function BoardCard({
  item,
  movable,
  onMove,
}: {
  item: SprintItem
  movable: boolean
  onMove: (itemId: string, to: ItemStatus) => void
}) {
  const { attributes, listeners, setNodeRef, transform, isDragging } = useDraggable({
    id: item.id,
    disabled: !movable,
  })
  const index = COLUMNS.indexOf(item.status)
  const style = transform
    ? { transform: `translate3d(${transform.x}px, ${transform.y}px, 0)` }
    : undefined
  return (
    <li
      ref={setNodeRef}
      style={style}
      className={`rounded-xl border border-line bg-surface p-3 ${isDragging ? 'relative z-20 shadow-lg' : ''}`}
    >
      <div className="mb-2 flex items-start justify-between gap-2">
        <span>{item.topic.text}</span>
        {movable && (
          // A separate drag handle: the card stays a plain list item, so the move
          // buttons below aren't nested inside another "button" (a11y).
          <button
            type="button"
            aria-label={`${nl.board.drag}: ${item.topic.text}`}
            className="-m-1 cursor-grab touch-none rounded-lg p-1 text-ink-3 hover:bg-surface-2 active:cursor-grabbing"
            {...attributes}
            {...listeners}
          >
            <svg viewBox="0 0 20 20" className="size-5" fill="currentColor" aria-hidden>
              <circle cx="7" cy="5" r="1.5" />
              <circle cx="13" cy="5" r="1.5" />
              <circle cx="7" cy="10" r="1.5" />
              <circle cx="13" cy="10" r="1.5" />
              <circle cx="7" cy="15" r="1.5" />
              <circle cx="13" cy="15" r="1.5" />
            </svg>
          </button>
        )}
      </div>
      <TopicLabels topic={item.topic} />
      <div className="mt-2 flex items-center justify-between gap-2 text-xs text-ink-3">
        <span>{item.assignee_name ?? nl.common.nobody}</span>
        {movable && (
          <span className="flex gap-1">
            {index > 0 && (
              <button
                type="button"
                className="rounded-lg px-2 py-1 hover:bg-surface-2"
                onClick={() => onMove(item.id, COLUMNS[index - 1])}
              >
                ← {nl.board.moveLeft}
              </button>
            )}
            {index < COLUMNS.length - 1 && (
              <button
                type="button"
                className="rounded-lg px-2 py-1 text-accent hover:bg-surface-2"
                onClick={() => onMove(item.id, COLUMNS[index + 1])}
              >
                {nl.board.moveRight} →
              </button>
            )}
          </span>
        )}
      </div>
    </li>
  )
}
