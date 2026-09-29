import { NavLink, Outlet } from 'react-router-dom'
import { nl } from '../i18n/nl'
import { useAuth } from '../state/auth'
import { useCurrentHousehold } from '../state/household'

const icons: Record<string, string> = {
  new: 'M12 5v14M5 12h14',
  backlog: 'M5 6h14M5 12h14M5 18h9',
  plan: 'M7 3v4M17 3v4M4 9h16M5 5h14v15H5z',
  board: 'M4 4h4v16H4zM10 4h4v10h-4zM16 4h4v6h-4z',
  review: 'M5 12l4 4L19 6',
  family:
    'M9 11a3 3 0 1 0 0-6 3 3 0 0 0 0 6zM3 20c0-3 3-5 6-5s6 2 6 5M16 5a3 3 0 0 1 0 6M18 15c2 .5 3 2 3 5',
}

const items = [
  { to: '/new', key: 'new', label: nl.nav.new },
  { to: '/backlog', key: 'backlog', label: nl.nav.backlog },
  { to: '/plan', key: 'plan', label: nl.nav.plan },
  { to: '/board', key: 'board', label: nl.nav.board },
  { to: '/review', key: 'review', label: nl.nav.review },
  { to: '/family', key: 'family', label: nl.nav.family },
]

function Icon({ path }: { path: string }) {
  return (
    <svg
      viewBox="0 0 24 24"
      className="size-5"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden
    >
      <path d={path} />
    </svg>
  )
}

export function Layout() {
  const { logout } = useAuth()
  const { household } = useCurrentHousehold()
  return (
    <div className="min-h-dvh pb-[calc(4.5rem+env(safe-area-inset-bottom))] md:pb-0">
      <header className="sticky top-0 z-10 border-b border-line bg-page/90 backdrop-blur">
        <div className="mx-auto flex max-w-4xl items-center gap-3 px-4 py-3">
          <img src="/favicon.svg" alt="" className="size-7" />
          <div className="min-w-0 flex-1">
            <div className="truncate font-semibold leading-tight">{nl.appName}</div>
            <div className="truncate text-xs text-ink-3">{household.name}</div>
          </div>
          <nav className="hidden gap-1 md:flex">
            {items.map((i) => (
              <NavLink
                key={i.to}
                to={i.to}
                className={({ isActive }) =>
                  `rounded-lg px-3 py-2 text-sm ${isActive ? 'bg-surface-2 font-medium text-ink' : 'text-ink-2 hover:bg-surface-2'}`
                }
              >
                {i.label}
              </NavLink>
            ))}
          </nav>
          <button
            onClick={logout}
            className="rounded-lg px-2 py-2 text-sm text-ink-3 hover:bg-surface-2"
          >
            {nl.auth.logout}
          </button>
        </div>
      </header>
      <main className="mx-auto max-w-4xl px-4 py-5">
        <Outlet />
      </main>
      <nav className="fixed inset-x-0 bottom-0 z-10 border-t border-line bg-surface pb-[env(safe-area-inset-bottom)] md:hidden">
        <div className="grid grid-cols-6">
          {items.map((i) => (
            <NavLink
              key={i.to}
              to={i.to}
              className={({ isActive }) =>
                `flex flex-col items-center gap-0.5 py-2 text-[11px] ${isActive ? 'text-accent' : 'text-ink-3'}`
              }
            >
              <Icon path={icons[i.key]} />
              {i.label}
            </NavLink>
          ))}
        </div>
      </nav>
    </div>
  )
}
