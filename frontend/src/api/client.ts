// Minimal fetch client: bearer token from localStorage, JSON in/out, typed errors.
const API_URL = (import.meta.env.VITE_API_URL ?? 'http://localhost:8000').replace(/\/$/, '')
const TOKEN_KEY = 'gezinsbord.token'

export class ApiError extends Error {
  status: number
  detail: unknown
  constructor(status: number, detail: unknown) {
    super(typeof detail === 'string' ? detail : `HTTP ${status}`)
    this.status = status
    this.detail = detail
  }
}

function readToken(): string | null {
  try {
    return localStorage.getItem(TOKEN_KEY)
  } catch {
    return null
  }
}

export const tokenStore = {
  get: readToken,
  set(token: string | null) {
    try {
      if (token) localStorage.setItem(TOKEN_KEY, token)
      else localStorage.removeItem(TOKEN_KEY)
    } catch {
      /* private mode: the session just won't persist */
    }
  },
}

let onUnauthorized: () => void = () => {}
export function setUnauthorizedHandler(handler: () => void) {
  onUnauthorized = handler
}

type Options = { method?: string; body?: unknown; form?: Record<string, string> }

export async function api<T>(
  path: string,
  { method = 'GET', body, form }: Options = {},
): Promise<T> {
  const headers: Record<string, string> = { Accept: 'application/json' }
  const token = readToken()
  if (token) headers.Authorization = `Bearer ${token}`
  let payload: BodyInit | undefined
  if (form) {
    headers['Content-Type'] = 'application/x-www-form-urlencoded'
    payload = new URLSearchParams(form)
  } else if (body !== undefined) {
    headers['Content-Type'] = 'application/json'
    payload = JSON.stringify(body)
  }
  const response = await fetch(`${API_URL}${path}`, { method, headers, body: payload })
  if (response.status === 401 && token) onUnauthorized()
  if (!response.ok) {
    let detail: unknown = null
    try {
      detail = (await response.json()).detail
    } catch {
      /* empty or non-JSON error body */
    }
    throw new ApiError(response.status, detail)
  }
  if (response.status === 204) return undefined as T
  return (await response.json()) as T
}
