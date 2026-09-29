import { useQueryClient } from '@tanstack/react-query'
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from 'react'
import { api, setUnauthorizedHandler, tokenStore } from '../api/client'
import { useMe } from '../api/hooks'
import type { User } from '../api/types'

interface AuthState {
  user: User | undefined
  loading: boolean
  login: (email: string, password: string) => Promise<void>
  register: (name: string, email: string, password: string) => Promise<void>
  logout: () => Promise<void>
}

const AuthContext = createContext<AuthState | null>(null)

export function AuthProvider({ children }: { children: ReactNode }) {
  const qc = useQueryClient()
  const [token, setToken] = useState(tokenStore.get)
  const me = useMe(!!token)

  const clear = useCallback(() => {
    tokenStore.set(null)
    setToken(null)
    qc.clear()
  }, [qc])

  useEffect(() => setUnauthorizedHandler(clear), [clear])

  const login = useCallback(async (email: string, password: string) => {
    const result = await api<{ access_token: string }>('/auth/login', {
      method: 'POST',
      form: { username: email, password },
    })
    tokenStore.set(result.access_token)
    setToken(result.access_token)
  }, [])

  const register = useCallback(
    async (name: string, email: string, password: string) => {
      await api('/auth/register', { method: 'POST', body: { email, password, display_name: name } })
      await login(email, password)
    },
    [login],
  )

  const logout = useCallback(async () => {
    try {
      await api('/auth/logout', { method: 'POST' }) // revokes the token server-side
    } finally {
      clear()
    }
  }, [clear])

  const value = useMemo(
    () => ({
      user: token ? me.data : undefined,
      loading: !!token && me.isLoading,
      login,
      register,
      logout,
    }),
    [token, me.data, me.isLoading, login, register, logout],
  )
  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

export function useAuth(): AuthState {
  const ctx = useContext(AuthContext)
  if (!ctx) throw new Error('useAuth outside AuthProvider')
  return ctx
}
