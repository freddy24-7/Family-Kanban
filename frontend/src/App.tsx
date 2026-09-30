import type { ReactNode } from 'react'
import { Navigate, Route, Routes, useLocation } from 'react-router-dom'
import { Layout } from './components/Layout'
import { Spinner } from './components/ui'
import {
  ForgotPasswordPage,
  LoginPage,
  RegisterPage,
  ResetPasswordPage,
  VerifyEmailPage,
} from './pages/Auth'
import { AdminPage } from './pages/Admin'
import { BacklogPage } from './pages/Backlog'
import { BoardPage } from './pages/Board'
import { FamilyPage } from './pages/Family'
import { InvitePage } from './pages/Invite'
import { NewTopicPage } from './pages/NewTopic'
import { OnboardingPage } from './pages/Onboarding'
import { PlanPage } from './pages/Plan'
import { ReviewPage } from './pages/Review'
import { useAuth } from './state/auth'
import { HouseholdProvider } from './state/household'

function RequireAuth({ children }: { children: ReactNode }) {
  const { user, loading } = useAuth()
  const location = useLocation()
  if (loading) return <Spinner />
  if (!user) return <Navigate to={`/login?next=${encodeURIComponent(location.pathname)}`} replace />
  return <>{children}</>
}

function AdminOnly() {
  const { user } = useAuth()
  return user?.is_superuser ? <AdminPage /> : <Navigate to="/" replace />
}

export default function App() {
  return (
    <Routes>
      {/* Paths in emails: /verify-email, /reset-password, /invite (see backend messages). */}
      <Route path="/login" element={<LoginPage />} />
      <Route path="/register" element={<RegisterPage />} />
      <Route path="/forgot-password" element={<ForgotPasswordPage />} />
      <Route path="/reset-password" element={<ResetPasswordPage />} />
      <Route path="/verify-email" element={<VerifyEmailPage />} />
      <Route path="/invite" element={<InvitePage />} />
      <Route
        element={
          <RequireAuth>
            <HouseholdProvider fallback={<Spinner />} empty={<OnboardingPage />}>
              <Layout />
            </HouseholdProvider>
          </RequireAuth>
        }
      >
        <Route index element={<Navigate to="/board" replace />} />
        <Route path="/new" element={<NewTopicPage />} />
        <Route path="/backlog" element={<BacklogPage />} />
        <Route path="/plan" element={<PlanPage />} />
        <Route path="/board" element={<BoardPage />} />
        <Route path="/review" element={<ReviewPage />} />
        <Route path="/family" element={<FamilyPage />} />
        <Route path="/admin" element={<AdminOnly />} />
      </Route>
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  )
}
