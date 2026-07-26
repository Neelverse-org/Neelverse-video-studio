import { useEffect, useState } from 'react'
import { LoaderCircle } from 'lucide-react'
import { api, ApiError } from './api'
import { Login } from './Login'
import { Studio } from './Studio'
import type { User } from './types'

export function App() {
  const [user, setUser] = useState<User | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    api
      .me()
      .then(setUser)
      .catch((error: unknown) => {
        if (!(error instanceof ApiError && error.status === 401)) console.error(error)
      })
      .finally(() => setLoading(false))
  }, [])

  async function logout() {
    await api.logout()
    setUser(null)
  }

  if (loading) {
    return <div className="boot-screen"><LoaderCircle className="spin" size={24} /><span>Opening Neelverse</span></div>
  }
  return user ? <Studio user={user} onLogout={() => void logout()} /> : <Login onLogin={setUser} />
}
