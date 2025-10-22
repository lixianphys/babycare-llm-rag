'use client'
import { useState, useEffect } from 'react'
import { LogIn } from 'lucide-react'
import LoginForm from './LoginForm'

export default function LoginButton() {
  const [showLoginForm, setShowLoginForm] = useState(false)

  useEffect(() => {
    const handleOpenLogin = () => setShowLoginForm(true)
    window.addEventListener('openLogin', handleOpenLogin)
    return () => window.removeEventListener('openLogin', handleOpenLogin)
  }, [])

  return (
    <>
      <button
        onClick={() => setShowLoginForm(true)}
        className="flex items-center space-x-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
      >
        <LogIn className="h-4 w-4" />
        <span>Sign In</span>
      </button>

      {showLoginForm && (
        <LoginForm onClose={() => setShowLoginForm(false)} />
      )}
    </>
  )
}
