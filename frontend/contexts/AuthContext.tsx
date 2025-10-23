'use client'
import React, { createContext, useContext, useState, useEffect } from 'react'
import axios from 'axios'
import Cookies from 'js-cookie'

interface User {
  id: number
  username: string
  email: string
  name: string
  created_at: string
  last_login: string
}

interface LoginData {
  username: string
  password: string
}

interface RegisterData {
  username: string
  email: string
  password: string
  name: string
}

interface AuthContextType {
  user: User | null
  isLoading: boolean
  login: (data: LoginData) => Promise<void>
  register: (data: RegisterData) => Promise<void>
  logout: () => void
  isAuthenticated: boolean
  isGuest: boolean
}

const AuthContext = createContext<AuthContextType | undefined>(undefined)

export const useAuth = () => {
  const context = useContext(AuthContext)
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider')
  }
  return context
}

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null)
  const [isLoading, setIsLoading] = useState(true)

  const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000'

  useEffect(() => {
    const token = Cookies.get('auth_token')
    if (token) {
      // Set token in axios headers
      axios.defaults.headers.common['Authorization'] = `Bearer ${token}`
      
      // Verify token and get user info
      axios.get(`${API_BASE_URL}/auth/me`)
        .then(response => {
          setUser(response.data)
        })
        .catch(() => {
          // Token is invalid, remove it
          Cookies.remove('auth_token')
          delete axios.defaults.headers.common['Authorization']
        })
        .finally(() => {
          setIsLoading(false)
        })
    } else {
      setIsLoading(false)
    }
  }, [])

  const login = async (data: LoginData) => {
    try {
      const response = await axios.post(`${API_BASE_URL}/auth/login`, data)
      const { access_token } = response.data
      
      // Store token
      Cookies.set('auth_token', access_token, { expires: 7 })
      axios.defaults.headers.common['Authorization'] = `Bearer ${access_token}`
      
      // Get user info
      const userResponse = await axios.get(`${API_BASE_URL}/auth/me`)
      setUser(userResponse.data)
    } catch (error) {
      console.error('Login failed:', error)
      throw error
    }
  }

  const register = async (data: RegisterData) => {
    try {
      const response = await axios.post(`${API_BASE_URL}/auth/register`, data)
      
      // Auto-login after registration
      const loginResponse = await axios.post(`${API_BASE_URL}/auth/login`, {
        username: data.username,
        password: data.password
      })
      
      const { access_token } = loginResponse.data
      
      // Store token
      Cookies.set('auth_token', access_token, { expires: 7 })
      axios.defaults.headers.common['Authorization'] = `Bearer ${access_token}`
      
      // Get user info
      const userResponse = await axios.get(`${API_BASE_URL}/auth/me`)
      setUser(userResponse.data)
    } catch (error) {
      console.error('Registration failed:', error)
      throw error
    }
  }

  const logout = () => {
    setUser(null)
    Cookies.remove('auth_token')
    delete axios.defaults.headers.common['Authorization']
  }

  const isAuthenticated = !!user
  const isGuest = !isAuthenticated && !isLoading

  const value = {
    user,
    isLoading,
    login,
    register,
    logout,
    isAuthenticated,
    isGuest
  }

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  )
}
