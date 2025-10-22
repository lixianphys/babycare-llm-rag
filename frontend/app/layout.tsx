import type { Metadata, Viewport } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'
import { AuthProvider } from '../contexts/AuthContext'

const inter = Inter({ 
  subsets: ['latin'],
  display: 'swap',
  variable: '--font-inter',
})

export const metadata: Metadata = {
  title: {
    default: 'Baby Care Assistant',
    template: '%s | Baby Care Assistant'
  },
  description: 'AI-powered baby care assistant with medical knowledge from MotherToBaby.org and MedicinesInPregnancy.org',
  keywords: ['baby care', 'pregnancy', 'AI assistant', 'medical advice', 'child development'],
  authors: [{ name: 'Baby Care Assistant Team' }],
  creator: 'Baby Care Assistant',
  publisher: 'Baby Care Assistant',
  formatDetection: {
    email: false,
    address: false,
    telephone: false,
  },
  metadataBase: new URL(process.env.NEXT_PUBLIC_APP_URL || 'http://localhost:3000'),
  alternates: {
    canonical: '/',
  },
  openGraph: {
    type: 'website',
    locale: 'en_US',
    url: '/',
    title: 'Baby Care Assistant',
    description: 'AI-powered baby care assistant with medical knowledge',
    siteName: 'Baby Care Assistant',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'Baby Care Assistant',
    description: 'AI-powered baby care assistant with medical knowledge',
    creator: '@babycareassistant',
  },
  robots: {
    index: true,
    follow: true,
    googleBot: {
      index: true,
      follow: true,
      'max-video-preview': -1,
      'max-image-preview': 'large',
      'max-snippet': -1,
    },
  },
  verification: {
    google: process.env.GOOGLE_SITE_VERIFICATION,
  },
}

export const viewport: Viewport = {
  width: 'device-width',
  initialScale: 1,
  maximumScale: 5,
  userScalable: true,
  themeColor: [
    { media: '(prefers-color-scheme: light)', color: '#ffffff' },
    { media: '(prefers-color-scheme: dark)', color: '#000000' },
  ],
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" className={inter.variable}>
      <body className={`${inter.className} antialiased`}>
        <AuthProvider>
          {children}
        </AuthProvider>
      </body>
    </html>
  )
}
