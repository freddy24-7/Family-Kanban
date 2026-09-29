import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'
import { VitePWA } from 'vite-plugin-pwa'

export default defineConfig({
  plugins: [
    react(),
    tailwindcss(),
    VitePWA({
      registerType: 'autoUpdate',
      includeAssets: ['favicon.svg', 'apple-touch-icon.png'],
      manifest: {
        name: 'Gezinsbord',
        short_name: 'Gezinsbord',
        description: 'Taken van het hele gezin, op één bord.',
        lang: 'nl',
        start_url: '/',
        display: 'standalone',
        background_color: '#f9f9f7',
        theme_color: '#2a78d6',
        icons: [
          { src: 'pwa-192.png', sizes: '192x192', type: 'image/png' },
          { src: 'pwa-512.png', sizes: '512x512', type: 'image/png' },
          { src: 'maskable-512.png', sizes: '512x512', type: 'image/png', purpose: 'maskable' },
        ],
      },
      workbox: {
        // Offline shell only: API calls always go to the network.
        navigateFallback: '/index.html',
        navigateFallbackDenylist: [/^\/api/],
      },
    }),
  ],
})
