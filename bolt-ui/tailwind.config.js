/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        bolt: {
          dark: '#0f1117',    // 主背景色
          panel: '#1e212b',   // 面板背景
          border: '#2e3344',  // 边框色
          accent: '#3b82f6',  // 强调色 (蓝色)
        }
      },
      animation: {
        'pulse-fast': 'pulse 1s cubic-bezier(0.4, 0, 0.6, 1) infinite',
      }
    },
  },
  plugins: [],
}