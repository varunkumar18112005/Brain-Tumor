/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        medical: {
          50: '#f0f7ff',
          100: '#e0effe',
          200: '#bae0fd',
          300: '#7cc7fb',
          400: '#38a8f8',
          500: '#0e8ce9',
          600: '#026ec7',
          700: '#0358a1',
          800: '#074a85',
          900: '#0c3f6e',
          950: '#082849',
        },
        tealbrand: {
          500: '#0d9488',
          600: '#0f766e',
          700: '#115e59',
        }
      }
    },
  },
  plugins: [],
}
