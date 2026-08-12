/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        // Royal blue used across the design for nav, hero and primary CTAs.
        brand: {
          50: '#eef1ff',
          100: '#dfe4ff',
          200: '#c2caff',
          500: '#3b53e6',
          600: '#2a3fd4',
          700: '#2133b0',
        },
        // Magenta/pink accent from the hero "Tư vấn ngay" button.
        accent: {
          400: '#e879c8',
          500: '#dc4fbd',
          600: '#c33aa6',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
      },
    },
  },
  plugins: [],
}
