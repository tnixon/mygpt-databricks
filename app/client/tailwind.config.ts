import type { Config } from 'tailwindcss';
import tailwindcssAnimate from 'tailwindcss-animate';

const config: Config = {
  darkMode: ['class', 'media'],
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      fontFamily: {
        display: ['Edge', 'Helvetica Neue', 'Helvetica', 'Arial', 'sans-serif'],
        sans: ['Helvetica Neue', 'Helvetica', 'Arial', 'ui-sans-serif', 'system-ui', 'sans-serif'],
      },
      // St. Jude brand colors (from their MyGPT tailwind.config) — available as
      // bg-nav / text-stjude / bg-panel3 / etc. Semantic tokens (primary, sidebar,
      // …) are themed via CSS vars in index.css.
      colors: {
        stjude: '#911938',
        nav: '#2A4759',
        panel1: '#687E8C',
        panel2: '#BFBCBA',
        panel3: '#B4CED9',
        'bsk-blue': '#CFE5FF',
        'bsk-dark-blue': '#7F97B3',
        'bsk-opp': '#CCA791',
      },
    },
  },
  plugins: [tailwindcssAnimate],
};

export default config;
