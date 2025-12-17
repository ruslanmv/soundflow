import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{js,ts,jsx,tsx,mdx}", "./components/**/*.{js,ts,jsx,tsx,mdx}", "./lib/**/*.{js,ts,jsx,tsx,mdx}"],
  theme: {
    extend: {
      colors: {
        primary: { 50: "#f0f9ff", 100: "#e0f2fe", 500: "#0ea5e9", 600: "#0284c7" },
        teal: { 400: "#2dd4bf", 500: "#14b8a6" },
        purple: { 400: "#c084fc", 500: "#a855f7" },
        blue: { 400: "#60a5fa", 500: "#3b82f6" }
      },
      fontFamily: { sans: ["Inter", "system-ui", "sans-serif"] },
      animation: { "pulse-slow": "pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite", wave: "wave 1.5s ease-in-out infinite" },
      keyframes: { wave: { "0%, 100%": { transform: "scaleY(0.4)" }, "50%": { transform: "scaleY(1)" } } },
      backdropBlur: { xs: "2px" }
    }
  },
  plugins: []
};
export default config;
