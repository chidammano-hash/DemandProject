import type { Config } from "tailwindcss";

const config: Config = {
  darkMode: ["class"],
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        border: "hsl(var(--border))",
        input: "hsl(var(--input))",
        ring: "hsl(var(--ring))",
        background: "hsl(var(--background))",
        foreground: "hsl(var(--foreground))",
        primary: {
          DEFAULT: "hsl(var(--primary))",
          foreground: "hsl(var(--primary-foreground))",
        },
        secondary: {
          DEFAULT: "hsl(var(--secondary))",
          foreground: "hsl(var(--secondary-foreground))",
        },
        muted: {
          DEFAULT: "hsl(var(--muted))",
          foreground: "hsl(var(--muted-foreground))",
        },
        accent: {
          DEFAULT: "hsl(var(--accent))",
          foreground: "hsl(var(--accent-foreground))",
        },
        destructive: {
          DEFAULT: "hsl(var(--destructive))",
          foreground: "hsl(var(--destructive-foreground))",
        },
        card: {
          DEFAULT: "hsl(var(--card))",
          foreground: "hsl(var(--card-foreground))",
        },
      },
      borderRadius: {
        lg: "var(--radius)",
        md: "calc(var(--radius) - 2px)",
        sm: "calc(var(--radius) - 4px)",
      },
      fontFamily: {
        sans: ["Inter", "Avenir Next", "Trebuchet MS", "Segoe UI", "sans-serif"],
      },
      keyframes: {
        "fade-in": {
          from: { opacity: "0", transform: "translateY(6px)" },
          to: { opacity: "1", transform: "translateY(0)" },
        },
        "pulse-glow": {
          "0%, 100%": { boxShadow: "0 0 0 0 rgba(99, 102, 241, 0.4)" },
          "50%": { boxShadow: "0 0 16px 4px rgba(99, 102, 241, 0.25)" },
        },
        "orbit-spin": {
          "0%": { transform: "rotate(0deg) scale(1)", opacity: "0.3" },
          "50%": { transform: "rotate(180deg) scale(1.15)", opacity: "0.9" },
          "100%": { transform: "rotate(360deg) scale(1)", opacity: "0.3" },
        },
        "flame-flicker": {
          "0%, 100%": { opacity: "0.6", filter: "brightness(1)" },
          "25%": { opacity: "1", filter: "brightness(1.3)" },
          "50%": { opacity: "0.8", filter: "brightness(1.1)" },
          "75%": { opacity: "1", filter: "brightness(1.2)" },
        },
        "zen-breathe": {
          "0%, 100%": { opacity: "0.7", transform: "scale(1)" },
          "50%": { opacity: "1", transform: "scale(1.04)" },
        },
        "pour-shimmer": {
          "0%": { backgroundPosition: "-200% 0" },
          "100%": { backgroundPosition: "200% 0" },
        },
      },
      animation: {
        "fade-in": "fade-in 250ms ease-out",
        "pulse-glow": "pulse-glow 1.5s ease-in-out infinite",
        "orbit-spin": "orbit-spin 3s linear infinite",
        "flame-flicker": "flame-flicker 0.8s ease-in-out infinite",
        "zen-breathe": "zen-breathe 4s ease-in-out infinite",
        "pour-shimmer": "pour-shimmer 2s ease-in-out infinite",
      },
    },
  },
  plugins: [],
};

export default config;
