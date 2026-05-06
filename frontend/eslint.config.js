import js from "@eslint/js";
import tseslint from "typescript-eslint";
import reactHooks from "eslint-plugin-react-hooks";

export default tseslint.config(
  // Global ignores
  {
    ignores: ["dist/", "node_modules/", ".vite/", "coverage/", "e2e/"],
  },

  // Base JS recommended rules
  js.configs.recommended,

  // TypeScript recommended rules
  ...tseslint.configs.recommended,

  // React hooks rules + TypeScript file settings
  {
    files: ["src/**/*.{ts,tsx}"],
    plugins: {
      "react-hooks": reactHooks,
    },
    rules: {
      // React hooks — error level
      "react-hooks/rules-of-hooks": "error",
      "react-hooks/exhaustive-deps": "error",

      // TypeScript adjustments
      "@typescript-eslint/no-unused-vars": [
        "warn",
        { argsIgnorePattern: "^_", varsIgnorePattern: "^_" },
      ],
      "@typescript-eslint/no-explicit-any": "warn",
      "@typescript-eslint/no-empty-object-type": "off",

      // Gen-4 Roadmap conventions. See frontend/CONTRIBUTING.md.
      //
      // (a) Discourage inline `queryKey: [ ... ]` array literals. Query keys
      //     must come from a `<domain>Keys` factory in src/api/queries/.
      // (b) Discourage accepting `theme` as a prop. Read it from the theme
      //     context (`useThemeContext()`) instead.
      //
      // Uses `warn` (not `error`) so in-flight branches aren't blocked;
      // code review + this warn is enough to prevent drift.
      "no-restricted-syntax": [
        "warn",
        {
          selector:
            "Property[key.name='queryKey'][value.type='ArrayExpression']",
          message:
            "Use a `<domain>Keys` factory from src/api/queries/ instead of a raw array literal for queryKey. See frontend/CONTRIBUTING.md §1.",
        },
        {
          selector:
            "TSPropertySignature[key.name='theme']",
          message:
            "Do not accept `theme` as a prop. Read from `useThemeContext()` instead. See frontend/CONTRIBUTING.md §2.",
        },
      ],
    },
  },

  // Ban raw `fetch(...)` outside src/api/. All HTTP calls must go through the
  // typed query layer in src/api/queries/. Files under src/api/ are explicitly
  // allowed below. Currently `warn` to avoid blocking in-flight branches —
  // TODO: tighten to `error` once existing offenders are migrated.
  {
    files: [
      "src/components/**/*.{ts,tsx}",
      "src/tabs/**/*.{ts,tsx}",
      "src/hooks/**/*.{ts,tsx}",
    ],
    rules: {
      "no-restricted-globals": [
        "warn",
        {
          name: "fetch",
          message:
            "Do not call fetch() directly outside src/api/. Use a typed query function from src/api/queries/. See frontend/CONTRIBUTING.md.",
        },
      ],
    },
  },

  // src/api/ is the only place allowed to call fetch directly.
  {
    files: ["src/api/**/*.{ts,tsx}"],
    rules: {
      "no-restricted-globals": "off",
    },
  }
);
