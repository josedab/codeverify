import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    environment: "node",
    include: ["src/**/*.{test,spec}.{ts,tsx}"],
    exclude: [
      "e2e/**",
      ".next/**",
      "out/**",
      "dist/**",
      "coverage/**",
      "playwright-report/**",
      "test-results/**",
      "**/node_modules/**",
    ],
  },
});
