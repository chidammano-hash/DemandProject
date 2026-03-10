import "@testing-library/jest-dom/vitest";

// ResizeObserver polyfill for recharts in jsdom
global.ResizeObserver = class ResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
};
