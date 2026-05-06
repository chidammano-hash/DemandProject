import "@testing-library/jest-dom/vitest";

// ResizeObserver polyfill for recharts in jsdom
global.ResizeObserver = class ResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
};

// IntersectionObserver polyfill — jsdom doesn't implement it. LazyPanel uses
// it to defer below-the-fold panels until scrolled into view; in tests we want
// panels to render eagerly so existing assertions keep passing.
class MockIntersectionObserver {
  private readonly callback: IntersectionObserverCallback;
  readonly root: Element | Document | null = null;
  readonly rootMargin: string = "0px";
  readonly thresholds: ReadonlyArray<number> = [0];
  constructor(callback: IntersectionObserverCallback) {
    this.callback = callback;
  }
  observe(target: Element) {
    // Fire synchronously with isIntersecting=true so LazyPanel mounts children.
    const entry: IntersectionObserverEntry = {
      target,
      isIntersecting: true,
      intersectionRatio: 1,
      boundingClientRect: target.getBoundingClientRect(),
      intersectionRect: target.getBoundingClientRect(),
      rootBounds: null,
      time: Date.now(),
    };
    this.callback([entry], this as unknown as IntersectionObserver);
  }
  unobserve() {}
  disconnect() {}
  takeRecords(): IntersectionObserverEntry[] { return []; }
}
global.IntersectionObserver = MockIntersectionObserver as unknown as typeof IntersectionObserver;
