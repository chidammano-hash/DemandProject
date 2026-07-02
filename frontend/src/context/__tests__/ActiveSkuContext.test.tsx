import { describe, it, expect } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { useState } from "react";
import {
  ActiveSkuProvider,
  useActiveSku,
  usePublishActiveSku,
} from "@/context/ActiveSkuContext";

function Reader() {
  const sku = useActiveSku();
  return <div data-testid="reader">{sku ? `${sku.item}@${sku.loc}` : "none"}</div>;
}

function Publisher({ item, loc }: { item: string; loc: string }) {
  usePublishActiveSku(item, loc);
  return null;
}

describe("ActiveSkuContext", () => {
  it("publishes the SKU a page shows and a reader inherits it", () => {
    render(
      <ActiveSkuProvider>
        <Publisher item="100320" loc="DC1" />
        <Reader />
      </ActiveSkuProvider>,
    );
    expect(screen.getByTestId("reader").textContent).toBe("100320@DC1");
  });

  it("clears the active SKU when the publishing page unmounts", () => {
    function Harness() {
      const [mounted, setMounted] = useState(true);
      return (
        <ActiveSkuProvider>
          {mounted && <Publisher item="100320" loc="DC1" />}
          <Reader />
          <button type="button" onClick={() => setMounted(false)}>
            unmount
          </button>
        </ActiveSkuProvider>
      );
    }
    render(<Harness />);
    expect(screen.getByTestId("reader").textContent).toBe("100320@DC1");
    fireEvent.click(screen.getByText("unmount"));
    expect(screen.getByTestId("reader").textContent).toBe("none");
  });

  it("trims input and ignores an empty SKU", () => {
    render(
      <ActiveSkuProvider>
        <Publisher item="  " loc="  " />
        <Reader />
      </ActiveSkuProvider>,
    );
    expect(screen.getByTestId("reader").textContent).toBe("none");
  });

  it("is tolerant of a missing provider — read is null, publish is a no-op", () => {
    // No ActiveSkuProvider mounted; neither hook should throw.
    render(
      <>
        <Publisher item="X" loc="Y" />
        <Reader />
      </>,
    );
    expect(screen.getByTestId("reader").textContent).toBe("none");
  });
});
