import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import {
  Select,
  SelectTrigger,
  SelectValue,
  SelectContent,
  SelectItem,
} from "@/components/ui/select";
import tailwindConfig from "../../../tailwind.config";

function Harness({ onChange = vi.fn() }: { onChange?: (v: string) => void }) {
  return (
    <Select value="a" onValueChange={onChange}>
      <SelectTrigger>
        <SelectValue placeholder="Pick" />
      </SelectTrigger>
      <SelectContent>
        <SelectItem value="a">Alpha</SelectItem>
        <SelectItem value="b">Beta</SelectItem>
      </SelectContent>
    </Select>
  );
}

describe("Select", () => {
  it("trigger reports collapsed state until opened", () => {
    render(<Harness />);
    const trigger = screen.getByRole("button");
    expect(trigger).toHaveAttribute("aria-haspopup", "listbox");
    expect(trigger).toHaveAttribute("aria-expanded", "false");
    expect(screen.queryByRole("listbox")).not.toBeInTheDocument();
  });

  it("opening the menu exposes a listbox and flips aria-expanded", async () => {
    render(<Harness />);
    const trigger = screen.getByRole("button");
    await userEvent.click(trigger);
    expect(trigger).toHaveAttribute("aria-expanded", "true");
    expect(screen.getByRole("listbox")).toBeInTheDocument();
    expect(screen.getByText("Beta")).toBeInTheDocument();
  });

  it("uses an opaque, scroll-bounded popover surface", async () => {
    render(<Harness />);
    await userEvent.click(screen.getByRole("button"));

    expect(screen.getByRole("listbox")).toHaveClass(
      "bg-popover",
      "text-popover-foreground",
      "max-h-60",
      "overflow-y-auto",
    );
  });

  it("Escape closes the open menu (respect the keyboard)", async () => {
    render(<Harness />);
    await userEvent.click(screen.getByRole("button"));
    expect(screen.getByRole("listbox")).toBeInTheDocument();
    await userEvent.keyboard("{Escape}");
    expect(screen.queryByRole("listbox")).not.toBeInTheDocument();
  });

  it("selecting an option emits the value and closes", async () => {
    const onChange = vi.fn();
    render(<Harness onChange={onChange} />);
    await userEvent.click(screen.getByRole("button"));
    await userEvent.click(screen.getByText("Beta"));
    expect(onChange).toHaveBeenCalledWith("b");
    expect(screen.queryByRole("listbox")).not.toBeInTheDocument();
  });
});

describe("Select theme contract", () => {
  it("maps the popover utilities to theme tokens", () => {
    const colors = tailwindConfig.theme?.extend?.colors;
    expect(colors).toMatchObject({
      popover: {
        DEFAULT: "hsl(var(--popover))",
        foreground: "hsl(var(--popover-foreground))",
      },
    });
  });

  it("keeps popovers opaque across runtime card themes", () => {
    const css = readFileSync(resolve("src/index.css"), "utf8");
    expect(css).toContain("--popover: var(--card);");
    expect(css).toContain("--popover-foreground: var(--card-foreground);");
  });
});
