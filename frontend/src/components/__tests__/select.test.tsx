import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { describe, it, expect, vi } from "vitest";
import { render, screen, within } from "@testing-library/react";
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

// Radix Select renders its trigger with role="combobox" (per the WAI-ARIA
// combobox pattern) and portals SelectContent to document.body when open —
// `screen` already queries the whole document, so no extra portal plumbing
// is needed in these queries.
describe("Select", () => {
  it("trigger reports collapsed state until opened", () => {
    render(<Harness />);
    const trigger = screen.getByRole("combobox");
    expect(trigger).toHaveAttribute("aria-expanded", "false");
    expect(screen.queryByRole("listbox")).not.toBeInTheDocument();
  });

  it("shows the label of the selected value before the menu is ever opened", () => {
    render(<Harness />);
    expect(screen.getByRole("combobox")).toHaveTextContent("Alpha");
  });

  it("opening the menu exposes a listbox and flips aria-expanded", async () => {
    const user = userEvent.setup();
    render(<Harness />);
    const trigger = screen.getByRole("combobox");
    await user.click(trigger);
    expect(trigger).toHaveAttribute("aria-expanded", "true");
    const listbox = screen.getByRole("listbox");
    expect(listbox).toBeInTheDocument();
    expect(within(listbox).getByText("Beta")).toBeInTheDocument();
  });

  it("uses an opaque, scroll-bounded popover surface", async () => {
    const user = userEvent.setup();
    render(<Harness />);
    await user.click(screen.getByRole("combobox"));

    expect(screen.getByRole("listbox")).toHaveClass(
      "bg-popover",
      "text-popover-foreground",
      "max-h-60",
      "overflow-y-auto",
    );
  });

  it("Escape closes the open menu (respect the keyboard)", async () => {
    const user = userEvent.setup();
    render(<Harness />);
    await user.click(screen.getByRole("combobox"));
    expect(screen.getByRole("listbox")).toBeInTheDocument();
    await user.keyboard("{Escape}");
    expect(screen.queryByRole("listbox")).not.toBeInTheDocument();
  });

  it("selecting an option emits the value and closes", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    render(<Harness onChange={onChange} />);
    await user.click(screen.getByRole("combobox"));
    await user.click(screen.getByRole("option", { name: "Beta" }));
    expect(onChange).toHaveBeenCalledWith("b");
    expect(screen.queryByRole("listbox")).not.toBeInTheDocument();
  });

  it("marks the currently selected option with aria-selected/data-state", async () => {
    const user = userEvent.setup();
    render(<Harness />);
    await user.click(screen.getByRole("combobox"));
    expect(screen.getByRole("option", { name: "Alpha" })).toHaveAttribute("data-state", "checked");
    expect(screen.getByRole("option", { name: "Beta" })).toHaveAttribute("data-state", "unchecked");
  });

  it("keeps data-value on each option for consumers/tests that query by raw value", async () => {
    const user = userEvent.setup();
    render(<Harness />);
    await user.click(screen.getByRole("combobox"));
    expect(screen.getByRole("option", { name: "Alpha" })).toHaveAttribute("data-value", "a");
    expect(screen.getByRole("option", { name: "Beta" })).toHaveAttribute("data-value", "b");
  });

  it("opening via the keyboard (Enter) exposes the listbox", async () => {
    const user = userEvent.setup();
    render(<Harness />);
    const trigger = screen.getByRole("combobox");
    trigger.focus();
    await user.keyboard("{Enter}");
    expect(screen.getByRole("listbox")).toBeInTheDocument();
  });

  it("shows the placeholder when no value is selected", () => {
    render(
      <Select value="" onValueChange={vi.fn()}>
        <SelectTrigger>
          <SelectValue placeholder="Pick" />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="a">Alpha</SelectItem>
        </SelectContent>
      </Select>,
    );
    expect(screen.getByRole("combobox")).toHaveTextContent("Pick");
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
