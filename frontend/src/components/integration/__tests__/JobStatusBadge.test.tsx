import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { JobStatusBadge } from "../JobStatusBadge";
import type { Job } from "../../../api/queries/integration";

type JobStatus = Job["status"];

const STATUS_COLORS: Record<JobStatus, string> = {
  queued: "gray",
  running: "blue",
  success: "green",
  failed: "red",
  skipped: "yellow",
};

describe("JobStatusBadge", () => {
  it("renders the status label text", () => {
    render(<JobStatusBadge status="success" />);
    // Component renders the raw word; CSS uppercases via tracking-wide class.
    expect(screen.getByText("success")).toBeInTheDocument();
  });

  it("exposes the status as an aria-label for screen readers", () => {
    render(<JobStatusBadge status="failed" />);
    expect(screen.getByLabelText("status: failed")).toBeInTheDocument();
  });

  it("applies pulse animation only when status is running", () => {
    const { container, rerender } = render(<JobStatusBadge status="running" />);
    // The dot span sits inside the badge; find it by aria-hidden marker.
    const runningDot = container.querySelector('[aria-hidden="true"]');
    expect(runningDot).not.toBeNull();
    expect(runningDot?.className).toContain("animate-pulse");

    rerender(<JobStatusBadge status="success" />);
    const successDot = container.querySelector('[aria-hidden="true"]');
    expect(successDot).not.toBeNull();
    expect(successDot?.className ?? "").not.toContain("animate-pulse");
  });

  it.each<JobStatus>(["queued", "running", "success", "failed", "skipped"])(
    "renders status %s with correct base color class",
    (status) => {
      const { container } = render(<JobStatusBadge status={status} />);
      const badge = container.firstElementChild as HTMLElement | null;
      expect(badge).not.toBeNull();
      const color = STATUS_COLORS[status];
      // Tailwind class like "bg-green-100" / "text-blue-700" — assert color token present.
      expect(badge!.className).toContain(color);
      // Status text always renders (rules out crash on any status).
      expect(screen.getByText(status)).toBeInTheDocument();
    },
  );

  it("uses the pill base classes (rounded-full, inline-flex)", () => {
    const { container } = render(<JobStatusBadge status="queued" />);
    const badge = container.firstElementChild as HTMLElement | null;
    expect(badge?.className).toContain("rounded-full");
    expect(badge?.className).toContain("inline-flex");
  });
});
