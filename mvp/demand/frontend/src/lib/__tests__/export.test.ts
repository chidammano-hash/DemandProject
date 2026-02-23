import { describe, it, expect, vi, beforeEach } from "vitest";
import { downloadCsv } from "@/lib/export";

describe("downloadCsv", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("does nothing for empty data", () => {
    const createSpy = vi.spyOn(document, "createElement");
    downloadCsv([], "test.csv");
    expect(createSpy).not.toHaveBeenCalled();
  });

  it("creates download link for valid data", () => {
    const clickSpy = vi.fn();
    vi.spyOn(document, "createElement").mockReturnValue({
      href: "",
      download: "",
      click: clickSpy,
    } as unknown as HTMLAnchorElement);
    vi.spyOn(URL, "createObjectURL").mockReturnValue("blob:test");
    vi.spyOn(URL, "revokeObjectURL").mockImplementation(() => {});

    downloadCsv([{ a: 1, b: "hello" }], "test.csv");

    expect(clickSpy).toHaveBeenCalled();
    expect(URL.revokeObjectURL).toHaveBeenCalledWith("blob:test");
  });

  it("uses provided columns order", () => {
    const clickSpy = vi.fn();
    vi.spyOn(document, "createElement").mockReturnValue({
      href: "",
      download: "",
      click: clickSpy,
    } as unknown as HTMLAnchorElement);
    vi.spyOn(URL, "createObjectURL").mockReturnValue("blob:test");
    vi.spyOn(URL, "revokeObjectURL").mockImplementation(() => {});

    downloadCsv([{ a: 1, b: 2, c: 3 }], "test.csv", ["c", "a"]);
    expect(clickSpy).toHaveBeenCalled();
  });

  it("sets correct filename on link", () => {
    const link = { href: "", download: "", click: vi.fn() };
    vi.spyOn(document, "createElement").mockReturnValue(link as unknown as HTMLAnchorElement);
    vi.spyOn(URL, "createObjectURL").mockReturnValue("blob:test");
    vi.spyOn(URL, "revokeObjectURL").mockImplementation(() => {});

    downloadCsv([{ x: 1 }], "my_export.csv");
    expect(link.download).toBe("my_export.csv");
  });
});
