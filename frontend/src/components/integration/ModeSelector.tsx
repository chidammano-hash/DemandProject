import type { JSX } from "react";

export type LoadMode = "onetime" | "delta" | "file";

interface ModeSelectorProps {
  value: LoadMode;
  onChange: (mode: LoadMode) => void;
  disabled?: boolean;
  label?: string;
  id?: string;
  /** Optional per-mode disable map. Key = mode, value = tooltip reason. */
  disabledModes?: Partial<Record<LoadMode, string>>;
  /** Optional per-mode description override (e.g. partition-specific hints). */
  descriptionOverrides?: Partial<Record<LoadMode, string>>;
}

interface ModeOption {
  value: LoadMode;
  title: string;
  description: string;
}

const OPTIONS: ModeOption[] = [
  {
    value: "onetime",
    title: "One-time",
    description: "Full reload (TRUNCATE + INSERT)",
  },
  {
    value: "delta",
    title: "Delta",
    description: "Skip if unchanged; otherwise upsert",
  },
  {
    value: "file",
    title: "File",
    description: "Reload a single slice or file",
  },
];

export function ModeSelector({
  value,
  onChange,
  disabled = false,
  label = "Mode",
  id = "mode-selector",
  disabledModes,
  descriptionOverrides,
}: ModeSelectorProps): JSX.Element {
  const groupName = `${id}-radio-group`;
  return (
    <div className="flex flex-col gap-1">
      <span
        id={`${id}-label`}
        className="text-xs font-medium uppercase tracking-wide text-muted-foreground"
      >
        {label}
      </span>
      <div
        role="radiogroup"
        aria-labelledby={`${id}-label`}
        className="grid grid-cols-1 gap-2 lg:grid-cols-3"
      >
        {OPTIONS.map((option) => {
          const optionId = `${id}-${option.value}`;
          const descriptionId = `${optionId}-desc`;
          const selected = value === option.value;
          const perModeReason = disabledModes?.[option.value];
          const isModeDisabled = disabled || perModeReason !== undefined;
          const base = "flex flex-col gap-1 rounded-md p-3 text-left transition-colors";
          const border = selected
            ? "border-2 border-blue-500 bg-blue-50 dark:bg-blue-950/30"
            : "border border-border hover:border-blue-300";
          const cursor = isModeDisabled ? "cursor-not-allowed opacity-50" : "cursor-pointer";
          const cardClasses = `${base} ${border} ${cursor}`.trim();
          return (
            <label
              key={option.value}
              htmlFor={optionId}
              className={cardClasses}
              title={perModeReason}
            >
              <div className="flex items-center gap-2">
                <input
                  id={optionId}
                  type="radio"
                  name={groupName}
                  value={option.value}
                  checked={selected}
                  disabled={isModeDisabled}
                  onChange={() => onChange(option.value)}
                  aria-describedby={descriptionId}
                  className="h-4 w-4 accent-blue-500"
                />
                <span className="text-sm font-semibold text-foreground">{option.title}</span>
              </div>
              <span id={descriptionId} className="pl-6 text-xs text-muted-foreground">
                {perModeReason ?? descriptionOverrides?.[option.value] ?? option.description}
              </span>
            </label>
          );
        })}
      </div>
    </div>
  );
}
