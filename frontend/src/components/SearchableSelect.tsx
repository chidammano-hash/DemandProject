import { useMemo, useRef, useState } from "react";

interface Props {
  value: string;
  options: string[];
  /** Shown when nothing is selected; also the "clear" option label. */
  placeholder: string;
  /** Accessible name for the combobox input. */
  ariaLabel: string;
  onChange: (value: string) => void;
  className?: string;
}

// Junk/sentinel values that pollute raw filter axes (e.g. Store Type). They are
// not dropped (a value could still be meaningful upstream) but demoted to the
// bottom so the type-ahead surfaces real segments first — U5.11.
const JUNK_RE = /^\s*(\*+|obsolete\b|all$|unknown\b|user defined$)/i;

/** True when an option looks like a sentinel/junk value rather than a real segment. */
export function isJunkOption(option: string): boolean {
  return JUNK_RE.test(option.trim());
}

/**
 * Stable sort that demotes junk/sentinel options to the bottom while preserving
 * the relative order of real values (and of junk values among themselves).
 */
export function demoteJunkOptions(options: string[]): string[] {
  return [...options].sort((a, b) => {
    const ja = isJunkOption(a) ? 1 : 0;
    const jb = isJunkOption(b) ? 1 : 0;
    return ja - jb; // Array.prototype.sort is stable in modern engines
  });
}

/**
 * A dependency-free searchable combobox: a text input (role=combobox) that
 * type-ahead-filters a long option list. Replaces native <select> for axes with
 * hundreds of options (Store Type has ~293) where scanning a flat list is
 * unusable. Junk/sentinel values are demoted to the bottom (U5.11).
 */
export function SearchableSelect({
  value,
  options,
  placeholder,
  ariaLabel,
  onChange,
  className = "w-36",
}: Props) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const blurTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const ordered = useMemo(() => demoteJunkOptions(options), [options]);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return ordered;
    return ordered.filter((o) => o.toLowerCase().includes(q));
  }, [ordered, query]);

  const select = (val: string) => {
    onChange(val);
    setQuery("");
    setOpen(false);
  };

  return (
    <div className={`relative ${className}`}>
      <input
        role="combobox"
        aria-label={ariaLabel}
        aria-expanded={open}
        aria-controls={`${ariaLabel}-listbox`}
        autoComplete="off"
        className="w-full px-2 py-1 text-sm border rounded"
        placeholder={placeholder}
        value={open ? query : value || ""}
        onFocus={() => {
          if (blurTimer.current) clearTimeout(blurTimer.current);
          setOpen(true);
          setQuery("");
        }}
        onChange={(e) => setQuery(e.target.value)}
        onBlur={() => {
          // Delay so a mouseDown on an option fires before close.
          blurTimer.current = setTimeout(() => setOpen(false), 120);
        }}
      />
      {open && (
        <ul
          id={`${ariaLabel}-listbox`}
          role="listbox"
          className="absolute z-50 mt-1 max-h-60 w-full overflow-auto rounded-md border border-border bg-card text-card-foreground shadow-elevated animate-scale-in origin-top"
        >
          <li
            role="option"
            aria-selected={value === ""}
            className="cursor-pointer px-2 py-1 text-sm text-muted-foreground hover:bg-accent"
            onMouseDown={() => select("")}
          >
            {placeholder}
          </li>
          {filtered.map((opt) => (
            <li
              key={opt}
              role="option"
              aria-selected={value === opt}
              className={`cursor-pointer px-2 py-1 text-sm hover:bg-accent ${
                isJunkOption(opt) ? "text-muted-foreground" : ""
              }`}
              onMouseDown={() => select(opt)}
            >
              {opt}
            </li>
          ))}
          {filtered.length === 0 && (
            <li className="px-2 py-1 text-sm text-muted-foreground">No matches</li>
          )}
        </ul>
      )}
    </div>
  );
}
