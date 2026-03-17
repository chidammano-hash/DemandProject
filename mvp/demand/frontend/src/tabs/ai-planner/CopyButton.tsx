/**
 * CopyButton — copies text to clipboard with transient "Copied!" state.
 */
import { useState } from "react";
import { Copy, Check } from "lucide-react";

export function CopyButton({ text, label = "Copy" }: { text: string; label?: string }) {
  const [copied, setCopied] = useState(false);
  function handleCopy() {
    navigator.clipboard.writeText(text).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  }
  return (
    <button
      onClick={handleCopy}
      className="flex items-center gap-1 rounded border border-input bg-background px-2 py-1 text-xs hover:bg-muted"
      title="Copy memo to clipboard"
    >
      {copied ? <Check className="h-3 w-3 text-green-600" /> : <Copy className="h-3 w-3" />}
      {copied ? "Copied!" : label}
    </button>
  );
}
