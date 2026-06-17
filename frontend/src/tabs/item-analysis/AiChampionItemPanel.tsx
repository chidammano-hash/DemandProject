/**
 * Interactive AI Champion adjuster for a single DFU (Item Analysis tab).
 *
 * Shows the promoted champion forward forecast and an "AI Adjust" button that
 * calls an LLM (Ollama / Google / Anthropic / OpenAI) with the item + location
 * attributes to propose an adjustment. The planner previews the result, then
 * optionally Saves it as the ai_champion forecast.
 * Spec: docs/specs/02-forecasting/27-ai-champion-forecast.md
 */
import { useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Loader2, Sparkles } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import {
  AI_CHAMPION_PROVIDERS,
  type AiChampionPreview,
  type AiChampionProvider,
  adjustAiChampion,
  aiChampionKeys,
  fetchAiChampionSaved,
  saveAiChampion,
} from "@/api/queries/ai-champion";

interface AiChampionItemPanelProps {
  itemId: string;
  loc: string;
}

interface Row {
  forecast_month: string | null;
  champion_qty: number | null;
  ai_qty: number | null;
  pct_change: number | null;
}

function formatQty(value: number | null): string {
  if (value == null) return "—";
  return value.toLocaleString(undefined, { maximumFractionDigits: 1 });
}

function formatPct(value: number | null): string {
  if (value == null) return "—";
  const sign = value > 0 ? "+" : "";
  return `${sign}${value.toFixed(1)}%`;
}

export function AiChampionItemPanel({ itemId, loc }: AiChampionItemPanelProps) {
  const item = itemId.trim();
  const location = loc.trim();
  const enabled = item.length > 0 && location.length > 0;
  const queryClient = useQueryClient();

  const [provider, setProvider] = useState<AiChampionProvider>("ollama");
  const [userComment, setUserComment] = useState("");
  const [preview, setPreview] = useState<AiChampionPreview | null>(null);

  const { data: saved } = useQuery({
    queryKey: aiChampionKeys.saved(item, location),
    queryFn: () => fetchAiChampionSaved(item, location),
    enabled,
    staleTime: 60_000,
  });

  const adjustMutation = useMutation({
    mutationFn: () =>
      adjustAiChampion({
        item_id: item,
        loc: location,
        provider,
        ...(userComment.trim() ? { user_comment: userComment.trim() } : {}),
      }),
    onSuccess: (data) => setPreview(data),
  });

  const saveMutation = useMutation({
    mutationFn: () => {
      if (!preview) throw new Error("nothing to save");
      return saveAiChampion({ item_id: item, loc: location, provider, preview });
    },
    onSuccess: () => {
      setPreview(null);
      queryClient.invalidateQueries({ queryKey: aiChampionKeys.saved(item, location) });
    },
  });

  // Preview (unsaved) takes precedence over the saved view.
  const showingPreview = preview != null;
  const savedRows = saved?.rows ?? [];
  const savedLead = savedRows[0] ?? null;

  const rows: Row[] = useMemo(() => {
    if (preview) return preview.months;
    return savedRows;
  }, [preview, savedRows]);

  const recCode = preview?.recommendation_code ?? savedLead?.recommendation_code ?? null;
  const confidence = preview?.confidence ?? savedLead?.confidence ?? null;
  const rationale = preview?.rationale ?? savedLead?.rationale ?? null;

  if (!enabled) return null;

  return (
    <Card>
      <CardContent className="space-y-3 pt-4">
        <div className="flex flex-col gap-2 md:flex-row md:items-start md:justify-between">
          <div>
            <div className="flex items-center gap-2">
              <Sparkles className="h-4 w-4 text-amber-600 dark:text-amber-400" />
              <h3 className="text-sm font-medium text-foreground">AI Champion Adjuster</h3>
            </div>
            <p className="mt-0.5 max-w-2xl text-xs text-muted-foreground">
              Adjust the promoted champion forward forecast for this DFU using an LLM with the
              item &amp; location attributes. Forward-only — not graded in the FVA ladder.
            </p>
          </div>
          <div className="flex items-center gap-2">
            <select
              className="h-8 rounded border border-input bg-background px-2 text-xs"
              value={provider}
              onChange={(e) => setProvider(e.target.value as AiChampionProvider)}
              aria-label="AI provider"
            >
              {AI_CHAMPION_PROVIDERS.map((p) => (
                <option key={p.value} value={p.value}>{p.label}</option>
              ))}
            </select>
            <Button
              size="sm"
              variant="outline"
              disabled={adjustMutation.isPending}
              onClick={() => adjustMutation.mutate()}
            >
              {adjustMutation.isPending ? (
                <><Loader2 className="mr-2 h-4 w-4 animate-spin" />Adjusting…</>
              ) : (
                "AI Adjust"
              )}
            </Button>
          </div>
        </div>

        <label className="block">
          <span className="mb-1 block text-xs font-medium text-muted-foreground">
            Planner comment (optional) — steers the AI, e.g. a known promo, listing change, or local event
          </span>
          <textarea
            className="min-h-[52px] w-full rounded border border-input bg-background px-2 py-1.5 text-xs"
            placeholder="e.g. New listing at a major customer ramps from May; expect +20% through summer."
            value={userComment}
            onChange={(e) => setUserComment(e.target.value)}
            maxLength={2000}
          />
        </label>

        {adjustMutation.isError && (
          <p className="rounded-md border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-800 dark:border-red-800 dark:bg-red-950/30 dark:text-red-300">
            Adjustment failed. Check that a champion forecast exists for this DFU and the provider is reachable.
          </p>
        )}

        {recCode == null ? (
          <p className="text-sm text-muted-foreground">
            No saved adjustment for {item}-{location}. Pick a provider and click AI Adjust.
          </p>
        ) : (
          <>
            <div className="flex flex-wrap items-center gap-2 text-xs">
              <span
                className={`rounded-full px-2 py-0.5 font-medium ${
                  showingPreview
                    ? "border border-amber-300 bg-amber-50 text-amber-800 dark:border-amber-700 dark:bg-amber-950/30 dark:text-amber-300"
                    : "border border-green-300 bg-green-50 text-green-800 dark:border-green-700 dark:bg-green-950/30 dark:text-green-300"
                }`}
              >
                {showingPreview ? "Preview — not saved" : "Saved"}
              </span>
              <span className="rounded-full border border-border bg-muted/40 px-2 py-0.5 font-medium">
                {recCode}
              </span>
              {confidence != null && (
                <span className="text-muted-foreground">confidence {(confidence * 100).toFixed(0)}%</span>
              )}
              {showingPreview && (
                <Button
                  size="sm"
                  className="ml-auto h-7"
                  disabled={saveMutation.isPending}
                  onClick={() => saveMutation.mutate()}
                >
                  {saveMutation.isPending ? "Saving…" : "Save"}
                </Button>
              )}
            </div>

            {rationale && (
              <p className="rounded-md border border-border/60 bg-muted/20 px-3 py-2 text-xs text-foreground">
                {rationale}
              </p>
            )}

            <div className="overflow-x-auto">
              <table className="w-full min-w-[420px] text-left text-xs">
                <thead>
                  <tr className="border-b border-border text-muted-foreground">
                    <th className="py-2 pr-2 font-medium">Month</th>
                    <th className="py-2 pr-2 font-medium">Champion</th>
                    <th className="py-2 pr-2 font-medium">AI</th>
                    <th className="py-2 font-medium">Δ%</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((r) => (
                    <tr key={r.forecast_month} className="border-b border-border/40">
                      <td className="py-1.5 pr-2">{r.forecast_month?.slice(0, 7) ?? "—"}</td>
                      <td className="py-1.5 pr-2 font-mono">{formatQty(r.champion_qty)}</td>
                      <td className="py-1.5 pr-2 font-mono">{formatQty(r.ai_qty)}</td>
                      <td className="py-1.5 font-mono">{formatPct(r.pct_change)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );
}
