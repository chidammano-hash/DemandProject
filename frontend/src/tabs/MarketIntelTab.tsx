import { useState, useEffect, useRef } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import {
  fetchMarketIntel,
  fetchDomainSuggest,
  STALE,
} from "@/api/queries";
import type { MarketIntelPayload } from "@/types";
import { useGlobalFilterContext } from "@/context/GlobalFilterContext";
import { LoadingElement } from "@/components/LoadingElement";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Globe, Loader2, RefreshCcw, Send } from "lucide-react";

export default function MarketIntelTab() {
  const { filters: globalFilters } = useGlobalFilterContext();
  const [miItemFilter, setMiItemFilter] = useState("");
  const [miLocationFilter, setMiLocationFilter] = useState("");
  const [miResult, setMiResult] = useState<MarketIntelPayload | null>(null);
  const [miError, setMiError] = useState("");

  // Sync global item/location filter into local inputs
  const syncedGlobalRef = useRef<string>("");
  useEffect(() => {
    const key = `${globalFilters.item.join(",")}_${globalFilters.location.join(",")}`;
    if (key === syncedGlobalRef.current) return;
    syncedGlobalRef.current = key;
    if (globalFilters.item.length === 1) setMiItemFilter(globalFilters.item[0]);
    if (globalFilters.location.length === 1) setMiLocationFilter(globalFilters.location[0]);
  }, [globalFilters.item, globalFilters.location]);

  // Item typeahead suggestions
  const { data: itemSuggestions = [] } = useQuery({
    queryKey: ["mi-item-suggest", miItemFilter],
    queryFn: () => fetchDomainSuggest("item", "item_id", miItemFilter.trim(), undefined, 12),
    staleTime: STALE.THIRTY_SEC,
    enabled: true,
  });

  // Location typeahead suggestions
  const { data: locationSuggestions = [] } = useQuery({
    queryKey: ["mi-location-suggest", miLocationFilter],
    queryFn: () => fetchDomainSuggest("location", "location_id", miLocationFilter.trim(), undefined, 12),
    staleTime: STALE.THIRTY_SEC,
    enabled: true,
  });

  // Generate briefing mutation
  const generateMutation = useMutation({
    mutationFn: () => fetchMarketIntel(miItemFilter, miLocationFilter),
    onSuccess: (data) => {
      setMiResult(data);
      setMiError("");
    },
    onError: (err: Error) => {
      setMiError(err.message || "Network error");
    },
  });

  const miLoading = generateMutation.isPending;

  function generateMarketIntel() {
    if (!miItemFilter.trim() || !miLocationFilter.trim() || miLoading) return;
    setMiError("");
    setMiResult(null);
    generateMutation.mutate();
  }

  return (
    <Card className="mt-4 animate-fade-in">
      <CardHeader>
        <div className="flex items-center gap-2">
          <Globe className="h-5 w-5" />
          <CardTitle className="text-base">Market Intelligence</CardTitle>
        </div>
        <CardDescription>
          Select a product and location to generate an AI-powered market briefing with web search
          results and demographic context.
        </CardDescription>
        <div className="grid gap-2 md:grid-cols-[1fr_1fr_auto]">
          <label className="space-y-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            Item (item_id)
            <Input
              className="h-9"
              placeholder="Search for item..."
              list="mi-item-suggest"
              value={miItemFilter}
              onChange={(e) => setMiItemFilter(e.target.value)}
            />
            <datalist id="mi-item-suggest">
              {itemSuggestions.map((val) => (
                <option key={val} value={val} />
              ))}
            </datalist>
          </label>
          <label className="space-y-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            Location (location_id)
            <Input
              className="h-9"
              placeholder="Search for location..."
              list="mi-location-suggest"
              value={miLocationFilter}
              onChange={(e) => setMiLocationFilter(e.target.value)}
            />
            <datalist id="mi-location-suggest">
              {locationSuggestions.map((val) => (
                <option key={val} value={val} />
              ))}
            </datalist>
          </label>
          <div className="flex items-end">
            <Button
              onClick={generateMarketIntel}
              disabled={!miItemFilter.trim() || !miLocationFilter.trim() || miLoading}
              className="h-9"
            >
              {miLoading ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" /> Generating...
                </>
              ) : (
                <>
                  <Send className="mr-2 h-4 w-4" /> Generate Briefing
                </>
              )}
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {miError ? (
          <Card className="border-destructive/30 bg-destructive/10">
            <CardContent className="pt-4 flex items-center justify-between gap-2">
              <span className="text-sm text-destructive">{miError}</span>
              <Button
                variant="outline"
                size="sm"
                onClick={() => {
                  setMiError("");
                  generateMarketIntel();
                }}
              >
                <RefreshCcw className="mr-1 h-3.5 w-3.5" /> Retry
              </Button>
            </CardContent>
          </Card>
        ) : null}

        {miLoading ? (
          <LoadingElement
            message="Searching the web and generating market briefing..."
          />
        ) : null}

        {miResult && !miLoading ? (
          <>
            {/* Product + Location context badges */}
            <div className="flex flex-wrap gap-2">
              {miResult.item_desc ? <Badge variant="secondary">{miResult.item_desc}</Badge> : null}
              {miResult.brand_name ? (
                <Badge variant="secondary">{miResult.brand_name}</Badge>
              ) : null}
              {miResult.category ? <Badge variant="secondary">{miResult.category}</Badge> : null}
              {miResult.state_id ? (
                <Badge variant="outline">State: {miResult.state_id}</Badge>
              ) : null}
              {miResult.site_desc ? (
                <Badge variant="outline">{miResult.site_desc}</Badge>
              ) : null}
            </div>

            {/* Search Results Cards */}
            {miResult.search_results.length > 0 ? (
              <div>
                <p className="text-xs uppercase tracking-wide text-muted-foreground mb-2">
                  Web Search Results ({miResult.search_results.length})
                </p>
                <div className="grid gap-2 md:grid-cols-2 lg:grid-cols-3">
                  {miResult.search_results.map((sr, idx) => (
                    <Card key={idx} className="border-muted bg-muted/10 shadow-none">
                      <CardContent className="pt-3 pb-3">
                        <a
                          href={sr.link}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-sm font-medium text-blue-700 hover:underline line-clamp-2"
                        >
                          {sr.title}
                        </a>
                        <p className="mt-1 text-xs text-muted-foreground line-clamp-3">
                          {sr.snippet}
                        </p>
                      </CardContent>
                    </Card>
                  ))}
                </div>
              </div>
            ) : null}

            {/* Narrative Story */}
            <Card className="border-sky-200 bg-sky-50/30">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm">Market Intelligence Briefing</CardTitle>
                <CardDescription className="text-xs">
                  Generated {new Date(miResult.generated_at).toLocaleString()}
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="prose prose-sm max-w-none text-sm whitespace-pre-wrap">
                  {miResult.narrative}
                </div>
              </CardContent>
            </Card>
          </>
        ) : null}
      </CardContent>
    </Card>
  );
}
