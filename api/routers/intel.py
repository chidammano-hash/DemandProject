"""Market Intelligence endpoint (feature 18) — web search + LLM narrative briefings."""
from __future__ import annotations

import logging
import os

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from api.core import get_conn, get_openai
from api.auth import require_api_key

logger = logging.getLogger(__name__)

router = APIRouter(tags=["market-intelligence"])


class MarketIntelRequest(BaseModel):
    item_no: str
    location_id: str


@router.post("/market-intelligence", dependencies=[Depends(require_api_key)])
def market_intelligence(req: MarketIntelRequest):
    """Generate an AI-powered market briefing for a product at a location."""
    item_no = req.item_no.strip()
    location_id = req.location_id.strip()
    if not item_no or not location_id:
        raise HTTPException(422, "Both item_no and location_id are required")

    # 1. Look up item metadata
    item_desc = brand_name = category = None
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            'SELECT item_desc, brand_name, category FROM dim_item WHERE item_no = %s LIMIT 1',
            [item_no],
        )
        row = cur.fetchone()
        if row:
            item_desc = row[0]
            brand_name = row[1]
            category = row[2]

    # 2. Look up location metadata
    state_id = site_desc = None
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT state_id, site_desc FROM dim_location WHERE location_id = %s LIMIT 1",
            [location_id],
        )
        row = cur.fetchone()
        if row:
            state_id = row[0]
            site_desc = row[1]

    # 3. Gather recent sales context
    sales_context = ""
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """SELECT month_start, SUM(qty)::double precision AS qty
               FROM agg_sales_monthly
               WHERE dmdunit = %s AND loc = %s
               GROUP BY 1 ORDER BY 1 DESC LIMIT 12""",
            [item_no, location_id],
        )
        rows = cur.fetchall()
        if rows:
            sales_lines = [f"  {r[0]}: {r[1]:.0f} units" for r in rows]
            sales_context = "Recent 12-month sales (newest first):\n" + "\n".join(sales_lines)

    # 4. Build product context for the AI
    product_parts = [f"Item: {item_no}"]
    if item_desc:
        product_parts.append(f"Description: {item_desc}")
    if brand_name:
        product_parts.append(f"Brand: {brand_name}")
    if category:
        product_parts.append(f"Category: {category}")
    product_parts.append(f"Location: {location_id}")
    if site_desc:
        product_parts.append(f"Site: {site_desc}")
    if state_id:
        product_parts.append(f"State: {state_id}")
    product_context = "\n".join(product_parts)

    # 5. Web search (Google Custom Search if configured, else OpenAI web search)
    search_results: list[dict[str, str]] = []
    google_api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    google_cx = os.getenv("GOOGLE_CX", "").strip()

    search_query_parts = []
    if brand_name:
        search_query_parts.append(brand_name)
    if item_desc:
        search_query_parts.append(item_desc)
    elif category:
        search_query_parts.append(category)
    search_query_parts.append("market trends demand forecast")
    if state_id:
        search_query_parts.append(state_id)
    search_query = " ".join(search_query_parts)

    if google_api_key and google_cx:
        import urllib.request
        import urllib.parse
        try:
            qs = urllib.parse.urlencode({"key": google_api_key, "cx": google_cx, "q": search_query, "num": 5})
            google_url = f"https://www.googleapis.com/customsearch/v1?{qs}"
            req_obj = urllib.request.Request(google_url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req_obj, timeout=10) as resp:
                import json as _json
                data = _json.loads(resp.read())
                for hit in data.get("items", [])[:5]:
                    search_results.append({
                        "title": hit.get("title", ""),
                        "link": hit.get("link", ""),
                        "snippet": hit.get("snippet", ""),
                    })
        except Exception:
            logger.warning("Google Custom Search failed for query %r", search_query, exc_info=True)

    # 6. Generate narrative with OpenAI
    client = get_openai()
    narrative = ""
    search_context = ""
    if search_results:
        search_lines = [f"- {sr['title']}: {sr['snippet']}" for sr in search_results]
        search_context = "\nWeb Search Results:\n" + "\n".join(search_lines)

    prompt = f"""You are a market intelligence analyst for a demand planning team.

Product Context:
{product_context}

{sales_context}
{search_context}

Generate a concise market intelligence briefing (3-5 paragraphs) covering:
1. Product/brand overview and market positioning
2. Recent market trends affecting this product category
3. Regional demand factors for the location/state
4. Competitive landscape and potential demand drivers
5. Forward-looking demand signals and risks

Be specific and actionable. Focus on factors that would help a planner or analyst make better decisions."""

    use_web_search = not search_results and not google_api_key
    if use_web_search:
        try:
            response = client.responses.create(
                model="gpt-4o-mini",
                input=prompt,
                tools=[{"type": "web_search_preview"}],
            )
            narrative = response.output_text or ""
            for out_item in (response.output or []):
                if getattr(out_item, "type", None) == "message":
                    for content in getattr(out_item, "content", []):
                        for ann in getattr(content, "annotations", []):
                            if getattr(ann, "type", None) == "url_citation":
                                search_results.append({
                                    "title": getattr(ann, "title", "") or ann.url,
                                    "link": ann.url,
                                    "snippet": getattr(ann, "title", ""),
                                })
        except Exception:
            logger.warning("OpenAI web search fallback failed", exc_info=True)
            use_web_search = False

    if not narrative:
        try:
            chat_resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a market intelligence analyst for demand planning."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=1500,
                temperature=0.7,
            )
            narrative = chat_resp.choices[0].message.content or ""
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"AI generation failed: {e}")

    from datetime import datetime, timezone
    return {
        "item_no": item_no,
        "location_id": location_id,
        "item_desc": item_desc,
        "brand_name": brand_name,
        "category": category,
        "state_id": state_id,
        "site_desc": site_desc,
        "search_results": search_results,
        "narrative": narrative,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
