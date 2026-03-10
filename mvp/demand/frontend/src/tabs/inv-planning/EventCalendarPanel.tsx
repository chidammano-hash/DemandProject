/**
 * EventCalendarPanel — F4.3 Event & Promotion Uplift Planning
 * Shows event calendar with uplift preview and approval status.
 */

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Calendar, CalendarDays, Plus, CheckCircle, Clock } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { EmptyState } from "@/components/EmptyState";
import {
  eventKeys,
  fetchEventCalendar,
  STALE_EVO,
  type CalendarEvent,
} from "@/api/queries/evolution";

const STATUS_COLORS: Record<string, string> = {
  approved: "bg-green-100 text-green-700",
  pending: "bg-amber-100 text-amber-700",
  draft: "bg-gray-100 text-gray-600",
  rejected: "bg-red-100 text-red-700",
};

const EVENT_TYPE_COLORS: Record<string, string> = {
  promotion: "bg-cyan-100 text-cyan-700",
  holiday: "bg-blue-100 text-blue-700",
  launch: "bg-teal-100 text-teal-700",
  clearance: "bg-orange-100 text-orange-700",
};

function EventRow({ ev }: { ev: CalendarEvent }) {
  return (
    <tr className="border-t hover:bg-muted/30 transition-colors">
      <td className="px-3 py-2 font-medium text-sm">{ev.event_name}</td>
      <td className="px-3 py-2">
        <span className={`text-xs px-1.5 py-0.5 rounded ${EVENT_TYPE_COLORS[ev.event_type] ?? "bg-gray-100 text-gray-600"}`}>
          {ev.event_type}
        </span>
      </td>
      <td className="px-3 py-2 text-xs">{ev.start_date}</td>
      <td className="px-3 py-2 text-xs">{ev.end_date}</td>
      <td className="px-3 py-2 text-xs font-mono">{ev.item_no ?? "All"}</td>
      <td className="px-3 py-2 text-xs">{ev.loc ?? "All"}</td>
      <td className="px-3 py-2 text-xs">
        {ev.is_hard_override ? (
          <span className="font-medium text-red-600">Override: {ev.override_qty?.toFixed(0) ?? "—"}</span>
        ) : (
          <span>{ev.uplift_multiplier.toFixed(2)}× {ev.additive_qty > 0 ? `+${ev.additive_qty.toFixed(0)}` : ""}</span>
        )}
      </td>
      <td className="px-3 py-2">
        <span className={`text-xs px-1.5 py-0.5 rounded ${STATUS_COLORS[ev.status] ?? "bg-gray-100 text-gray-600"}`}>
          {ev.status}
        </span>
      </td>
    </tr>
  );
}

export function EventCalendarPanel() {
  const now = new Date();
  const [year, setYear] = useState(now.getFullYear());
  const [month, setMonth] = useState(now.getMonth() + 1);
  const [eventType, setEventType] = useState("");
  const [status, setStatus] = useState("");

  const params = {
    year,
    month,
    event_type: eventType || undefined,
    status: status || undefined,
  };

  const { data, isLoading } = useQuery({
    queryKey: eventKeys.calendar(params),
    queryFn: () => fetchEventCalendar(params),
    staleTime: STALE_EVO.ONE_MIN,
  });

  const events = data?.events ?? [];
  const total = data?.total ?? 0;

  const approvedCount = events.filter((e) => e.status === "approved").length;
  const pendingCount = events.filter((e) => e.status === "pending").length;

  const prevMonth = () => {
    if (month === 1) { setYear(y => y - 1); setMonth(12); }
    else setMonth(m => m - 1);
  };
  const nextMonth = () => {
    if (month === 12) { setYear(y => y + 1); setMonth(1); }
    else setMonth(m => m + 1);
  };

  const monthLabel = new Date(year, month - 1).toLocaleString("default", { month: "long", year: "numeric" });

  return (
    <div className="space-y-6 p-4">
      {/* Header with month navigation */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <button onClick={prevMonth} className="px-2 py-1 border rounded text-sm">←</button>
          <h2 className="text-lg font-semibold flex items-center gap-2">
            <Calendar size={18} /> {monthLabel}
          </h2>
          <button onClick={nextMonth} className="px-2 py-1 border rounded text-sm">→</button>
        </div>
        <button className="flex items-center gap-1 px-3 py-1.5 bg-primary text-primary-foreground rounded text-sm font-medium">
          <Plus size={14} /> New Event
        </button>
      </div>

      {/* KPI summary */}
      <div className="grid grid-cols-3 gap-4">
        {[
          { label: "Total Events", value: total.toString(), icon: <Calendar size={16} /> },
          { label: "Approved", value: approvedCount.toString(), icon: <CheckCircle size={16} />, green: true },
          { label: "Pending Approval", value: pendingCount.toString(), icon: <Clock size={16} />, warn: pendingCount > 0 },
        ].map((c) => (
          <Card key={c.label} className={c.warn ? "border-amber-400" : ""}>
            <CardHeader className="pb-2 flex flex-row items-center justify-between">
              <CardTitle className="text-sm font-medium text-muted-foreground">{c.label}</CardTitle>
              <span className={c.green ? "text-green-500" : c.warn ? "text-amber-500" : "text-muted-foreground"}>{c.icon}</span>
            </CardHeader>
            <CardContent>
              <p className="text-2xl font-bold">{c.value}</p>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Filter row */}
      <div className="flex flex-wrap gap-3 items-end">
        <div>
          <label className="text-xs font-medium block mb-1">Event Type</label>
          <select className="border rounded px-2 py-1 text-sm" value={eventType} onChange={(e) => setEventType(e.target.value)}>
            <option value="">All Types</option>
            <option value="promotion">Promotion</option>
            <option value="holiday">Holiday</option>
            <option value="launch">Launch</option>
            <option value="clearance">Clearance</option>
          </select>
        </div>
        <div>
          <label className="text-xs font-medium block mb-1">Status</label>
          <select className="border rounded px-2 py-1 text-sm" value={status} onChange={(e) => setStatus(e.target.value)}>
            <option value="">All Statuses</option>
            <option value="draft">Draft</option>
            <option value="pending">Pending</option>
            <option value="approved">Approved</option>
            <option value="rejected">Rejected</option>
          </select>
        </div>
      </div>

      {/* Events table */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Events for {monthLabel}</CardTitle>
        </CardHeader>
        <CardContent className="p-0">
          {isLoading ? (
            <p className="p-4 text-sm text-muted-foreground">Loading…</p>
          ) : events?.length === 0 ? (
            <div className="p-6">
              <EmptyState
                icon={CalendarDays}
                title="No events or promotions planned"
                description="The event calendar records promotions, seasonal uplifts, product launches, and clearance events that override the baseline statistical forecast. Each event requires approval before it flows into the consensus demand plan."
                steps={[
                  { label: "Click 'New Event' above to create the first event", command: "(no CLI command — events are created in the UI)" },
                ]}
              />
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="bg-muted/50 text-xs uppercase text-muted-foreground">
                  <tr>
                    {["Event Name", "Type", "Start", "End", "Item", "Loc", "Uplift", "Status"].map((h) => (
                      <th key={h} className="px-3 py-2 text-left font-medium">{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {events.map((ev) => <EventRow key={ev.event_id} ev={ev} />)}
                </tbody>
              </table>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
