/**
 * RecentJobsCard -- recent production-forecast job history for the ForecastPanel.
 */
import type { Job } from "@/types/jobs";
import { timeAgo, StatusBadge, formatDuration } from "@/components/shared-tuning-utils";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

export function RecentJobsCard({ recentJobs }: { recentJobs: Job[] }) {
  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-sm font-semibold">Recent Forecast Jobs</CardTitle>
      </CardHeader>
      <CardContent className="p-0">
        {recentJobs.length === 0 ? (
          <div className="py-8 text-center text-sm text-muted-foreground">
            No forecast jobs found. Click "Generate Forecast" to start.
          </div>
        ) : (
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="text-xs">Job ID</TableHead>
                <TableHead className="text-xs">Label</TableHead>
                <TableHead className="text-xs">Status</TableHead>
                <TableHead className="text-xs">Submitted</TableHead>
                <TableHead className="text-xs">Duration</TableHead>
                <TableHead className="text-xs">Progress</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {recentJobs.map((job) => (
                <TableRow key={job.job_id}>
                  <TableCell className="text-xs font-mono max-w-[120px] truncate">
                    {job.job_id.slice(0, 8)}
                  </TableCell>
                  <TableCell className="text-xs">
                    {job.job_label || "--"}
                  </TableCell>
                  <TableCell>
                    <StatusBadge status={job.status} />
                  </TableCell>
                  <TableCell className="text-xs text-muted-foreground">
                    {timeAgo(job.submitted_at)}
                  </TableCell>
                  <TableCell className="text-xs text-muted-foreground">
                    {formatDuration(job.started_at, job.completed_at)}
                  </TableCell>
                  <TableCell className="text-xs">
                    {job.status === "running"
                      ? `${job.progress_pct}%`
                      : job.status === "completed"
                        ? "Done"
                        : "--"}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        )}
      </CardContent>
    </Card>
  );
}
