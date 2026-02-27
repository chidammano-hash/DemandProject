import { createContext, useCallback, useContext, useState } from "react";

export interface CompletedJob {
  id: string;
  type: string;
  label: string;
  runtimeSeconds: number;
  status: "completed" | "failed";
}

export interface ActiveJob {
  id: string;
  type: string;
  label: string;
}

export interface JobNotificationContextValue {
  activeJobs: Map<string, ActiveJob>;
  recentCompletions: CompletedJob[];
  activeJobCount: number;
  startJob: (id: string, type: string, label: string) => void;
  completeJob: (job: CompletedJob) => void;
  failJob: (id: string) => void;
  dismissCompletion: (id: string) => void;
}

const JobNotificationContext = createContext<JobNotificationContextValue | null>(null);

export function JobNotificationProvider({ children }: { children: React.ReactNode }) {
  const [activeJobs, setActiveJobs] = useState<Map<string, ActiveJob>>(new Map());
  const [recentCompletions, setRecentCompletions] = useState<CompletedJob[]>([]);

  const startJob = useCallback((id: string, type: string, label: string) => {
    setActiveJobs((prev) => {
      const next = new Map(prev);
      next.set(id, { id, type, label });
      return next;
    });
  }, []);

  const completeJob = useCallback((job: CompletedJob) => {
    setActiveJobs((prev) => {
      const next = new Map(prev);
      next.delete(job.id);
      return next;
    });
    setRecentCompletions((prev) => [job, ...prev]);
  }, []);

  const failJob = useCallback((id: string) => {
    setActiveJobs((prev) => {
      const next = new Map(prev);
      next.delete(id);
      return next;
    });
  }, []);

  const dismissCompletion = useCallback((id: string) => {
    setRecentCompletions((prev) => prev.filter((c) => c.id !== id));
  }, []);

  return (
    <JobNotificationContext.Provider
      value={{
        activeJobs,
        recentCompletions,
        activeJobCount: activeJobs.size,
        startJob,
        completeJob,
        failJob,
        dismissCompletion,
      }}
    >
      {children}
    </JobNotificationContext.Provider>
  );
}

export function useJobNotification(): JobNotificationContextValue {
  const ctx = useContext(JobNotificationContext);
  if (!ctx) throw new Error("useJobNotification must be used within JobNotificationProvider");
  return ctx;
}
