import { createContext, useContext, useMemo, useReducer } from "react";

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

interface JobState {
  activeJobs: Map<string, ActiveJob>;
  recentCompletions: CompletedJob[];
}

type JobAction =
  | { type: "START_JOB"; id: string; jobType: string; label: string }
  | { type: "COMPLETE_JOB"; job: CompletedJob }
  | { type: "FAIL_JOB"; id: string }
  | { type: "DISMISS_COMPLETION"; id: string };

function jobReducer(state: JobState, action: JobAction): JobState {
  switch (action.type) {
    case "START_JOB": {
      const next = new Map(state.activeJobs);
      next.set(action.id, { id: action.id, type: action.jobType, label: action.label });
      return { ...state, activeJobs: next };
    }
    case "COMPLETE_JOB": {
      const next = new Map(state.activeJobs);
      next.delete(action.job.id);
      return { activeJobs: next, recentCompletions: [action.job, ...state.recentCompletions] };
    }
    case "FAIL_JOB": {
      const next = new Map(state.activeJobs);
      next.delete(action.id);
      return { ...state, activeJobs: next };
    }
    case "DISMISS_COMPLETION":
      return { ...state, recentCompletions: state.recentCompletions.filter((c) => c.id !== action.id) };
  }
}

const initialState: JobState = { activeJobs: new Map(), recentCompletions: [] };

const JobNotificationContext = createContext<JobNotificationContextValue | null>(null);

export function JobNotificationProvider({ children }: { children: React.ReactNode }) {
  const [state, dispatch] = useReducer(jobReducer, initialState);

  const value = useMemo<JobNotificationContextValue>(() => ({
    activeJobs: state.activeJobs,
    recentCompletions: state.recentCompletions,
    activeJobCount: state.activeJobs.size,
    startJob: (id, type, label) => dispatch({ type: "START_JOB", id, jobType: type, label }),
    completeJob: (job) => dispatch({ type: "COMPLETE_JOB", job }),
    failJob: (id) => dispatch({ type: "FAIL_JOB", id }),
    dismissCompletion: (id) => dispatch({ type: "DISMISS_COMPLETION", id }),
  }), [state]);

  return (
    <JobNotificationContext.Provider value={value}>
      {children}
    </JobNotificationContext.Provider>
  );
}

export function useJobNotification(): JobNotificationContextValue {
  const ctx = useContext(JobNotificationContext);
  if (!ctx) throw new Error("useJobNotification must be used within JobNotificationProvider");
  return ctx;
}
