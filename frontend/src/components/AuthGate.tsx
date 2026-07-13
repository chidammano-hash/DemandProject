import { useState, type FormEvent, type ReactNode } from "react";
import { Loader2, LockKeyhole } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { useAuth } from "@/context/AuthContext";
import { formatApiError } from "@/lib/formatApiError";

export function AuthGate({ children }: { children: ReactNode }): JSX.Element {
  const { user, loading, login } = useAuth();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  if (loading) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-background text-muted-foreground">
        <Loader2 className="mr-2 h-5 w-5 animate-spin" /> Checking session…
      </div>
    );
  }
  if (user) return <>{children}</>;

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    setSubmitting(true);
    setError(null);
    try {
      await login({ email, password });
    } catch (reason) {
      setError(formatApiError(reason));
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <main className="flex min-h-screen items-center justify-center bg-muted/30 px-4">
      <form onSubmit={submit} className="w-full max-w-sm rounded-xl border bg-card p-6 shadow-card">
        <div className="mb-6 flex items-start gap-3">
          <span className="rounded-lg bg-primary/10 p-2 text-primary"><LockKeyhole className="h-5 w-5" /></span>
          <div>
            <h1 className="text-lg font-semibold">Sign in to Demand Studio</h1>
            <p className="mt-1 text-sm text-muted-foreground">Use your planning account to run forecasting workflows.</p>
          </div>
        </div>
        <label className="mb-4 block text-sm font-medium">
          Email
          <Input className="mt-1.5" type="email" autoComplete="username" value={email} onChange={(event) => setEmail(event.target.value)} required autoFocus />
        </label>
        <label className="mb-4 block text-sm font-medium">
          Password
          <Input className="mt-1.5" type="password" autoComplete="current-password" value={password} onChange={(event) => setPassword(event.target.value)} required />
        </label>
        {error && <p role="alert" className="mb-4 rounded-md bg-destructive/10 px-3 py-2 text-sm text-destructive">{error}</p>}
        <Button className="w-full" type="submit" disabled={submitting}>
          {submitting && <Loader2 className="h-4 w-4 animate-spin" />}
          {submitting ? "Signing in…" : "Sign in"}
        </Button>
      </form>
    </main>
  );
}
