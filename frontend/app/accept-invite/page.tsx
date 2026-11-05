"use client";

import { FormEvent, useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { BACKEND_URL } from "../../lib/api";
import { supabase } from "../../lib/supabaseClient";

export default function AcceptInvite() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [isCheckingSession, setIsCheckingSession] = useState(true);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");
  const router = useRouter();

  useEffect(() => {
    supabase.auth.getSession().then(({ data: { session }, error }) => {
      if (error) setErrorMessage(error.message);
      setEmail(session?.user.email ?? "");
      setIsCheckingSession(false);
    });
  }, []);

  const acceptInvitation = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setErrorMessage("");
    setIsSubmitting(true);

    try {
      const {
        data: { session },
        error: sessionError,
      } = await supabase.auth.getSession();
      if (sessionError) throw sessionError;
      if (!session) throw new Error("Open this page from the invitation email.");

      const { error: passwordError } = await supabase.auth.updateUser({ password });
      if (passwordError) throw passwordError;

      const {
        data: { session: updatedSession },
        error: updatedSessionError,
      } = await supabase.auth.getSession();
      if (updatedSessionError) throw updatedSessionError;
      if (!updatedSession) throw new Error("Your session expired. Please reopen the invitation.");

      const response = await fetch(`${BACKEND_URL}/api/users/accept-invitation`, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${updatedSession.access_token}`,
        },
      });
      const result = await response.json();
      if (!response.ok) {
        throw new Error(result.detail || "The invitation could not be accepted.");
      }

      router.push("/");
    } catch (error) {
      setErrorMessage(
        error instanceof Error ? error.message : "The invitation could not be accepted.",
      );
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <main className="flex min-h-screen items-center justify-center bg-slate-50 px-4 text-slate-900">
      <form
        onSubmit={acceptInvitation}
        className="w-full max-w-md space-y-4 rounded-xl border border-slate-200 bg-white p-8 shadow-sm"
      >
        <h1 className="text-2xl font-semibold">Accept your invitation</h1>
        {isCheckingSession ? (
          <p className="text-sm text-slate-600">Verifying invitation...</p>
        ) : (
          <>
            <p className="text-sm text-slate-600">
              {email ? `Set a password for ${email}.` : "Set a password to finish joining."}
            </p>
            <div>
              <label htmlFor="invite-password" className="mb-1 block text-sm font-medium">
                Password
              </label>
              <input
                id="invite-password"
                type="password"
                value={password}
                onChange={(event) => setPassword(event.target.value)}
                minLength={6}
                autoComplete="new-password"
                required
                className="w-full rounded-md border border-slate-300 px-3 py-2 text-sm focus:border-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-100"
              />
            </div>
            {errorMessage && (
              <p role="alert" className="text-sm text-red-700">
                {errorMessage}
              </p>
            )}
            <button
              type="submit"
              disabled={isSubmitting}
              className="w-full rounded-md bg-indigo-600 py-2 text-sm font-medium text-white hover:bg-indigo-700 disabled:cursor-not-allowed disabled:opacity-60"
            >
              {isSubmitting ? "Accepting..." : "Set Password and Continue"}
            </button>
          </>
        )}
      </form>
    </main>
  );
}
