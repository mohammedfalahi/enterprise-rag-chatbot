"use client";

import { FormEvent, useState } from "react";
import { supabase } from "../lib/supabaseClient";
import { BACKEND_URL } from "../lib/api";

const roles = ["employee", "sales_rep", "manager", "executive"] as const;

export default function UserManagement() {
  const [email, setEmail] = useState("");
  const [role, setRole] = useState<(typeof roles)[number]>("employee");
  const [errorMessage, setErrorMessage] = useState("");
  const [successMessage, setSuccessMessage] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);

  const sendInvitation = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setErrorMessage("");
    setSuccessMessage("");
    setIsSubmitting(true);

    try {
      const {
        data: { session },
        error: sessionError,
      } = await supabase.auth.getSession();
      if (sessionError) throw sessionError;
      if (!session) throw new Error("Please sign in again before sending an invitation.");

      const response = await fetch(`${BACKEND_URL}/api/users/invitations`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${session.access_token}`,
        },
        body: JSON.stringify({ email, role }),
      });
      const result = await response.json();
      if (!response.ok) {
        throw new Error(result.detail || "The invitation could not be sent.");
      }

      setEmail("");
      setSuccessMessage(result.message || "Invitation sent.");
    } catch (error) {
      setErrorMessage(
        error instanceof Error ? error.message : "The invitation could not be sent.",
      );
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <main className="flex-1 px-4 py-8 sm:px-6">
      <section className="mx-auto max-w-xl rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
        <h2 className="text-xl font-semibold text-slate-900">User Management</h2>
        <p className="mt-1 text-sm text-slate-600">
          Invite a user to your tenant and assign their role.
        </p>

        <form onSubmit={sendInvitation} className="mt-6 space-y-4">
          <div>
            <label htmlFor="invite-email" className="mb-1 block text-sm font-medium text-slate-700">
              Email address
            </label>
            <input
              id="invite-email"
              type="email"
              value={email}
              onChange={(event) => setEmail(event.target.value)}
              autoComplete="email"
              required
              className="w-full rounded-md border border-slate-300 px-3 py-2 text-sm text-slate-900 focus:border-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-100"
            />
          </div>

          <div>
            <label htmlFor="invite-role" className="mb-1 block text-sm font-medium text-slate-700">
              Role
            </label>
            <select
              id="invite-role"
              value={role}
              onChange={(event) => setRole(event.target.value as (typeof roles)[number])}
              className="w-full rounded-md border border-slate-300 px-3 py-2 text-sm text-slate-900 focus:border-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-100"
            >
              {roles.map((option) => (
                <option key={option} value={option}>
                  {option}
                </option>
              ))}
            </select>
          </div>

          {errorMessage && (
            <p role="alert" className="text-sm text-red-700">
              {errorMessage}
            </p>
          )}
          {successMessage && (
            <p role="status" className="text-sm text-emerald-700">
              {successMessage}
            </p>
          )}

          <button
            type="submit"
            disabled={isSubmitting}
            className="rounded-md bg-indigo-600 px-4 py-2 text-sm font-medium text-white hover:bg-indigo-700 disabled:cursor-not-allowed disabled:opacity-60"
          >
            {isSubmitting ? "Sending..." : "Send Invitation"}
          </button>
        </form>
      </section>
    </main>
  );
}
