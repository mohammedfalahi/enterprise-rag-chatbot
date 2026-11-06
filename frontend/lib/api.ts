import type { EvaluationRun } from "./types";

export const BACKEND_URL =
  process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

export async function getAuthHeaders(token?: string): Promise<Record<string, string>> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };
  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }
  return headers;
}

export async function fetchEvaluationSummary(token: string) {
  const headers = await getAuthHeaders(token);
  const res = await fetch(`${BACKEND_URL}/api/admin/evaluations/latest`, {
    headers,
  });
  if (!res.ok) {
    throw new Error(`Failed to fetch evaluation summary: ${res.statusText}`);
  }
  return res.json();
}

export async function fetchEvaluationHistory(token: string) {
  const headers = await getAuthHeaders(token);
  const res = await fetch(`${BACKEND_URL}/api/admin/evaluations/history`, {
    headers,
  });
  if (!res.ok) {
    throw new Error(`Failed to fetch evaluation history: ${res.statusText}`);
  }
  return res.json();
}

export async function fetchQueryEvaluations(
  token: string,
  params?: {
    status?: string;
    role?: string;
    category?: string;
    search?: string;
    page?: number;
    limit?: number;
  }
) {
  const headers = await getAuthHeaders(token);
  const query = new URLSearchParams();
  if (params?.status && params.status !== "all") query.set("status", params.status);
  if (params?.role) query.set("role", params.role);
  if (params?.category) query.set("category", params.category);
  if (params?.search) query.set("search", params.search);
  if (params?.page) query.set("page", params.page.toString());
  if (params?.limit) query.set("limit", params.limit.toString());

  const res = await fetch(
    `${BACKEND_URL}/api/admin/evaluations/queries?${query.toString()}`,
    { headers }
  );
  if (!res.ok) {
    throw new Error(`Failed to fetch query evaluations: ${res.statusText}`);
  }
  return res.json();
}

export async function fetchErrorAnalysis(token: string) {
  const headers = await getAuthHeaders(token);
  const res = await fetch(`${BACKEND_URL}/api/admin/evaluations/error-analysis`, {
    headers,
  });
  if (!res.ok) {
    throw new Error(`Failed to fetch error analysis: ${res.statusText}`);
  }
  return res.json();
}

export async function triggerEvaluationRun(
  token: string,
  payload: { sample_size?: number; mode?: string }
): Promise<{ run: EvaluationRun; deduplicated: boolean }> {
  const headers = await getAuthHeaders(token);
  const res = await fetch(`${BACKEND_URL}/api/admin/evaluations/run`, {
    method: "POST",
    headers,
    body: JSON.stringify({
      sample_size: payload.sample_size ?? 10,
      mode: payload.mode ?? "hybrid_rerank",
    }),
  });
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    throw new Error(
      body?.detail || `Failed to trigger evaluation run (${res.status}).`
    );
  }
  return res.json();
}

export async function fetchLatestEvaluationRun(token: string): Promise<EvaluationRun | null> {
  const headers = await getAuthHeaders(token);
  const res = await fetch(`${BACKEND_URL}/api/admin/evaluations/runs/latest`, {
    headers,
    cache: "no-store",
  });
  if (!res.ok) {
    throw new Error(`Failed to fetch latest evaluation run (${res.status}).`);
  }
  const data = await res.json();
  return data.run;
}

export async function fetchEvaluationRun(
  token: string,
  runId: string
): Promise<EvaluationRun> {
  const headers = await getAuthHeaders(token);
  const res = await fetch(
    `${BACKEND_URL}/api/admin/evaluations/runs/${encodeURIComponent(runId)}`,
    { headers, cache: "no-store" }
  );
  if (!res.ok) {
    throw new Error(`Failed to fetch evaluation run (${res.status}).`);
  }
  return res.json();
}
