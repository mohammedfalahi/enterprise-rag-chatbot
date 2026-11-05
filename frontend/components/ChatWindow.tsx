"use client";

import React, { useEffect, useRef, useState, useCallback } from "react";
import { supabase } from "../lib/supabaseClient";
import { BACKEND_URL } from "../lib/api";
import Sidebar from "./Sidebar";
import MarkdownMessage from "./MarkdownMessage";
import type { Citation, ChatMessage, ConversationSummary } from "../lib/types";

interface ChatWindowProps {
  isSidebarOpen?: boolean;
  onCloseSidebar?: () => void;
  roles?: string[];
  initialPrompt?: string | null;
}

function getStarterPrompts(primaryRole: string): string[] {
  switch (primaryRole) {
    case "sales_rep":
      return [
        "What are our enterprise product pricing tiers?",
        "Summarize customer discount approval guidelines",
        "What are the key competitor battlecard points?",
        "How do I submit a custom enterprise contract request?",
      ];
    case "manager":
      return [
        "What is the quarterly performance review workflow?",
        "How do I submit an approval for new team headcount?",
        "Review our department budget authorization limits",
        "What are the promotion and compensation guidelines?",
      ];
    case "executive":
      return [
        "Summarize our strategic company objectives and OKRs",
        "What are our corporate financial governance policies?",
        "Review company-wide compliance and audit protocols",
        "What are our annual growth and expansion targets?",
      ];
    case "employee":
    default:
      return [
        "What is our company remote work and travel policy?",
        "How do I submit an expense reimbursement report?",
        "Where can I find employee healthcare and 401(k) details?",
        "What are the official company holiday and leave rules?",
      ];
  }
}

function formatRoleLabel(role: string): string {
  switch (role) {
    case "executive":
      return "Executive";
    case "manager":
      return "Manager";
    case "sales_rep":
      return "Sales Rep";
    case "employee":
    default:
      return "Employee";
  }
}

export default function ChatWindow({
  isSidebarOpen = true,
  onCloseSidebar = () => {},
  roles = ["employee"],
  initialPrompt = null,
}: ChatWindowProps) {
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [activeConversationId, setActiveConversationId] = useState<string | null>(null);
  const [activeTitle, setActiveTitle] = useState<string>("New Conversation");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState(initialPrompt || "");
  const [loading, setLoading] = useState(false);
  const [loadingHistory, setLoadingHistory] = useState(false);
  const [loadingConversations, setLoadingConversations] = useState(true);
  const [isDeleting, setIsDeleting] = useState<string | null>(null);
  const [isEditingHeaderTitle, setIsEditingHeaderTitle] = useState(false);
  const [headerTitleDraft, setHeaderTitleDraft] = useState("");
  const [copiedIndex, setCopiedIndex] = useState<number | null>(null);
  const [lastFailedMessage, setLastFailedMessage] = useState<string | null>(null);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const streamRef = useRef<{ content: string; citations: Citation[] }>({
    content: "",
    citations: [],
  });
  const primaryRole = roles[0] || "employee";
  const starterPrompts = getStarterPrompts(primaryRole);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, loading]);

  const getAuthToken = async (): Promise<string | undefined> => {
    const {
      data: { session },
    } = await supabase.auth.getSession();
    return session?.access_token;
  };

  const refreshConversations = useCallback(async () => {
    try {
      const token = await getAuthToken();
      if (!token) return;

      const res = await fetch(`${BACKEND_URL}/api/conversations?limit=50`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (res.ok) {
        const data: ConversationSummary[] = await res.json();
        setConversations(data);
      }
    } catch (err) {
      console.error("Failed to fetch conversations:", err);
    }
  }, []);

  useEffect(() => {
    let ignore = false;
    async function initConversations() {
      const {
        data: { session },
      } = await supabase.auth.getSession();
      const token = session?.access_token;
      if (!token) {
        setLoadingConversations(false);
        return;
      }

      try {
        const res = await fetch(`${BACKEND_URL}/api/conversations?limit=50`, {
          headers: { Authorization: `Bearer ${token}` },
        });
        if (res.ok && !ignore) {
          const data: ConversationSummary[] = await res.json();
          setConversations(data);
        }
      } catch (err) {
        console.error("Failed to fetch initial conversations:", err);
      } finally {
        if (!ignore) setLoadingConversations(false);
      }
    }

    initConversations();
    return () => {
      ignore = true;
    };
  }, []);

  const selectConversation = async (conv: ConversationSummary) => {
    if (loading || conv.id === activeConversationId) return;

    setActiveConversationId(conv.id);
    setActiveTitle(conv.title);
    setIsEditingHeaderTitle(false);
    setLoadingHistory(true);
    setLastFailedMessage(null);

    try {
      const token = await getAuthToken();
      const res = await fetch(
        `${BACKEND_URL}/api/conversations/${conv.id}/messages`,
        {
          headers: { Authorization: `Bearer ${token}` },
        }
      );
      if (res.ok) {
        const data: ChatMessage[] = await res.json();
        setMessages(data);
      }
    } catch (err) {
      console.error("Failed to load conversation history:", err);
    } finally {
      setLoadingHistory(false);
    }
  };

  const startNewConversation = () => {
    if (loading) return;
    setActiveConversationId(null);
    setActiveTitle("New Conversation");
    setIsEditingHeaderTitle(false);
    setMessages([]);
    setInput("");
    setLastFailedMessage(null);
  };

  const renameConversationThread = async (convId: string, newTitle: string) => {
    const token = await getAuthToken();
    if (!token) return;

    const res = await fetch(`${BACKEND_URL}/api/conversations/${convId}`, {
      method: "PATCH",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify({ title: newTitle }),
    });

    if (res.ok) {
      const updated: ConversationSummary = await res.json();
      setConversations((prev) =>
        prev.map((c) => (c.id === convId ? { ...c, title: updated.title } : c))
      );
      if (activeConversationId === convId) {
        setActiveTitle(updated.title);
      }
    } else {
      throw new Error(`Failed to rename conversation (status: ${res.status})`);
    }
  };

  const deleteConversationThread = async (
    convId: string,
    e: React.MouseEvent
  ) => {
    e.stopPropagation();
    if (isDeleting) return;

    setIsDeleting(convId);
    try {
      const token = await getAuthToken();
      const res = await fetch(`${BACKEND_URL}/api/conversations/${convId}`, {
        method: "DELETE",
        headers: { Authorization: `Bearer ${token}` },
      });

      if (res.ok) {
        setConversations((prev) => prev.filter((c) => c.id !== convId));
        if (activeConversationId === convId) {
          startNewConversation();
        }
      }
    } catch (err) {
      console.error("Failed to delete conversation:", err);
    } finally {
      setIsDeleting(null);
    }
  };

  const copyMessageContent = async (text: string, index: number) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedIndex(index);
      setTimeout(() => setCopiedIndex(null), 2000);
    } catch (err) {
      console.error("Failed to copy:", err);
    }
  };

  const sendMessage = async (overrideMessage?: string) => {
    const rawInput = overrideMessage !== undefined ? overrideMessage : input;
    const trimmedInput = rawInput.trim();
    if (!trimmedInput || loading) return;

    const userMessage: ChatMessage = { role: "user", content: trimmedInput };
    setMessages((prev) => [...prev, userMessage]);
    if (overrideMessage === undefined) setInput("");
    setLoading(true);
    setLastFailedMessage(null);

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 45000);

    try {
      const token = await getAuthToken();

      const res = await fetch(`${BACKEND_URL}/api/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          message: trimmedInput,
          conversation_id: activeConversationId || null,
        }),
        signal: controller.signal,
      });

      if (!res.ok) throw new Error(`HTTP Error: ${res.status}`);

      const headerConvId = res.headers.get("x-conversation-id");
      if (headerConvId && !activeConversationId) {
        setActiveConversationId(headerConvId);
      }

      if (res.body) {
        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let pendingLine = "";
        streamRef.current = { content: "", citations: [] };

        setMessages((prev) => [
          ...prev,
          { role: "assistant", content: "", citations: [] },
        ]);

        const handleSSELine = (line: string) => {
          if (!line.startsWith("data: ") || line === "data: [DONE]") return;

          let data: Record<string, unknown>;
          try {
            const parsed: unknown = JSON.parse(line.slice(6));
            if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
              throw new Error("SSE payload must be a JSON object.");
            }
            data = parsed as Record<string, unknown>;
          } catch (error) {
            console.error("Failed to parse a complete chat stream event:", error);
            throw new Error("The assistant returned an invalid stream event.");
          }

          if (typeof data.conversation_id === "string" && !activeConversationId) {
            setActiveConversationId(data.conversation_id);
          }
          if (typeof data.title === "string" && activeTitle === "New Conversation") {
            setActiveTitle(data.title);
          }

          if (data.type === "citations" && Array.isArray(data.citations)) {
            streamRef.current = {
              ...streamRef.current,
              citations: data.citations as Citation[],
            };
          } else if (typeof data.text === "string") {
            streamRef.current = {
              ...streamRef.current,
              content: streamRef.current.content + data.text,
            };
          }

          const currentMessage = { ...streamRef.current };
          setMessages((prev) => [
            ...prev.slice(0, -1),
            {
              role: "assistant",
              content: currentMessage.content,
              citations: currentMessage.citations,
            },
          ]);
        };

        while (true) {
          const { done, value } = await reader.read();
          pendingLine += decoder.decode(value, { stream: !done });
          let newlineIndex = pendingLine.indexOf("\n");
          while (newlineIndex !== -1) {
            handleSSELine(pendingLine.slice(0, newlineIndex).replace(/\r$/, ""));
            pendingLine = pendingLine.slice(newlineIndex + 1);
            newlineIndex = pendingLine.indexOf("\n");
          }
          if (done) break;
        }
        if (pendingLine) handleSSELine(pendingLine.replace(/\r$/, ""));
      }

      await refreshConversations();
    } catch (err: unknown) {
      console.error(err);
      setLastFailedMessage(trimmedInput);
      const isAbort = err instanceof Error && err.name === "AbortError";
      const errorText = isAbort
        ? "Request timed out after 45 seconds. Please try again."
        : "Failed to connect to the knowledge assistant. Please check your connection and try again.";
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: errorText },
      ]);
    } finally {
      clearTimeout(timeoutId);
      setLoading(false);
    }
  };

  return (
    <div className="relative flex h-[calc(100dvh-7rem)] min-h-0 flex-1 overflow-hidden md:h-[calc(100dvh-4rem)]">
      {/* Responsive Collapsible Sidebar */}
      <Sidebar
        isOpen={isSidebarOpen}
        onClose={onCloseSidebar}
        conversations={conversations}
        activeConversationId={activeConversationId}
        onSelectConversation={selectConversation}
        onNewConversation={startNewConversation}
        onDeleteConversation={deleteConversationThread}
        onRenameConversation={renameConversationThread}
        isDeletingId={isDeleting}
        loading={loading}
        loadingConversations={loadingConversations}
      />

      {/* Main Chat Panel */}
      <section className="flex min-h-0 min-w-0 flex-1 flex-col justify-between overflow-hidden bg-slate-50">
        {/* Active Thread Header */}
        <header className="flex shrink-0 items-center justify-between border-b border-slate-200 bg-white px-4 py-3 sm:px-6">
          <div className="min-w-0 flex-1 mr-2">
            {isEditingHeaderTitle ? (
              <form
                onSubmit={async (e) => {
                  e.preventDefault();
                  if (!activeConversationId || !headerTitleDraft.trim()) return;
                  try {
                    await renameConversationThread(
                      activeConversationId,
                      headerTitleDraft.trim()
                    );
                    setIsEditingHeaderTitle(false);
                  } catch (err) {
                    console.error(err);
                  }
                }}
                className="flex items-center space-x-2"
              >
                <input
                  type="text"
                  value={headerTitleDraft}
                  onChange={(e) => setHeaderTitleDraft(e.target.value)}
                  autoFocus
                  className="text-sm font-semibold text-slate-800 bg-slate-50 border border-indigo-500 rounded px-2 py-1 focus:outline-none"
                />
                <button
                  type="submit"
                  className="p-1 text-emerald-600 hover:text-emerald-700"
                  title="Save title"
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                  </svg>
                </button>
                <button
                  type="button"
                  onClick={() => setIsEditingHeaderTitle(false)}
                  className="p-1 text-slate-400 hover:text-slate-600"
                  title="Cancel"
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </form>
            ) : (
              <div className="flex items-center space-x-2">
                <h2 className="font-semibold text-slate-800 text-sm sm:text-base truncate">
                  {activeTitle}
                </h2>
                {activeConversationId && (
                  <button
                    onClick={() => {
                      setHeaderTitleDraft(activeTitle);
                      setIsEditingHeaderTitle(true);
                    }}
                    className="p-1 text-slate-400 hover:text-indigo-600 rounded transition"
                    title="Rename thread"
                  >
                    <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.572L16.732 3.732z"
                      />
                    </svg>
                  </button>
                )}
              </div>
            )}
            <p className="text-xs text-slate-500 flex items-center space-x-1.5 mt-0.5">
              <span
                className={`w-1.5 h-1.5 rounded-full ${
                  activeConversationId ? "bg-emerald-500" : "bg-indigo-500"
                }`}
              />
              <span>
                {activeConversationId
                  ? "Multi-turn context thread active"
                  : "New session (context resets)"}
              </span>
            </p>
          </div>
        </header>

        {/* Message Stream */}
        <div className="min-h-0 flex-1 space-y-6 overflow-y-auto px-3 py-5 sm:px-6 sm:py-7">
          {loadingHistory ? (
            <div className="flex flex-col items-center justify-center h-full text-sm text-slate-500 space-y-2">
              <div className="w-5 h-5 border-2 border-indigo-600 border-t-transparent rounded-full animate-spin" />
              <span>Loading conversation history...</span>
            </div>
          ) : messages.length === 0 ? (
            /* Role-Aware Empty State with Starter Prompts */
            <div className="mx-auto flex h-full max-w-3xl flex-col items-center justify-center space-y-8 p-4 text-center">
              <div className="space-y-3">
                <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-2xl border border-indigo-100 bg-indigo-50 text-indigo-700">
                  <svg className="h-6 w-6" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.7} d="M12 3v2m6.364.636-1.414 1.414M21 12h-2m-.636 6.364-1.414-1.414M12 21v-2m-6.364-.636 1.414-1.414M3 12h2m.636-6.364 1.414 1.414M9 9l6 6m0-6-6 6" />
                  </svg>
                </div>
                <h3 className="text-xl font-semibold tracking-tight text-slate-900 sm:text-2xl">
                  Enterprise Knowledge Assistant
                </h3>
                <p className="mx-auto max-w-lg text-sm leading-6 text-slate-600">
                  Answers are grounded in enterprise documentation authorized for your{" "}
                  <span className="font-medium text-slate-700">
                    {formatRoleLabel(primaryRole)}
                  </span>{" "}
                  role. Select a starter prompt below or enter your own query.
                </p>
              </div>

              {/* Starter Prompt Cards */}
              <div className="grid w-full grid-cols-1 gap-3 text-left sm:grid-cols-2">
                {starterPrompts.map((promptText, pIdx) => (
                  <button
                    key={pIdx}
                    onClick={() => sendMessage(promptText)}
                    disabled={loading}
                    className="group flex min-h-16 cursor-pointer items-center justify-between rounded-xl border border-slate-200 bg-white p-4 text-sm font-medium text-slate-700 transition-colors hover:border-indigo-300 hover:bg-indigo-50/40 hover:text-indigo-900"
                  >
                    <span className="truncate pr-2">{promptText}</span>
                    <span className="text-slate-400 group-hover:text-indigo-600 transition-transform group-hover:translate-x-0.5 shrink-0">
                      →
                    </span>
                  </button>
                ))}
              </div>
            </div>
          ) : (
            messages.map((m, i) => (
              <div
                key={i}
                className={`mx-auto flex w-full max-w-4xl gap-3 sm:gap-4 ${
                  m.role === "user" ? "justify-end" : "justify-start"
                }`}
              >
                {/* Assistant Avatar */}
                {m.role === "assistant" && (
                  <div className="mt-1 flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-indigo-700 text-xs font-semibold text-white">
                    EK
                  </div>
                )}

                <div
                  className={`group relative min-w-0 max-w-[92%] rounded-2xl px-4 py-3.5 text-sm leading-6 sm:px-5 sm:py-4 ${
                    m.role === "user"
                      ? "max-w-[88%] rounded-br-sm bg-indigo-700 text-white sm:max-w-[78%]"
                      : "flex-1 border border-slate-200 bg-white text-slate-800"
                  }`}
                >
                  {/* Assistant Message Header Bar with Copy Action */}
                  {m.role === "assistant" && (
                    <div className="mb-3 flex items-center justify-between border-b border-slate-100 pb-2 text-xs text-slate-400">
                      <span className="text-xs font-semibold text-slate-700">
                        Enterprise Assistant
                      </span>
                      <button
                        onClick={() => copyMessageContent(m.content, i)}
                        className="flex items-center space-x-1 rounded-md p-1.5 text-xs transition hover:bg-slate-100 hover:text-slate-700"
                        title="Copy message to clipboard"
                      >
                        {copiedIndex === i ? (
                          <>
                            <svg className="w-3.5 h-3.5 text-emerald-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                            </svg>
                            <span className="text-emerald-600 font-medium">Copied!</span>
                          </>
                        ) : (
                          <>
                            <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
                            </svg>
                            <span>Copy</span>
                          </>
                        )}
                      </button>
                    </div>
                  )}

                  <MarkdownMessage content={m.content} />

                  {/* Rich Sources & Citations */}
                  {m.citations && m.citations.length > 0 && (
                    <div className="mt-5 border-t border-slate-100 pt-4 text-xs text-slate-600">
                      <div className="mb-2.5 flex items-center space-x-2 font-semibold text-slate-700">
                        <svg
                          className="w-3.5 h-3.5 text-indigo-600"
                          fill="none"
                          stroke="currentColor"
                          viewBox="0 0 24 24"
                        >
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth={2}
                            d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253"
                          />
                        </svg>
                        <span>Sources</span>
                      </div>
                      <div className="grid gap-2 sm:grid-cols-2">
                        {m.citations.map((c, cIdx) => {
                          const sourcePath = c.heading_path
                            ?.filter(
                              (part) =>
                                part.trim() &&
                                part.toLocaleLowerCase() !==
                                  c.title?.trim().toLocaleLowerCase(),
                            )
                            .join(" / ");

                          return (
                            <div
                              key={cIdx}
                              className="min-w-0 rounded-lg border border-slate-200 bg-slate-50/70 px-3 py-2.5 text-slate-800"
                            >
                              <div className="flex items-start gap-2">
                                <svg
                                  className="mt-0.5 h-4 w-4 shrink-0 text-indigo-600"
                                  fill="none"
                                  stroke="currentColor"
                                  viewBox="0 0 24 24"
                                  aria-hidden="true"
                                >
                                  <path
                                    strokeLinecap="round"
                                    strokeLinejoin="round"
                                    strokeWidth={1.8}
                                    d="M7 3.75h7l4.25 4.25v12A1.25 1.25 0 0117 21.25H7A1.25 1.25 0 015.75 20V5A1.25 1.25 0 017 3.75zM14 4v4h4M8.5 13h7m-7 3.5h7"
                                  />
                                </svg>
                                <span className="line-clamp-2 min-w-0 font-medium leading-5 text-slate-900">
                                  {c.title || "Source document"}
                                </span>
                              </div>
                              {sourcePath && (
                                <p
                                  className="ml-6 mt-1 truncate text-xs text-slate-500"
                                  title={sourcePath}
                                >
                                  {sourcePath}
                                </p>
                              )}
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  )}
                </div>

                {/* User Avatar */}
                {m.role === "user" && (
                  <div className="mt-1 flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-slate-200 text-[10px] font-semibold text-slate-700">
                    You
                  </div>
                )}
              </div>
            ))
          )}

          {/* Thinking / Streaming Indicator */}
          {loading && (
            <div className="flex items-center space-x-2 text-xs text-slate-500 italic px-2 py-1">
              <span className="w-1.5 h-1.5 bg-indigo-600 rounded-full animate-bounce" />
              <span className="w-1.5 h-1.5 bg-indigo-600 rounded-full animate-bounce [animation-delay:0.2s]" />
              <span className="w-1.5 h-1.5 bg-indigo-600 rounded-full animate-bounce [animation-delay:0.4s]" />
              <span>Retrieving accessible documents & generating response...</span>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Retry Banner on Error */}
        {lastFailedMessage && !loading && (
          <div className="mx-4 sm:mx-6 mb-2 p-2.5 rounded-lg bg-amber-50 border border-amber-200 text-xs text-amber-800 flex items-center justify-between shrink-0 shadow-2xs">
            <div className="flex items-center space-x-2 min-w-0 mr-2">
              <svg className="w-4 h-4 text-amber-600 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
              </svg>
              <span className="truncate">Your previous request failed to reach the server.</span>
            </div>
            <div className="flex items-center space-x-2 shrink-0">
              <button
                onClick={() => {
                  const msg = lastFailedMessage;
                  setLastFailedMessage(null);
                  sendMessage(msg);
                }}
                className="px-2.5 py-1 bg-amber-600 hover:bg-amber-700 text-white rounded font-medium text-[11px] transition shadow-2xs cursor-pointer"
              >
                Retry
              </button>
              <button
                onClick={() => setLastFailedMessage(null)}
                className="text-amber-500 hover:text-amber-800 p-0.5"
                title="Dismiss"
              >
                ✕
              </button>
            </div>
          </div>
        )}

        {/* Input Bar */}
        <div className="shrink-0 border-t border-slate-200 bg-white px-3 py-3 sm:px-6 sm:py-4">
          <form
            className="mx-auto flex max-w-4xl items-end gap-2 rounded-2xl border border-slate-300 bg-slate-50 p-2 transition focus-within:border-indigo-400 focus-within:bg-white focus-within:ring-2 focus-within:ring-indigo-100"
            onSubmit={(event) => {
              event.preventDefault();
              sendMessage();
            }}
          >
            <textarea
              rows={1}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  sendMessage();
                }
              }}
              disabled={loading}
              className="max-h-40 min-h-10 flex-1 resize-y bg-transparent px-3 py-2 text-sm leading-6 text-slate-900 placeholder-slate-400 focus:outline-none disabled:cursor-not-allowed"
              placeholder={
                loading
                  ? "Awaiting assistant response..."
                  : `Ask a question as ${formatRoleLabel(primaryRole)}...`
              }
              aria-label="Message"
            />
            <button
              type="submit"
              disabled={loading || !input.trim()}
              className="shrink-0 rounded-xl bg-indigo-700 px-4 py-2.5 text-sm font-medium text-white transition-colors hover:bg-indigo-800 disabled:cursor-not-allowed disabled:opacity-40"
            >
              Send
            </button>
          </form>
        </div>
      </section>
    </div>
  );
}
