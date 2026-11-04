"use client";

import React, { useEffect, useRef, useState, useCallback } from "react";
import { supabase } from "../lib/supabaseClient";
import { BACKEND_URL } from "../lib/api";
import Sidebar from "./Sidebar";
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
        streamRef.current = { content: "", citations: [] };

        setMessages((prev) => [
          ...prev,
          { role: "assistant", content: "", citations: [] },
        ]);

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          const chunk = decoder.decode(value);
          const lines = chunk.split("\n");

          for (const line of lines) {
            if (line.startsWith("data: ") && line !== "data: [DONE]") {
              try {
                const data = JSON.parse(line.slice(6));

                if (data.conversation_id && !activeConversationId) {
                  setActiveConversationId(data.conversation_id);
                }
                if (data.title && activeTitle === "New Conversation") {
                  setActiveTitle(data.title);
                }

                if (data.type === "citations" && Array.isArray(data.citations)) {
                  streamRef.current = {
                    ...streamRef.current,
                    citations: data.citations,
                  };
                  const currentMsg = { ...streamRef.current };
                  setMessages((prev) => [
                    ...prev.slice(0, -1),
                    {
                      role: "assistant",
                      content: currentMsg.content,
                      citations: currentMsg.citations,
                    },
                  ]);
                } else if (data.text) {
                  streamRef.current = {
                    ...streamRef.current,
                    content: streamRef.current.content + data.text,
                  };
                  const currentMsg = { ...streamRef.current };
                  setMessages((prev) => [
                    ...prev.slice(0, -1),
                    {
                      role: "assistant",
                      content: currentMsg.content,
                      citations: currentMsg.citations,
                    },
                  ]);
                }
              } catch {
                // Ignore incomplete SSE chunks
              }
            }
          }
        }
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
    <div className="flex flex-1 h-[calc(100vh-4rem)] overflow-hidden relative">
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
      <section className="flex-1 flex flex-col justify-between bg-slate-50 min-w-0 overflow-hidden">
        {/* Active Thread Header */}
        <header className="px-4 sm:px-6 py-3 border-b border-slate-200/80 bg-white flex items-center justify-between shrink-0 shadow-2xs">
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
        <div className="flex-1 overflow-y-auto p-4 sm:p-6 space-y-5">
          {loadingHistory ? (
            <div className="flex flex-col items-center justify-center h-full text-sm text-slate-500 space-y-2">
              <div className="w-5 h-5 border-2 border-indigo-600 border-t-transparent rounded-full animate-spin" />
              <span>Loading conversation history...</span>
            </div>
          ) : messages.length === 0 ? (
            /* Role-Aware Empty State with Starter Prompts */
            <div className="flex flex-col items-center justify-center h-full max-w-2xl mx-auto p-4 text-center space-y-6">
              <div className="space-y-2">
                <div className="w-12 h-12 rounded-2xl bg-indigo-50 border border-indigo-100 flex items-center justify-center text-indigo-600 text-xl shadow-2xs mx-auto">
                  💡
                </div>
                <h3 className="font-semibold text-slate-800 text-lg sm:text-xl">
                  Enterprise Knowledge Assistant
                </h3>
                <p className="text-xs text-slate-500 max-w-md mx-auto leading-relaxed">
                  Answers are grounded in enterprise documentation authorized for your{" "}
                  <span className="font-medium text-slate-700">
                    {formatRoleLabel(primaryRole)}
                  </span>{" "}
                  role. Select a starter prompt below or enter your own query.
                </p>
              </div>

              {/* Starter Prompt Cards */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5 w-full text-left">
                {starterPrompts.map((promptText, pIdx) => (
                  <button
                    key={pIdx}
                    onClick={() => sendMessage(promptText)}
                    disabled={loading}
                    className="p-3 rounded-xl bg-white hover:bg-slate-50 border border-slate-200/80 hover:border-indigo-300 text-slate-700 hover:text-indigo-900 transition-all text-xs font-medium flex items-center justify-between group shadow-2xs cursor-pointer"
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
                className={`flex gap-3 ${
                  m.role === "user" ? "justify-end" : "justify-start"
                }`}
              >
                {/* Assistant Avatar */}
                {m.role === "assistant" && (
                  <div className="w-8 h-8 rounded-lg bg-indigo-600 text-white flex items-center justify-center font-bold text-xs shrink-0 shadow-2xs mt-1">
                    EK
                  </div>
                )}

                <div
                  className={`group relative p-4 rounded-2xl text-sm leading-relaxed ${
                    m.role === "user"
                      ? "bg-indigo-600 text-white max-w-[85%] sm:max-w-[75%] rounded-br-xs shadow-xs"
                      : "bg-white text-slate-900 max-w-[90%] sm:max-w-[85%] rounded-bl-xs border border-slate-200/80 shadow-xs"
                  }`}
                >
                  {/* Assistant Message Header Bar with Copy Action */}
                  {m.role === "assistant" && (
                    <div className="flex items-center justify-between mb-2 pb-1.5 border-b border-slate-100 text-xs text-slate-400">
                      <span className="font-medium text-slate-600 text-[11px]">
                        Enterprise Assistant
                      </span>
                      <button
                        onClick={() => copyMessageContent(m.content, i)}
                        className="flex items-center space-x-1 hover:text-slate-700 p-1 rounded hover:bg-slate-100 transition text-[11px]"
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

                  <div className="whitespace-pre-wrap">{m.content}</div>

                  {/* Rich Sources & Citations */}
                  {m.citations && m.citations.length > 0 && (
                    <div className="mt-3.5 pt-2.5 border-t border-slate-200 text-xs text-slate-600">
                      <div className="font-semibold text-slate-700 mb-2 flex items-center space-x-1.5">
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
                        <span>Verified Citations ({m.citations.length}):</span>
                      </div>
                      <div className="space-y-2">
                        {m.citations.map((c, cIdx) => (
                          <div
                            key={cIdx}
                            className="bg-slate-50 p-2.5 rounded-lg border border-slate-200/90 text-slate-800 space-y-1 hover:border-slate-300 transition"
                          >
                            <div className="flex items-center justify-between gap-2">
                              <span className="font-semibold text-slate-900 text-xs truncate">
                                📄 {c.title}
                              </span>
                              <span className="inline-block px-1.5 py-0.5 bg-indigo-50 text-indigo-700 rounded text-[10px] font-mono border border-indigo-100 shrink-0">
                                score: {c.score}
                              </span>
                            </div>
                            {c.heading_path && c.heading_path.length > 0 && (
                              <div className="text-[11px] text-slate-500 font-medium">
                                Path: {c.heading_path.join(" → ")}
                              </div>
                            )}
                            {c.chunk_id && (
                              <div className="text-[10px] text-slate-400 font-mono">
                                chunk: {c.chunk_id.slice(0, 12)}...
                              </div>
                            )}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>

                {/* User Avatar */}
                {m.role === "user" && (
                  <div className="w-8 h-8 rounded-lg bg-slate-800 text-white flex items-center justify-center font-bold text-xs shrink-0 shadow-2xs mt-1">
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
        <div className="p-3 sm:p-4 border-t border-slate-200/80 bg-white shrink-0">
          <div className="max-w-4xl mx-auto flex items-center gap-2">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) =>
                e.key === "Enter" && !e.shiftKey && sendMessage()
              }
              disabled={loading}
              className="flex-1 bg-slate-50 border border-slate-300 rounded-xl px-4 py-2.5 text-sm text-slate-900 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:bg-white focus:border-transparent transition disabled:bg-slate-100"
              placeholder={
                loading
                  ? "Awaiting assistant response..."
                  : `Ask a question as ${formatRoleLabel(primaryRole)}...`
              }
            />
            <button
              onClick={() => sendMessage()}
              disabled={loading || !input.trim()}
              className="bg-indigo-600 hover:bg-indigo-500 text-white px-5 py-2.5 rounded-xl text-sm font-medium transition disabled:opacity-40 disabled:cursor-not-allowed shadow-xs shrink-0 cursor-pointer"
            >
              Send
            </button>
          </div>
        </div>
      </section>
    </div>
  );
}
