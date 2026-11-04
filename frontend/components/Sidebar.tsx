"use client";

import React, { useState, useMemo } from "react";
import type { ConversationSummary } from "../lib/types";

interface SidebarProps {
  isOpen: boolean;
  onClose: () => void;
  conversations: ConversationSummary[];
  activeConversationId: string | null;
  onSelectConversation: (conv: ConversationSummary) => void;
  onNewConversation: () => void;
  onDeleteConversation: (convId: string, e: React.MouseEvent) => void;
  onRenameConversation: (convId: string, newTitle: string) => Promise<void>;
  isDeletingId?: string | null;
  loading?: boolean;
  loadingConversations?: boolean;
}

function groupConversations(
  conversations: ConversationSummary[]
): Record<string, ConversationSummary[]> {
  const groups: Record<string, ConversationSummary[]> = {
    Today: [],
    Yesterday: [],
    "Previous 7 Days": [],
    Older: [],
  };

  const now = new Date();
  const startOfToday = new Date(
    now.getFullYear(),
    now.getMonth(),
    now.getDate()
  ).getTime();
  const startOfYesterday = startOfToday - 86400000;
  const startOf7DaysAgo = startOfToday - 86400000 * 7;

  for (const conv of conversations) {
    const time = new Date(conv.updated_at).getTime();
    if (isNaN(time)) {
      groups.Older.push(conv);
    } else if (time >= startOfToday) {
      groups.Today.push(conv);
    } else if (time >= startOfYesterday) {
      groups.Yesterday.push(conv);
    } else if (time >= startOf7DaysAgo) {
      groups["Previous 7 Days"].push(conv);
    } else {
      groups.Older.push(conv);
    }
  }

  return groups;
}

export default function Sidebar({
  isOpen,
  onClose,
  conversations,
  activeConversationId,
  onSelectConversation,
  onNewConversation,
  onDeleteConversation,
  onRenameConversation,
  isDeletingId,
  loading,
  loadingConversations = false,
}: SidebarProps) {
  const [searchQuery, setSearchQuery] = useState("");
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editTitle, setEditTitle] = useState("");
  const [isSavingRename, setIsSavingRename] = useState(false);
  const [confirmDeleteId, setConfirmDeleteId] = useState<string | null>(null);

  // Filter conversations based on search
  const filteredConversations = useMemo(() => {
    const trimmed = searchQuery.trim().toLowerCase();
    if (!trimmed) return conversations;
    return conversations.filter((c) =>
      (c.title || "Untitled Conversation").toLowerCase().includes(trimmed)
    );
  }, [conversations, searchQuery]);

  // Group chronologically if not searching
  const grouped = useMemo(() => {
    return groupConversations(filteredConversations);
  }, [filteredConversations]);

  const handleStartRename = (conv: ConversationSummary, e: React.MouseEvent) => {
    e.stopPropagation();
    setEditingId(conv.id);
    setEditTitle(conv.title || "Untitled Conversation");
  };

  const handleSaveRename = async (convId: string, e: React.FormEvent | React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    const clean = editTitle.trim();
    if (!clean || isSavingRename) return;

    setIsSavingRename(true);
    try {
      await onRenameConversation(convId, clean);
      setEditingId(null);
    } catch (err) {
      console.error("Failed to rename conversation:", err);
    } finally {
      setIsSavingRename(false);
    }
  };

  const handleCancelRename = (e: React.MouseEvent) => {
    e.stopPropagation();
    setEditingId(null);
    setEditTitle("");
  };

  const handleDeleteClick = (convId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (confirmDeleteId === convId) {
      onDeleteConversation(convId, e);
      setConfirmDeleteId(null);
    } else {
      setConfirmDeleteId(convId);
    }
  };

  return (
    <>
      {/* Mobile Drawer Backdrop */}
      {isOpen && (
        <div
          onClick={onClose}
          aria-hidden="true"
          className="fixed inset-0 bg-slate-900/50 backdrop-blur-xs z-30 lg:hidden transition-opacity"
        />
      )}

      {/* Responsive Sidebar Container */}
      <aside
        className={`fixed lg:static top-0 bottom-0 left-0 z-40 bg-slate-900 text-slate-200 flex flex-col border-r border-slate-800 transition-all duration-300 ease-in-out shrink-0 ${
          isOpen
            ? "w-72 translate-x-0"
            : "-translate-x-full lg:translate-x-0 lg:w-0 lg:opacity-0 lg:overflow-hidden pointer-events-none lg:pointer-events-none"
        }`}
      >
        {/* Mobile Header in Drawer */}
        <div className="h-16 px-4 flex items-center justify-between border-b border-slate-800 lg:hidden">
          <div className="flex items-center space-x-2.5">
            <div className="w-8 h-8 rounded-lg bg-indigo-600 text-white flex items-center justify-center font-bold text-sm shadow-xs">
              EK
            </div>
            <span className="font-semibold text-sm text-white">
              Enterprise Knowledge
            </span>
          </div>
          <button
            onClick={onClose}
            aria-label="Close sidebar"
            className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Action Button: New Chat */}
        <div className="p-3 border-b border-slate-800/80 space-y-2">
          <button
            onClick={() => {
              onNewConversation();
              onClose();
            }}
            disabled={loading}
            className="w-full flex items-center justify-center space-x-2 px-3 py-2.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white text-sm font-medium transition-colors shadow-xs"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
            </svg>
            <span>New Conversation</span>
          </button>

          {/* Search Filter Input */}
          <div className="relative">
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search conversations..."
              className="w-full bg-slate-800/80 text-xs text-slate-200 placeholder-slate-400 pl-8 pr-7 py-1.5 rounded-md border border-slate-700/80 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition"
            />
            <svg
              className="w-3.5 h-3.5 text-slate-400 absolute left-2.5 top-2.5"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
              />
            </svg>
            {searchQuery && (
              <button
                onClick={() => setSearchQuery("")}
                className="absolute right-2 top-2 text-slate-400 hover:text-white p-0.5 rounded"
                title="Clear search"
              >
                <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            )}
          </div>
        </div>

        {/* Conversation List / History Area */}
        <div className="flex-1 overflow-y-auto p-2 space-y-3">
          {loadingConversations ? (
            /* Loading Skeletons */
            <div className="p-2 space-y-2">
              <div className="h-3 w-20 bg-slate-800 rounded animate-pulse" />
              <div className="h-8 bg-slate-800/80 rounded-lg animate-pulse" />
              <div className="h-8 bg-slate-800/80 rounded-lg animate-pulse" />
              <div className="h-8 bg-slate-800/80 rounded-lg animate-pulse" />
            </div>
          ) : filteredConversations.length === 0 ? (
            <div className="p-4 text-center rounded-lg bg-slate-800/30 border border-slate-800/60 text-xs text-slate-400">
              <p className="font-medium text-slate-300">
                {searchQuery ? "No matches found" : "No previous sessions"}
              </p>
              <p className="mt-1 text-[11px] text-slate-400">
                {searchQuery
                  ? `No conversations match "${searchQuery}"`
                  : "Your conversations will be saved and listed here."}
              </p>
            </div>
          ) : searchQuery ? (
            /* Flat search list */
            <div className="space-y-1">
              <div className="px-2 py-1 text-[10px] font-semibold text-slate-400 uppercase tracking-wider">
                Matching Results ({filteredConversations.length})
              </div>
              {filteredConversations.map((c) => renderConversationItem(c))}
            </div>
          ) : (
            /* Chronologically Grouped List */
            Object.entries(grouped).map(([groupTitle, items]) => {
              if (items.length === 0) return null;
              return (
                <div key={groupTitle} className="space-y-1">
                  <div className="px-2 py-0.5 text-[10px] font-semibold text-slate-400 uppercase tracking-wider">
                    {groupTitle}
                  </div>
                  {items.map((c) => renderConversationItem(c))}
                </div>
              );
            })
          )}
        </div>

        {/* System / Environment Footer */}
        <div className="p-3 border-t border-slate-800 text-xs text-slate-400">
          <div className="flex items-center space-x-2">
            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
            <span className="text-slate-300 font-medium text-[11px]">
              Hybrid RAG Connected
            </span>
          </div>
          <p className="mt-1 text-[10px] text-slate-400">
            PostgreSQL Vector + FTS + RRF
          </p>
        </div>
      </aside>
    </>
  );

  function renderConversationItem(c: ConversationSummary) {
    const isActive = c.id === activeConversationId;
    const isEditing = editingId === c.id;
    const isDeleting = isDeletingId === c.id;
    const isConfirmingDelete = confirmDeleteId === c.id;

    if (isEditing) {
      return (
        <form
          key={c.id}
          onSubmit={(e) => handleSaveRename(c.id, e)}
          className="flex items-center space-x-1.5 p-1.5 rounded-lg bg-slate-800 border border-indigo-500/80"
          onClick={(e) => e.stopPropagation()}
        >
          <input
            type="text"
            value={editTitle}
            onChange={(e) => setEditTitle(e.target.value)}
            disabled={isSavingRename}
            autoFocus
            className="flex-1 bg-slate-900 text-white text-xs px-2 py-1 rounded border border-slate-700 focus:outline-none focus:ring-1 focus:ring-indigo-500"
          />
          <button
            type="submit"
            disabled={isSavingRename || !editTitle.trim()}
            className="p-1 text-emerald-400 hover:text-emerald-300 rounded hover:bg-slate-700"
            title="Save title"
          >
            <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
            </svg>
          </button>
          <button
            type="button"
            onClick={handleCancelRename}
            className="p-1 text-slate-400 hover:text-slate-200 rounded hover:bg-slate-700"
            title="Cancel"
          >
            <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </form>
      );
    }

    return (
      <div
        key={c.id}
        onClick={() => {
          setConfirmDeleteId(null);
          onSelectConversation(c);
          onClose();
        }}
        className={`group flex items-center justify-between px-2.5 py-2 rounded-lg text-xs cursor-pointer transition-colors ${
          isActive
            ? "bg-slate-800 text-white font-medium border border-slate-700/80 shadow-2xs"
            : "text-slate-300 hover:bg-slate-800/60 hover:text-white"
        }`}
      >
        <div className="flex items-center space-x-2 min-w-0 mr-1 flex-1">
          <svg
            className={`w-3.5 h-3.5 shrink-0 ${
              isActive ? "text-indigo-400" : "text-slate-500"
            }`}
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"
            />
          </svg>
          <span className="truncate" title={c.title}>
            {c.title || "Untitled Conversation"}
          </span>
        </div>

        {/* Hover action buttons */}
        <div className="flex items-center space-x-0.5 shrink-0">
          {isConfirmingDelete ? (
            <div className="flex items-center space-x-1" onClick={(e) => e.stopPropagation()}>
              <button
                onClick={(e) => handleDeleteClick(c.id, e)}
                disabled={isDeleting}
                className="px-1.5 py-0.5 bg-red-600 hover:bg-red-500 text-white text-[10px] rounded font-medium transition"
              >
                Delete?
              </button>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  setConfirmDeleteId(null);
                }}
                className="p-1 text-slate-400 hover:text-white rounded"
                title="Cancel"
              >
                ✕
              </button>
            </div>
          ) : (
            <>
              {/* Rename Button */}
              <button
                onClick={(e) => handleStartRename(c, e)}
                className="opacity-0 group-hover:opacity-100 hover:bg-slate-700 text-slate-400 hover:text-indigo-300 p-1 rounded transition-opacity"
                title="Rename conversation"
                aria-label="Rename conversation"
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

              {/* Delete Button */}
              <button
                onClick={(e) => handleDeleteClick(c.id, e)}
                disabled={isDeleting}
                className="opacity-0 group-hover:opacity-100 hover:bg-slate-700 text-slate-400 hover:text-red-400 p-1 rounded transition-opacity"
                title="Delete conversation"
                aria-label="Delete conversation"
              >
                <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
                  />
                </svg>
              </button>
            </>
          )}
        </div>
      </div>
    );
  }
}
