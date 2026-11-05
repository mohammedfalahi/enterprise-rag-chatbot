"use client";

import React from "react";
import { supabase } from "../lib/supabaseClient";

interface AppHeaderProps {
  userEmail: string | null;
  roles: string[];
  tenantId: string | null;
  isSidebarOpen: boolean;
  onToggleSidebar: () => void;
  activeView?: "chat" | "hub" | "users";
  onViewChange?: (view: "chat" | "hub" | "users") => void;
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
      return "Employee";
    default:
      return role.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
  }
}

function getRoleBadgeStyle(role: string): string {
  switch (role) {
    case "executive":
      return "bg-purple-50 text-purple-700 border-purple-200";
    case "manager":
      return "bg-blue-50 text-blue-700 border-blue-200";
    case "sales_rep":
      return "bg-emerald-50 text-emerald-700 border-emerald-200";
    case "employee":
    default:
      return "bg-slate-100 text-slate-700 border-slate-200";
  }
}

export default function AppHeader({
  userEmail,
  roles,
  tenantId,
  isSidebarOpen,
  onToggleSidebar,
  activeView = "chat",
  onViewChange,
}: AppHeaderProps) {
  const primaryRole = roles[0] || "employee";

  return (
    <header className="sticky top-0 z-20 flex min-h-16 flex-wrap items-center justify-between gap-x-4 gap-y-2 border-b border-slate-200 bg-white px-3 py-2 md:flex-nowrap md:px-6 md:py-0">
      <div className="flex min-w-0 items-center space-x-2 sm:space-x-3">
        {/* Toggle sidebar button (mobile & desktop) */}
        <button
          onClick={onToggleSidebar}
          aria-label={isSidebarOpen ? "Collapse sidebar" : "Expand sidebar"}
          className="rounded-lg p-2 text-slate-600 transition-colors hover:bg-slate-100 hover:text-slate-900 focus:outline-none focus:ring-2 focus:ring-indigo-300"
        >
          <svg
            className="w-5 h-5"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
            xmlns="http://www.w3.org/2000/svg"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M4 6h16M4 12h16M4 18h16"
            />
          </svg>
        </button>

        {/* Enterprise Brand Identity */}
        <div className="flex items-center space-x-2.5">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-indigo-700 text-sm font-semibold text-white">
            EK
          </div>
          <div>
            <h1 className="truncate text-sm font-semibold leading-tight text-slate-900 sm:text-base">
              Enterprise Knowledge
            </h1>
            <p className="text-xs text-slate-500 hidden sm:block">
              RAG Intelligence Assistant
            </p>
          </div>
        </div>
      </div>

      {/* View Switcher: Chat vs Role Hub */}
      {onViewChange && (
        <nav aria-label="Main navigation" className="order-3 flex w-full items-center gap-1 overflow-x-auto rounded-xl border border-slate-200 bg-slate-50 p-1 md:order-none md:w-auto">
          <button
            onClick={() => onViewChange("chat")}
            className={`shrink-0 rounded-lg px-3 py-1.5 text-xs font-medium transition-colors cursor-pointer ${
              activeView === "chat"
                ? "bg-white text-indigo-700 shadow-2xs font-semibold"
                : "text-slate-600 hover:text-slate-900"
            }`}
          >
            Chat Assistant
          </button>
          <button
            onClick={() => onViewChange("hub")}
            className={`flex shrink-0 cursor-pointer items-center space-x-1 rounded-lg px-3 py-1.5 text-xs font-medium transition-colors ${
              activeView === "hub"
                ? "bg-white text-indigo-700 shadow-2xs font-semibold"
                : "text-slate-600 hover:text-slate-900"
            }`}
          >
            <span>Role Hub</span>
          </button>
          {roles.includes("manager") && (
            <button
              onClick={() => onViewChange("users")}
              className={`shrink-0 cursor-pointer rounded-lg px-3 py-1.5 text-xs font-medium transition-colors ${
                activeView === "users"
                  ? "bg-white text-indigo-700 shadow-2xs font-semibold"
                  : "text-slate-600 hover:text-slate-900"
              }`}
            >
              User Management
            </button>
          )}
        </nav>
      )}

      {/* User Information & Role */}
      <div className="ml-auto flex shrink-0 items-center space-x-2 sm:space-x-4">
        <div className="flex items-center space-x-2 text-right">
          <div className="hidden md:block">
            <div className="text-xs font-medium text-slate-800 truncate max-w-[180px]">
              {userEmail || "Authenticated User"}
            </div>
            {tenantId && (
              <div className="text-[10px] text-slate-400 font-mono truncate max-w-[180px]">
                tenant: {tenantId.slice(0, 8)}...
              </div>
            )}
          </div>

          {/* Role badge */}
          <span
            className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium border ${getRoleBadgeStyle(
              primaryRole
            )}`}
            title={`Assigned roles: ${roles.join(", ")}`}
          >
            {formatRoleLabel(primaryRole)}
          </span>
        </div>

        <div className="h-5 w-px bg-slate-200 hidden sm:block" />

        <button
          onClick={() => supabase.auth.signOut()}
          className="rounded-md px-2.5 py-1.5 text-xs font-medium text-slate-600 transition-colors hover:bg-slate-100 hover:text-slate-900 sm:text-sm"
        >
          Sign Out
        </button>
      </div>
    </header>
  );
}
