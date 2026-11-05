"use client";

import { useEffect, useState, useCallback } from "react";
import type { Session } from "@supabase/supabase-js";
import { supabase } from "../lib/supabaseClient";
import AppHeader from "../components/AppHeader";
import ChatWindow from "../components/ChatWindow";
import RoleDashboard from "../components/RoleDashboard";
import DocumentUploadModal from "../components/DocumentUploadModal";
import UserManagement from "../components/UserManagement";
import AdminEvaluationDashboard from "../components/AdminEvaluationDashboard";
import { useRouter } from "next/navigation";
import type { UserProfile } from "../lib/types";

export default function Home() {
  const [session, setSession] = useState<Session | null>(null);
  const [profile, setProfile] = useState<UserProfile | null>(null);
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [activeView, setActiveView] = useState<"chat" | "hub" | "users" | "evaluations">("chat");
  const [isUploadModalOpen, setIsUploadModalOpen] = useState(false);
  const [initialPrompt, setInitialPrompt] = useState<string | null>(null);
  const router = useRouter();

  const loadUserProfile = useCallback(async (userId: string) => {
    try {
      const { data, error } = await supabase
        .from("user_profiles")
        .select("id, tenant_id, roles")
        .eq("id", userId)
        .single();

      if (!error && data) {
        setProfile(data as UserProfile);
      } else {
        setProfile({
          id: userId,
          tenant_id: "",
          roles: ["employee"],
        });
      }
    } catch {
      setProfile({
        id: userId,
        tenant_id: "",
        roles: ["employee"],
      });
    }
  }, []);

  useEffect(() => {
    supabase.auth.getSession().then(({ data: { session } }) => {
      if (!session) {
        router.push("/login");
      } else {
        setSession(session);
        loadUserProfile(session.user.id);
      }
    });

    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange((_event, session) => {
      if (!session) {
        router.push("/login");
      } else {
        setSession(session);
        loadUserProfile(session.user.id);
      }
    });

    return () => subscription.unsubscribe();
  }, [router, loadUserProfile]);

  const handleAskFromHub = (query: string) => {
    setInitialPrompt(query);
    setActiveView("chat");
  };

  if (!session) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-slate-50 text-slate-600 text-sm font-medium">
        <div className="flex items-center space-x-2">
          <span className="w-2.5 h-2.5 rounded-full bg-indigo-600 animate-ping" />
          <span>Verifying authentication session...</span>
        </div>
      </div>
    );
  }

  const userRoles =
    profile?.roles && profile.roles.length > 0 ? profile.roles : ["employee"];

  return (
    <div className="min-h-screen flex flex-col bg-slate-50 text-slate-900">
      <AppHeader
        userEmail={session.user.email ?? null}
        roles={userRoles}
        tenantId={profile?.tenant_id ?? null}
        isSidebarOpen={isSidebarOpen}
        onToggleSidebar={() => setIsSidebarOpen((prev) => !prev)}
        activeView={activeView}
        onViewChange={(view) => setActiveView(view)}
      />

      {activeView === "chat" ? (
        <ChatWindow
          key={initialPrompt || "chat"}
          isSidebarOpen={isSidebarOpen}
          onCloseSidebar={() => setIsSidebarOpen(false)}
          roles={userRoles}
          initialPrompt={initialPrompt}
        />
      ) : activeView === "hub" ? (
        <RoleDashboard
          roles={userRoles}
          tenantId={profile?.tenant_id ?? null}
          onAskQuestion={handleAskFromHub}
          onOpenUploadModal={() => setIsUploadModalOpen(true)}
        />
      ) : activeView === "users" && userRoles.includes("manager") ? (
        <UserManagement />
      ) : activeView === "evaluations" &&
        (userRoles.includes("executive") ||
          userRoles.includes("admin") ||
          userRoles.includes("manager")) ? (
        <AdminEvaluationDashboard />
      ) : (
        <ChatWindow
          isSidebarOpen={isSidebarOpen}
          onCloseSidebar={() => setIsSidebarOpen(false)}
          roles={userRoles}
          initialPrompt={null}
        />
      )}

      <DocumentUploadModal
        isOpen={isUploadModalOpen}
        onClose={() => setIsUploadModalOpen(false)}
      />
    </div>
  );
}
