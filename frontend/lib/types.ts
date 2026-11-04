export type EnterpriseRole = "employee" | "sales_rep" | "manager" | "executive";

export interface UserProfile {
  id: string;
  tenant_id: string;
  roles: EnterpriseRole[] | string[];
}

export interface Citation {
  chunk_id: string;
  document_id: string;
  title: string;
  heading_path: string[];
  score: number;
}

export interface ChatMessage {
  id?: string;
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
  created_at?: string;
}

export interface ConversationSummary {
  id: string;
  tenant_id: string;
  user_id: string;
  title: string;
  created_at: string;
  updated_at: string;
}
