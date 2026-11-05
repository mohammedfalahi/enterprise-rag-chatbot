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
  notice?: string;
}

export interface ConversationSummary {
  id: string;
  tenant_id: string;
  user_id: string;
  title: string;
  created_at: string;
  updated_at: string;
}

export interface EvaluationSummary {
  tenant_id: string;
  sample_size: number;
  total_benchmark_cases: number;
  models: {
    embedding_model: string;
    embedding_dimensions: number;
    reranker_model: string;
    generation_model: string;
  };
  generation_metrics: {
    faithfulness: number;
    answer_relevancy: number;
    answer_correctness: number;
    context_recall: number;
    context_precision: number;
  };
  retrieval_metrics: {
    top_k: number;
    recall_at_k: number;
    precision_at_k: number;
    mrr: number;
    ndcg_at_k: number;
    permission_leakage_rate: number;
  };
  refusal_metrics: {
    refusal_precision: number;
    refusal_recall: number;
    refusal_f1: number;
    prompt_injection_refusal_rate: number;
  };
}

export interface MetricTrendPoint {
  name: string;
  mode: string;
  top_k: number;
  recall: number;
  precision: number;
  mrr: number;
  ndcg: number;
  leakage: number;
  refusal: number;
}

export interface EvaluationHistoryResponse {
  tenant_id: string;
  runs_count: number;
  trends: MetricTrendPoint[];
  experiments: Array<Record<string, unknown>>;
}

export interface QueryEvaluationItem {
  test_id: string;
  question: string;
  role: string;
  category: string;
  expected_behavior: string;
  status: "passed" | "partial" | "failed";
  recall_at_5: number;
  precision_at_5: number;
  mrr: number;
  ndcg_at_5: number;
  gold_doc_ids: string[];
  retrieved_doc_ids: string[];
  retrieved_titles: string[];
  generation_scores?: {
    faithfulness?: number;
    answer_relevancy?: number;
    answer_correctness?: number;
    context_recall?: number;
    llm_context_precision_with_reference?: number;
  };
  response?: string | null;
  reference?: string | null;
}

export interface QueryEvaluationsResponse {
  total: number;
  page: number;
  limit: number;
  items: QueryEvaluationItem[];
}

export interface ErrorAnalysisResponse {
  tenant_id: string;
  total_zero_hits: number;
  categories_distribution: Record<string, number>;
  diagnosed_failures: Array<{
    test_id: string;
    category: string;
    reason: string;
    dense_best_rank?: number;
    cross_encoder_score?: number;
  }>;
}

