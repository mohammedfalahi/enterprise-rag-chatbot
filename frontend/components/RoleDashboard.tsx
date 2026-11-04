"use client";

import React from "react";

interface RoleDashboardProps {
  roles: string[];
  tenantId: string | null;
  onAskQuestion: (query: string) => void;
  onOpenUploadModal?: () => void;
}

interface ResourceCard {
  title: string;
  category: string;
  description: string;
  suggestedQuery: string;
}

function getRoleMetadata(primaryRole: string): {
  title: string;
  description: string;
  canUpload: boolean;
  resources: ResourceCard[];
} {
  switch (primaryRole) {
    case "sales_rep":
      return {
        title: "Sales Representative Hub",
        description:
          "Access commercial documentation, enterprise tier pricing, customer discount thresholds, and competitor battlecards authorized for commercial representatives.",
        canUpload: false,
        resources: [
          {
            title: "Enterprise Product Pricing Schedules",
            category: "Commercial",
            description:
              "Verified enterprise contract tiers, volume discounts, and annual renewal commitments.",
            suggestedQuery: "What are our enterprise product pricing tiers and annual discount rates?",
          },
          {
            title: "Customer Discount Authorization Matrix",
            category: "Governance",
            description:
              "Threshold rules governing sales rep discount discretion vs. required manager/executive escalation.",
            suggestedQuery: "What are the discount approval limits for sales representatives?",
          },
          {
            title: "Competitor Battlecards & Product Comparison",
            category: "Strategy",
            description:
              "Positioning matrices, key product differentiators, and competitive displacement strategies.",
            suggestedQuery: "What are our primary competitive advantages and competitor battlecard points?",
          },
          {
            title: "Enterprise Custom Contract SLA Guidelines",
            category: "Legal & SLA",
            description:
              "Standard customer service level agreement terms, uptime guarantees, and legal boilerplate.",
            suggestedQuery: "What are our standard enterprise SLA uptime commitments and support tiers?",
          },
        ],
      };

    case "manager":
      return {
        title: "Management Intelligence Hub",
        description:
          "Departmental operational guidelines, performance review workflows, hiring headcount authorization, and document ingestion for team knowledge.",
        canUpload: true,
        resources: [
          {
            title: "Team Headcount & Hiring Approval Workflow",
            category: "Operations",
            description:
              "Requisition submission steps, interview panel standards, and job level compensation bands.",
            suggestedQuery: "What is the procedure for requesting and approving new team headcount?",
          },
          {
            title: "Quarterly Performance & Review Rubrics",
            category: "People",
            description:
              "Evaluation criteria, rating calibration guidelines, and promotion justification standards.",
            suggestedQuery: "What are the quarterly performance evaluation rubrics and review cycles?",
          },
          {
            title: "Department Budget & Authorization Limits",
            category: "Finance",
            description:
              "Signing authority matrices for manager operational expenditures, software tools, and travel.",
            suggestedQuery: "What are the department spending approval thresholds for managers?",
          },
          {
            title: "Promotion & Career Leveling Criteria",
            category: "Talent",
            description:
              "Role leveling ladders, competencies, and promotion committee review calendars.",
            suggestedQuery: "What are the internal promotion and career advancement criteria?",
          },
        ],
      };

    case "executive":
      return {
        title: "Executive Governance Hub",
        description:
          "Corporate governance, board review summaries, strategic company objectives, and enterprise-wide document ingestion authorization.",
        canUpload: true,
        resources: [
          {
            title: "Corporate Strategic Objectives & Annual OKRs",
            category: "Strategy",
            description:
              "Company-wide annual growth targets, strategic focus areas, and executive key results.",
            suggestedQuery: "Summarize our strategic company objectives, annual OKRs, and growth targets.",
          },
          {
            title: "Corporate Governance & Board Guidelines",
            category: "Governance",
            description:
              "Charter responsibilities, board audit guidelines, and delegated signature powers.",
            suggestedQuery: "What are the corporate governance protocols and board review guidelines?",
          },
          {
            title: "Enterprise Risk & Compliance Audit Matrix",
            category: "Compliance",
            description:
              "Regulatory compliance controls, SOC2 / ISO audit requirements, and risk mitigation policies.",
            suggestedQuery: "What are our enterprise risk compliance requirements and audit controls?",
          },
          {
            title: "Delegated Financial Powers & Signing Authority",
            category: "Finance",
            description:
              "Executive contract signing thresholds, M&A review workflows, and corporate treasury policies.",
            suggestedQuery: "What are the delegated financial signing limits for executive officers?",
          },
        ],
      };

    case "employee":
    default:
      return {
        title: "Employee Resource Hub",
        description:
          "Company-wide policies, workplace guidelines, employee benefits directories, and operational FAQs accessible to all team members.",
        canUpload: false,
        resources: [
          {
            title: "Remote Work & Travel Policy Guide",
            category: "Policy",
            description:
              "Flexible workplace rules, home office allowances, and business travel booking procedures.",
            suggestedQuery: "What is our company remote work and business travel expense policy?",
          },
          {
            title: "Employee Healthcare & 401(k) Benefits",
            category: "Benefits",
            description:
              "Health coverage plans, employer 401(k) matching tiers, and open enrollment schedules.",
            suggestedQuery: "What are our employee healthcare options and 401(k) employer matching details?",
          },
          {
            title: "Expense Reimbursement & Credit Card Rules",
            category: "Finance",
            description:
              "Permissible meal and travel expenses, receipt submission timelines, and reimbursement workflows.",
            suggestedQuery: "How do I submit an expense report and what are the reimbursement limits?",
          },
          {
            title: "Official Holidays & Paid Leave Entitlements",
            category: "Time Off",
            description:
              "Accrual rates, statutory holiday calendars, parental leave policies, and sick day logging.",
            suggestedQuery: "What is our official paid time off policy and annual holiday schedule?",
          },
        ],
      };
  }
}

export default function RoleDashboard({
  roles,
  tenantId,
  onAskQuestion,
  onOpenUploadModal,
}: RoleDashboardProps) {
  const primaryRole = roles[0] || "employee";
  const { title, description, canUpload, resources } = getRoleMetadata(primaryRole);

  return (
    <div className="flex-1 overflow-y-auto bg-slate-50 p-4 sm:p-8">
      <div className="max-w-5xl mx-auto space-y-6">
        {/* Hub Banner */}
        <div className="p-6 rounded-2xl bg-white border border-slate-200/90 shadow-2xs flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div className="space-y-1.5 max-w-2xl">
            <div className="flex items-center space-x-2">
              <span className="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-indigo-50 text-indigo-700 border border-indigo-100 uppercase tracking-wider">
                {primaryRole.replace(/_/g, " ")} View
              </span>
              {tenantId && (
                <span className="text-xs font-mono text-slate-400">
                  tenant: {tenantId.slice(0, 8)}...
                </span>
              )}
            </div>
            <h2 className="text-xl sm:text-2xl font-bold text-slate-900">
              {title}
            </h2>
            <p className="text-xs sm:text-sm text-slate-500 leading-relaxed">
              {description}
            </p>
          </div>

          {/* Action Trigger for Manager / Executive */}
          {canUpload && onOpenUploadModal ? (
            <button
              onClick={onOpenUploadModal}
              className="px-4 py-2.5 bg-indigo-600 hover:bg-indigo-500 text-white rounded-xl text-xs sm:text-sm font-medium transition shadow-xs flex items-center justify-center space-x-2 shrink-0 cursor-pointer"
            >
              <span>📄</span>
              <span>Ingest Document</span>
            </button>
          ) : (
            <div className="p-3 rounded-xl bg-slate-50 border border-slate-200 text-[11px] text-slate-500 max-w-xs shrink-0">
              <span className="font-semibold text-slate-700">Document Upload:</span> Restricted to Manager & Executive roles.
            </div>
          )}
        </div>

        {/* Resources Section Header */}
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-base font-semibold text-slate-900">
              Verified Enterprise Resources
            </h3>
            <p className="text-xs text-slate-500">
              Select any resource to query the knowledge assistant with pre-grounded context.
            </p>
          </div>
        </div>

        {/* Reusable Resource Cards Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {resources.map((res, idx) => (
            <div
              key={idx}
              className="p-5 rounded-2xl bg-white border border-slate-200/80 shadow-2xs hover:border-indigo-300 hover:shadow-xs transition-all flex flex-col justify-between space-y-4"
            >
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-[10px] font-semibold uppercase tracking-wider px-2 py-0.5 rounded bg-slate-100 text-slate-600 border border-slate-200">
                    {res.category}
                  </span>
                  <span className="text-slate-400 text-xs">Verified Doc</span>
                </div>
                <h4 className="font-semibold text-slate-900 text-sm">
                  {res.title}
                </h4>
                <p className="text-xs text-slate-500 leading-relaxed">
                  {res.description}
                </p>
              </div>

              <div className="pt-3 border-t border-slate-100 flex items-center justify-between">
                <span className="text-[11px] text-slate-400 italic truncate max-w-[240px]">
                  &ldquo;{res.suggestedQuery}&rdquo;
                </span>
                <button
                  onClick={() => onAskQuestion(res.suggestedQuery)}
                  className="px-3 py-1.5 bg-indigo-50 hover:bg-indigo-100 text-indigo-700 rounded-lg text-xs font-medium transition flex items-center space-x-1 shrink-0 cursor-pointer"
                >
                  <span>Query Assistant</span>
                  <span>→</span>
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
