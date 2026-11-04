"use client";

import React, { useState } from "react";
import { supabase } from "../lib/supabaseClient";
import { BACKEND_URL } from "../lib/api";

interface DocumentUploadModalProps {
  isOpen: boolean;
  onClose: () => void;
  onUploadSuccess?: (doc: { document_id: string; title: string }) => void;
}

const MAX_SIZE_BYTES = 15 * 1024 * 1024; // 15 MB
const ALLOWED_EXTS = [".pdf", ".docx", ".txt", ".md", ".markdown"];

export default function DocumentUploadModal({
  isOpen,
  onClose,
  onUploadSuccess,
}: DocumentUploadModalProps) {
  const [file, setFile] = useState<File | null>(null);
  const [title, setTitle] = useState("");
  const [selectedRoles, setSelectedRoles] = useState<string[]>([
    "employee",
    "sales_rep",
    "manager",
    "executive",
  ]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [successInfo, setSuccessInfo] = useState<{
    document_id: string;
    title: string;
    size_bytes: number;
  } | null>(null);
  const [isDragging, setIsDragging] = useState(false);

  if (!isOpen) return null;

  const handleFileSelection = (selectedFile: File) => {
    setError(null);
    setSuccessInfo(null);

    const ext = "." + (selectedFile.name.split(".").pop() || "").toLowerCase();
    if (!ALLOWED_EXTS.includes(ext)) {
      setError(
        `Unsupported file type '${ext}'. Please choose a PDF, DOCX, TXT, or MD document.`
      );
      return;
    }

    if (selectedFile.size > MAX_SIZE_BYTES) {
      setError("File exceeds the maximum allowable size of 15 MB.");
      return;
    }

    setFile(selectedFile);
    if (!title.trim()) {
      setTitle(selectedFile.name.replace(/\.[^/.]+$/, ""));
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileSelection(e.dataTransfer.files[0]);
    }
  };

  const toggleRole = (role: string) => {
    setSelectedRoles((prev) =>
      prev.includes(role)
        ? prev.filter((r) => r !== role)
        : [...prev, role]
    );
  };

  const handleUpload = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file || loading) return;

    if (!title.trim()) {
      setError("Please provide a title for the document.");
      return;
    }

    if (selectedRoles.length === 0) {
      setError("Please select at least one role authorized to access this document.");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const {
        data: { session },
      } = await supabase.auth.getSession();
      const token = session?.access_token;

      const formData = new FormData();
      formData.append("file", file);
      formData.append("title", title.trim());
      formData.append("access_roles", selectedRoles.join(","));

      const res = await fetch(`${BACKEND_URL}/api/documents/upload`, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${token}`,
        },
        body: formData,
      });

      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.detail || `Upload failed with status ${res.status}`);
      }

      setSuccessInfo(data);
      if (onUploadSuccess) {
        onUploadSuccess({ document_id: data.document_id, title: data.title });
      }
    } catch (err: unknown) {
      console.error("Document upload error:", err);
      const msg = err instanceof Error ? err.message : "Failed to upload document.";
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  const resetForm = () => {
    setFile(null);
    setTitle("");
    setError(null);
    setSuccessInfo(null);
    setSelectedRoles(["employee", "sales_rep", "manager", "executive"]);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/60 backdrop-blur-xs">
      <div
        className="w-full max-w-lg bg-white rounded-2xl shadow-xl border border-slate-200 overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Modal Header */}
        <div className="px-6 py-4 border-b border-slate-200 flex items-center justify-between bg-slate-50">
          <div>
            <h3 className="font-semibold text-slate-900 text-base">
              Ingest Enterprise Document
            </h3>
            <p className="text-xs text-slate-500">
              Authorized for Managers & Executives (PDF, DOCX, TXT, MD up to 15MB)
            </p>
          </div>
          <button
            onClick={() => {
              resetForm();
              onClose();
            }}
            className="p-1.5 rounded-lg text-slate-400 hover:text-slate-700 hover:bg-slate-200 transition"
            title="Close modal"
          >
            ✕
          </button>
        </div>

        {/* Modal Body */}
        <div className="p-6 space-y-4">
          {successInfo ? (
            /* Success Feedback */
            <div className="p-4 rounded-xl bg-emerald-50 border border-emerald-200 text-center space-y-3">
              <div className="w-10 h-10 rounded-full bg-emerald-100 text-emerald-700 flex items-center justify-center mx-auto text-lg font-bold">
                ✓
              </div>
              <div>
                <h4 className="font-semibold text-emerald-900 text-sm">
                  Document Ingested Successfully
                </h4>
                <p className="text-xs text-emerald-700 mt-1">
                  &ldquo;{successInfo.title}&rdquo; is now indexed with pgvector embeddings and available for hybrid retrieval.
                </p>
              </div>
              <div className="text-[11px] font-mono text-emerald-800 bg-white/80 p-2 rounded border border-emerald-200/80 truncate">
                ID: {successInfo.document_id}
              </div>
              <div className="flex justify-center space-x-2 pt-1">
                <button
                  onClick={resetForm}
                  className="px-3 py-1.5 bg-emerald-600 hover:bg-emerald-700 text-white rounded-lg text-xs font-medium transition cursor-pointer"
                >
                  Upload Another
                </button>
                <button
                  onClick={() => {
                    resetForm();
                    onClose();
                  }}
                  className="px-3 py-1.5 bg-slate-200 hover:bg-slate-300 text-slate-800 rounded-lg text-xs font-medium transition cursor-pointer"
                >
                  Done
                </button>
              </div>
            </div>
          ) : (
            <form onSubmit={handleUpload} className="space-y-4">
              {/* Drag-and-Drop Area */}
              <div
                onDragOver={(e) => {
                  e.preventDefault();
                  setIsDragging(true);
                }}
                onDragLeave={() => setIsDragging(false)}
                onDrop={handleDrop}
                className={`border-2 border-dashed rounded-xl p-5 text-center transition cursor-pointer ${
                  isDragging
                    ? "border-indigo-500 bg-indigo-50/50"
                    : file
                    ? "border-emerald-400 bg-emerald-50/30"
                    : "border-slate-300 hover:border-indigo-400 bg-slate-50/50"
                }`}
                onClick={() => {
                  const el = document.getElementById("doc-file-input");
                  el?.click();
                }}
              >
                <input
                  id="doc-file-input"
                  type="file"
                  accept=".pdf,.docx,.txt,.md,.markdown"
                  className="hidden"
                  onChange={(e) => {
                    if (e.target.files && e.target.files.length > 0) {
                      handleFileSelection(e.target.files[0]);
                    }
                  }}
                />
                {file ? (
                  <div className="space-y-1">
                    <span className="text-xl">📄</span>
                    <p className="text-xs font-semibold text-slate-800 truncate">
                      {file.name}
                    </p>
                    <p className="text-[11px] text-slate-500">
                      {(file.size / (1024 * 1024)).toFixed(2)} MB • Click to replace
                    </p>
                  </div>
                ) : (
                  <div className="space-y-1">
                    <span className="text-xl">📁</span>
                    <p className="text-xs font-semibold text-slate-700">
                      Drag & drop your enterprise document here
                    </p>
                    <p className="text-[11px] text-slate-500">
                      or click to browse from your computer (Max 15MB)
                    </p>
                  </div>
                )}
              </div>

              {/* Title Field */}
              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1">
                  Document Title
                </label>
                <input
                  type="text"
                  value={title}
                  onChange={(e) => setTitle(e.target.value)}
                  placeholder="e.g. Q4 Corporate Travel Policy"
                  className="w-full text-xs bg-slate-50 border border-slate-300 rounded-lg px-3 py-2 text-slate-900 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                  required
                />
              </div>

              {/* Target Access Roles Selector */}
              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1.5">
                  Authorized Enterprise Roles
                </label>
                <div className="grid grid-cols-2 gap-2 text-xs">
                  {[
                    { id: "employee", label: "Employee" },
                    { id: "sales_rep", label: "Sales Rep" },
                    { id: "manager", label: "Manager" },
                    { id: "executive", label: "Executive" },
                  ].map((r) => {
                    const isChecked = selectedRoles.includes(r.id);
                    return (
                      <label
                        key={r.id}
                        onClick={() => toggleRole(r.id)}
                        className={`flex items-center space-x-2 p-2 rounded-lg border cursor-pointer select-none transition ${
                          isChecked
                            ? "bg-indigo-50/70 border-indigo-300 text-indigo-950 font-medium"
                            : "bg-slate-50 border-slate-200 text-slate-600 hover:bg-slate-100"
                        }`}
                      >
                        <input
                          type="checkbox"
                          checked={isChecked}
                          onChange={() => {}}
                          className="rounded text-indigo-600 focus:ring-indigo-500 w-3.5 h-3.5"
                        />
                        <span>{r.label}</span>
                      </label>
                    );
                  })}
                </div>
              </div>

              {/* Error feedback */}
              {error && (
                <div className="p-2.5 rounded-lg bg-red-50 border border-red-200 text-xs text-red-700 flex items-center space-x-2">
                  <span>⚠️</span>
                  <span>{error}</span>
                </div>
              )}

              {/* Actions */}
              <div className="flex justify-end space-x-2 pt-2">
                <button
                  type="button"
                  onClick={() => {
                    resetForm();
                    onClose();
                  }}
                  disabled={loading}
                  className="px-4 py-2 text-xs font-medium text-slate-600 hover:text-slate-800 rounded-lg hover:bg-slate-100 transition cursor-pointer"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={loading || !file}
                  className="px-4 py-2 text-xs font-medium text-white bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 rounded-lg transition shadow-xs flex items-center space-x-2 cursor-pointer"
                >
                  {loading && (
                    <div className="w-3.5 h-3.5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                  )}
                  <span>{loading ? "Ingesting Document..." : "Ingest Document"}</span>
                </button>
              </div>
            </form>
          )}
        </div>
      </div>
    </div>
  );
}
