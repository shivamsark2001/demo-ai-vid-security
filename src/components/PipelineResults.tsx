'use client';

import { AlertTriangle, CheckCircle, Eye, Brain } from 'lucide-react';

interface PipelineResultsProps {
  result: {
    category: string;
    reasoning: string;
    anomalyFramesB64?: string | null;
    error?: string;
  };
}

export function PipelineResults({ result }: PipelineResultsProps) {
  // Handle errors
  if (result.error) {
    return (
      <div className="glass-card p-6 text-center">
        <AlertTriangle className="w-12 h-12 mx-auto mb-3 text-[var(--accent-danger)] opacity-60" />
        <p className="text-[var(--text-secondary)]">Analysis Failed</p>
        <p className="text-xs text-[var(--text-muted)] mt-1">{result.error}</p>
      </div>
    );
  }

  const isAnomaly = result.category?.toLowerCase() !== 'normal';
  const category = result.category || 'unknown';
  const reasoning = result.reasoning || 'No analysis available';

  return (
    <div className="space-y-4">
      {/* Main Verdict Card */}
      <div className="glass-card overflow-hidden">
        <div className={`p-6 ${
          isAnomaly 
            ? 'bg-gradient-to-r from-[var(--accent-danger)]/20 to-transparent' 
            : 'bg-gradient-to-r from-[var(--accent-primary)]/20 to-transparent'
        }`}>
          <div className="flex items-start gap-4">
            <div className={`w-16 h-16 rounded-xl flex items-center justify-center ${
              isAnomaly 
                ? 'bg-[var(--accent-danger)]/20' 
                : 'bg-[var(--accent-primary)]/20'
            }`}>
              {isAnomaly ? (
                <AlertTriangle className="w-8 h-8 text-[var(--accent-danger)]" />
              ) : (
                <CheckCircle className="w-8 h-8 text-[var(--accent-primary)]" />
              )}
            </div>
            <div className="flex-1">
              <h3 className={`text-2xl font-bold ${
                isAnomaly ? 'text-[var(--accent-danger)]' : 'text-[var(--accent-primary)]'
              }`}>
                {isAnomaly ? '🚨 Anomaly Detected' : '✅ Normal Activity'}
              </h3>
              {isAnomaly && (
                <div className="mt-2">
                  <span className={`inline-block px-3 py-1 rounded-full text-sm font-semibold uppercase tracking-wider ${
                    'bg-[var(--accent-danger)]/20 text-[var(--accent-danger)]'
                  }`}>
                    {category}
                  </span>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Anomaly Frames Grid - 8 frames (2x4) */}
      {result.anomalyFramesB64 && (
        <div className="glass-card p-4">
          <div className="flex items-center gap-2 mb-3">
            <Eye className="w-4 h-4 text-[var(--accent-danger)]" />
            <span className="text-xs font-medium text-[var(--text-muted)] uppercase tracking-wider">
              Anomaly Evidence
            </span>
            <span className="text-[10px] text-[var(--text-muted)] ml-auto">
              8 frames analyzed
            </span>
          </div>
          <img 
            src={result.anomalyFramesB64} 
            alt="Anomaly frames grid" 
            className={`w-full rounded-lg border ${
              isAnomaly 
                ? 'border-[var(--accent-danger)]/30' 
                : 'border-[var(--accent-primary)]/30'
            }`}
          />
        </div>
      )}

      {/* AI Reasoning */}
      <div className="glass-card p-5">
        <div className="flex items-center gap-2 mb-4">
          <Brain className={`w-5 h-5 ${
            isAnomaly ? 'text-[var(--accent-danger)]' : 'text-[var(--accent-primary)]'
          }`} />
          <span className="text-sm font-medium text-[var(--text-muted)] uppercase tracking-wider">
            AI Analysis
          </span>
        </div>
        <p className="text-base text-[var(--text-secondary)] leading-relaxed">
          {reasoning}
        </p>
      </div>
    </div>
  );
}
