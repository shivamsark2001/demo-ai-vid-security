'use client';

import { AnalysisResult } from '@/types';
import { AlertTriangle, CheckCircle, Eye } from 'lucide-react';

interface PipelineResultsProps {
  result: AnalysisResult;
}

export function PipelineResults({ result }: PipelineResultsProps) {
  const gemini = result.geminiVerification;

  if (!gemini) {
    return (
      <div className="glass-card p-6 text-center">
        <CheckCircle className="w-12 h-12 mx-auto mb-3 text-[var(--accent-primary)] opacity-60" />
        <p className="text-[var(--text-secondary)]">No anomalies detected</p>
        <p className="text-xs text-[var(--text-muted)] mt-1">Video appears normal</p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Annotated Grid */}
      {result.annotatedGridB64 && (
        <div className="glass-card p-4">
          <div className="flex items-center gap-2 mb-3">
            <Eye className="w-4 h-4 text-[var(--accent-primary)]" />
            <span className="text-xs font-medium text-[var(--text-muted)] uppercase tracking-wider">
              Analyzed Frames
            </span>
            <span className="text-[10px] text-[var(--text-muted)] ml-auto">
              {result.frameCount} frames
            </span>
          </div>
          <img 
            src={result.annotatedGridB64} 
            alt="Frame analysis" 
            className="w-full rounded-lg"
          />
        </div>
      )}

      {/* Main Result Card */}
      <div className="glass-card overflow-hidden">
        {/* Verdict Header */}
        <div className={`p-5 ${
          gemini.isAnomaly 
            ? 'bg-gradient-to-r from-[var(--accent-danger)]/20 to-transparent' 
            : 'bg-gradient-to-r from-[var(--accent-primary)]/20 to-transparent'
        }`}>
          <div className="flex items-start gap-4">
            <div className={`w-12 h-12 rounded-xl flex items-center justify-center ${
              gemini.isAnomaly 
                ? 'bg-[var(--accent-danger)]/20' 
                : 'bg-[var(--accent-primary)]/20'
            }`}>
              {gemini.isAnomaly ? (
                <AlertTriangle className="w-6 h-6 text-[var(--accent-danger)]" />
              ) : (
                <CheckCircle className="w-6 h-6 text-[var(--accent-primary)]" />
              )}
            </div>
            <div className="flex-1">
              <div className="flex items-center gap-3 mb-1">
                <h3 className={`text-lg font-semibold ${
                  gemini.isAnomaly ? 'text-[var(--accent-danger)]' : 'text-[var(--accent-primary)]'
                }`}>
                  {gemini.isAnomaly ? 'Anomaly Detected' : 'Normal Activity'}
                </h3>
                <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${
                  gemini.isAnomaly 
                    ? 'bg-[var(--accent-danger)]/20 text-[var(--accent-danger)]' 
                    : 'bg-[var(--accent-primary)]/20 text-[var(--accent-primary)]'
                }`}>
                  {Math.round(gemini.confidence * 100)}%
                </span>
              </div>
              {gemini.anomalyType && (
                <p className="text-sm text-[var(--text-secondary)]">
                  {gemini.anomalyType}
                </p>
              )}
            </div>
          </div>
        </div>

        {/* Key Observations */}
        {gemini.keyObservations && gemini.keyObservations.length > 0 && (
          <div className="p-5">
            <h4 className="text-xs font-medium text-[var(--text-muted)] uppercase tracking-wider mb-3">
              Key Observations
            </h4>
            <ul className="space-y-2">
              {gemini.keyObservations.map((obs, i) => (
                <li key={i} className="flex items-start gap-2 text-sm text-[var(--text-secondary)]">
                  <span className="text-[var(--accent-primary)] mt-0.5">•</span>
                  {obs}
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  );
}
