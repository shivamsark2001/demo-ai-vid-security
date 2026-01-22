'use client';

import { useState } from 'react';
import { AnalysisResult } from '@/types';
import { AlertTriangle, CheckCircle, ChevronDown, ChevronUp, Eye } from 'lucide-react';

interface PipelineResultsProps {
  result: AnalysisResult;
}

export function PipelineResults({ result }: PipelineResultsProps) {
  const [showDetails, setShowDetails] = useState(false);

  const gemini = result.geminiVerification;
  const edge = result.edgeDetection;

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
            ? 'bg-gradient-to-r from-[var(--accent-danger)]/20 to-transparent border-b border-[var(--accent-danger)]/20' 
            : 'bg-gradient-to-r from-[var(--accent-primary)]/20 to-transparent border-b border-[var(--accent-primary)]/20'
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
          <div className="p-5 border-b border-[var(--border-color)]">
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

        {/* Reasoning */}
        <div className="p-5">
          <button
            onClick={() => setShowDetails(!showDetails)}
            className="flex items-center gap-2 text-xs font-medium text-[var(--text-muted)] uppercase tracking-wider hover:text-[var(--text-secondary)] transition-colors w-full"
          >
            {showDetails ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
            Analysis Details
          </button>
          
          {showDetails && (
            <div className="mt-4 space-y-4">
              {/* Reasoning */}
              <div>
                <p className="text-xs text-[var(--text-muted)] mb-2">AI Reasoning</p>
                <p className="text-sm text-[var(--text-secondary)] leading-relaxed bg-[var(--bg-tertiary)] p-4 rounded-lg">
                  {gemini.reasoning || 'No detailed reasoning provided.'}
                </p>
              </div>

              {/* Frame Analysis */}
              {gemini.frameAnalysis && (
                <div>
                  <p className="text-xs text-[var(--text-muted)] mb-2">Frame Analysis</p>
                  <p className="text-sm text-[var(--text-secondary)] leading-relaxed bg-[var(--bg-tertiary)] p-4 rounded-lg">
                    {gemini.frameAnalysis}
                  </p>
                </div>
              )}

              {/* Edge Detection Stats */}
              {edge && (
                <div>
                  <p className="text-xs text-[var(--text-muted)] mb-2">Detection Scores</p>
                  <div className="grid grid-cols-2 gap-2">
                    <div className="bg-[var(--bg-tertiary)] p-3 rounded-lg">
                      <p className="text-[10px] text-[var(--text-muted)] uppercase">Best Match</p>
                      <p className="text-xs text-[var(--text-secondary)] truncate">{edge.topAnomalyPrompt}</p>
                    </div>
                    <div className="bg-[var(--bg-tertiary)] p-3 rounded-lg">
                      <p className="text-[10px] text-[var(--text-muted)] uppercase">Score Diff</p>
                      <p className="text-sm font-mono">{edge.scoreDiff > 0 ? '+' : ''}{edge.scoreDiff.toFixed(4)}</p>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
