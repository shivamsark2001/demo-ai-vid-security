'use client';

import { useState } from 'react';
import { AnalysisResult } from '@/types';
import { Brain, Cpu, FileText, ChevronDown, ChevronUp, AlertTriangle, CheckCircle } from 'lucide-react';

interface PipelineResultsProps {
  result: AnalysisResult;
}

type Tab = 'overview' | 'prompts' | 'stage1' | 'stage2';

export function PipelineResults({ result }: PipelineResultsProps) {
  const [activeTab, setActiveTab] = useState<Tab>('overview');
  const [expandedSections, setExpandedSections] = useState<Set<string>>(new Set(['reasoning']));

  const toggleSection = (section: string) => {
    const newSet = new Set(expandedSections);
    if (newSet.has(section)) {
      newSet.delete(section);
    } else {
      newSet.add(section);
    }
    setExpandedSections(newSet);
  };

  const tabs = [
    { id: 'overview' as Tab, label: 'Overview', icon: FileText },
    { id: 'prompts' as Tab, label: 'Prompts', icon: FileText },
    { id: 'stage1' as Tab, label: 'Stage 1', icon: Cpu },
    { id: 'stage2' as Tab, label: 'Stage 2', icon: Brain },
  ];

  const edge = result.edgeDetection;
  const gemini = result.geminiVerification;
  const prompts = result.promptBanks;

  return (
    <div className="space-y-4">
      {/* Annotated Grid */}
      {result.annotatedGridB64 && (
        <div className="glass-card p-4">
          <h3 className="text-xs font-medium text-[var(--text-muted)] uppercase tracking-wider mb-3">
            Frame Analysis Grid
          </h3>
          <img 
            src={result.annotatedGridB64} 
            alt="Annotated frame grid" 
            className="w-full rounded-lg"
          />
          <p className="text-[10px] text-[var(--text-muted)] mt-2">
            {result.frameCount} frames analyzed • Colors indicate anomaly likelihood
          </p>
        </div>
      )}

      {/* Tabs */}
      <div className="glass-card overflow-hidden">
        <div className="flex border-b border-[var(--border-color)]">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex-1 flex items-center justify-center gap-1.5 py-3 text-xs font-medium transition-colors ${
                  activeTab === tab.id
                    ? 'text-[var(--accent-primary)] border-b-2 border-[var(--accent-primary)] bg-[var(--accent-primary)]/5'
                    : 'text-[var(--text-muted)] hover:text-[var(--text-secondary)]'
                }`}
              >
                <Icon className="w-3.5 h-3.5" />
                {tab.label}
              </button>
            );
          })}
        </div>

        <div className="p-4">
          {/* Overview Tab */}
          {activeTab === 'overview' && gemini && (
            <div className="space-y-4">
              {/* Verdict */}
              <div className={`p-4 rounded-lg ${
                gemini.isAnomaly 
                  ? 'bg-[var(--accent-danger)]/10 border border-[var(--accent-danger)]/30' 
                  : 'bg-[var(--accent-primary)]/10 border border-[var(--accent-primary)]/30'
              }`}>
                <div className="flex items-center gap-2 mb-2">
                  {gemini.isAnomaly ? (
                    <AlertTriangle className="w-5 h-5 text-[var(--accent-danger)]" />
                  ) : (
                    <CheckCircle className="w-5 h-5 text-[var(--accent-primary)]" />
                  )}
                  <span className={`font-semibold ${
                    gemini.isAnomaly ? 'text-[var(--accent-danger)]' : 'text-[var(--accent-primary)]'
                  }`}>
                    {gemini.isAnomaly ? 'Anomaly Detected' : 'Normal Activity'}
                  </span>
                  <span className="ml-auto text-sm font-mono">
                    {Math.round(gemini.confidence * 100)}% confidence
                  </span>
                </div>
                {gemini.anomalyType && (
                  <p className="text-sm text-[var(--text-secondary)]">
                    Type: <span className="font-medium">{gemini.anomalyType}</span>
                  </p>
                )}
              </div>

              {/* Key Observations */}
              {gemini.keyObservations && gemini.keyObservations.length > 0 && (
                <div>
                  <h4 className="text-xs font-medium text-[var(--text-muted)] uppercase tracking-wider mb-2">
                    Key Observations
                  </h4>
                  <ul className="space-y-1.5">
                    {gemini.keyObservations.map((obs, i) => (
                      <li key={i} className="flex items-start gap-2 text-sm text-[var(--text-secondary)]">
                        <span className="text-[var(--accent-primary)] mt-1">•</span>
                        {obs}
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Reasoning */}
              <div>
                <button
                  onClick={() => toggleSection('reasoning')}
                  className="flex items-center gap-2 text-xs font-medium text-[var(--text-muted)] uppercase tracking-wider mb-2 hover:text-[var(--text-secondary)]"
                >
                  {expandedSections.has('reasoning') ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
                  Full Reasoning
                </button>
                {expandedSections.has('reasoning') && (
                  <p className="text-sm text-[var(--text-secondary)] leading-relaxed bg-[var(--bg-tertiary)] p-3 rounded-lg">
                    {gemini.reasoning || 'No detailed reasoning provided.'}
                  </p>
                )}
              </div>
            </div>
          )}

          {/* Prompts Tab */}
          {activeTab === 'prompts' && prompts && (
            <div className="space-y-4">
              {prompts.detectionSummary && (
                <div className="p-3 bg-[var(--bg-tertiary)] rounded-lg">
                  <p className="text-sm text-[var(--text-secondary)]">{prompts.detectionSummary}</p>
                </div>
              )}
              
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <h4 className="text-xs font-medium text-[var(--accent-primary)] uppercase tracking-wider mb-2">
                    Normal ({prompts.normalPrompts.length})
                  </h4>
                  <div className="space-y-1 max-h-48 overflow-y-auto">
                    {prompts.normalPrompts.slice(0, 10).map((p, i) => (
                      <p key={i} className="text-xs text-[var(--text-muted)] py-1 px-2 bg-[var(--bg-tertiary)] rounded">
                        {p}
                      </p>
                    ))}
                    {prompts.normalPrompts.length > 10 && (
                      <p className="text-[10px] text-[var(--text-muted)] italic">
                        +{prompts.normalPrompts.length - 10} more
                      </p>
                    )}
                  </div>
                </div>
                <div>
                  <h4 className="text-xs font-medium text-[var(--accent-danger)] uppercase tracking-wider mb-2">
                    Anomaly ({prompts.anomalyPrompts.length})
                  </h4>
                  <div className="space-y-1 max-h-48 overflow-y-auto">
                    {prompts.anomalyPrompts.slice(0, 10).map((p, i) => (
                      <p key={i} className="text-xs text-[var(--text-muted)] py-1 px-2 bg-[var(--bg-tertiary)] rounded">
                        {p}
                      </p>
                    ))}
                    {prompts.anomalyPrompts.length > 10 && (
                      <p className="text-[10px] text-[var(--text-muted)] italic">
                        +{prompts.anomalyPrompts.length - 10} more
                      </p>
                    )}
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Stage 1 Tab */}
          {activeTab === 'stage1' && edge && (
            <div className="space-y-4">
              <div className={`p-3 rounded-lg ${
                edge.isAnomaly 
                  ? 'bg-[var(--accent-warning)]/10 border border-[var(--accent-warning)]/30' 
                  : 'bg-[var(--accent-primary)]/10 border border-[var(--accent-primary)]/30'
              }`}>
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium">
                    SigLIP2 Assessment: {edge.isAnomaly ? '⚠️ Anomaly Likely' : '✓ Normal'}
                  </span>
                  <span className="text-xs font-mono">
                    Score diff: {edge.scoreDiff > 0 ? '+' : ''}{edge.scoreDiff.toFixed(4)}
                  </span>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-3 text-sm">
                <div className="p-2 bg-[var(--bg-tertiary)] rounded">
                  <p className="text-[10px] text-[var(--text-muted)] uppercase">Anomaly Score</p>
                  <p className="font-mono">{edge.anomalyScore.toFixed(4)}</p>
                </div>
                <div className="p-2 bg-[var(--bg-tertiary)] rounded">
                  <p className="text-[10px] text-[var(--text-muted)] uppercase">Normal Score</p>
                  <p className="font-mono">{edge.normalScore.toFixed(4)}</p>
                </div>
              </div>

              <div>
                <h4 className="text-xs font-medium text-[var(--text-muted)] uppercase tracking-wider mb-2">
                  Best Matching Prompts
                </h4>
                <div className="space-y-2">
                  <div className="p-2 bg-[var(--accent-danger)]/10 rounded text-sm">
                    <span className="text-[10px] text-[var(--accent-danger)]">ANOMALY:</span>
                    <p className="text-[var(--text-secondary)]">{edge.topAnomalyPrompt}</p>
                  </div>
                  <div className="p-2 bg-[var(--accent-primary)]/10 rounded text-sm">
                    <span className="text-[10px] text-[var(--accent-primary)]">NORMAL:</span>
                    <p className="text-[var(--text-secondary)]">{edge.topNormalPrompt}</p>
                  </div>
                </div>
              </div>

              {/* Per-frame scores */}
              <div>
                <h4 className="text-xs font-medium text-[var(--text-muted)] uppercase tracking-wider mb-2">
                  Per-Frame Scores
                </h4>
                <div className="flex flex-wrap gap-1">
                  {edge.perFrameScores.map((score, i) => (
                    <div
                      key={i}
                      className={`px-2 py-1 rounded text-[10px] font-mono ${
                        score > 0.015
                          ? 'bg-[var(--accent-danger)]/20 text-[var(--accent-danger)]'
                          : score < -0.015
                          ? 'bg-[var(--accent-primary)]/20 text-[var(--accent-primary)]'
                          : 'bg-[var(--bg-tertiary)] text-[var(--text-muted)]'
                      }`}
                    >
                      F{i + 1}: {score > 0 ? '+' : ''}{score.toFixed(3)}
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* Stage 2 Tab */}
          {activeTab === 'stage2' && gemini && (
            <div className="space-y-4">
              <div className={`p-3 rounded-lg ${
                gemini.isAnomaly 
                  ? 'bg-[var(--accent-danger)]/10 border border-[var(--accent-danger)]/30' 
                  : 'bg-[var(--accent-primary)]/10 border border-[var(--accent-primary)]/30'
              }`}>
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium">
                    Gemini Verdict: {gemini.isAnomaly ? '🚨 Anomaly Confirmed' : '✓ Normal Confirmed'}
                  </span>
                  <span className="text-xs font-mono">
                    {Math.round(gemini.confidence * 100)}% confidence
                  </span>
                </div>
                {gemini.anomalyType && (
                  <p className="text-xs text-[var(--text-secondary)] mt-1">
                    Type: {gemini.anomalyType}
                  </p>
                )}
              </div>

              {gemini.frameAnalysis && (
                <div>
                  <h4 className="text-xs font-medium text-[var(--text-muted)] uppercase tracking-wider mb-2">
                    Frame Analysis
                  </h4>
                  <p className="text-sm text-[var(--text-secondary)] bg-[var(--bg-tertiary)] p-3 rounded-lg">
                    {gemini.frameAnalysis}
                  </p>
                </div>
              )}

              <div>
                <h4 className="text-xs font-medium text-[var(--text-muted)] uppercase tracking-wider mb-2">
                  Full Reasoning
                </h4>
                <p className="text-sm text-[var(--text-secondary)] bg-[var(--bg-tertiary)] p-3 rounded-lg leading-relaxed">
                  {gemini.reasoning}
                </p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
