'use client';

import { AlertTriangle, CheckCircle, Eye, Clock, Activity, Flame } from 'lucide-react';

interface HysteresisEvent {
  start: number;
  end: number;
  duration: number;
  peakScore: number;
  avgScore: number;
}

interface HysteresisVerdict {
  isAnomaly: boolean;
  numEvents?: number;
  totalAnomalyDuration?: number;
  anomalyPercentage?: number;
  peakScore?: number;
}

interface GeminiVerification {
  isAnomaly: boolean;
  anomalyType?: string | null;
  confidence: number;
  reasoning?: string;
  keyObservations?: string[];
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
interface PipelineResultsProps {
  result: any;
}

function formatTime(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return mins > 0 ? `${mins}m ${secs}s` : `${secs}s`;
}

export function PipelineResults({ result }: PipelineResultsProps) {
  // Handle HYSTERESIS mode
  if (result.mode === 'hysteresis' && result.finalVerdict) {
    const verdict = result.finalVerdict;
    const events = result.events || [];
    const isAnomaly = verdict.isAnomaly;
    const gemini = result.geminiVerification as GeminiVerification | null;

    return (
      <div className="space-y-4">
        {/* Main Verdict Card */}
        <div className="glass-card overflow-hidden">
          <div className={`p-5 ${
            isAnomaly 
              ? 'bg-gradient-to-r from-[var(--accent-danger)]/20 to-transparent' 
              : 'bg-gradient-to-r from-[var(--accent-primary)]/20 to-transparent'
          }`}>
            <div className="flex items-start gap-4">
              <div className={`w-14 h-14 rounded-xl flex items-center justify-center ${
                isAnomaly 
                  ? 'bg-[var(--accent-danger)]/20' 
                  : 'bg-[var(--accent-primary)]/20'
              }`}>
                {isAnomaly ? (
                  <Flame className="w-7 h-7 text-[var(--accent-danger)]" />
                ) : (
                  <CheckCircle className="w-7 h-7 text-[var(--accent-primary)]" />
                )}
              </div>
              <div className="flex-1">
                <h3 className={`text-xl font-bold ${
                  isAnomaly ? 'text-[var(--accent-danger)]' : 'text-[var(--accent-primary)]'
                }`}>
                  {isAnomaly ? '🚨 Anomaly Detected' : '✅ Normal Activity'}
                </h3>
                {/* Show anomaly type from verdict or gemini */}
                {isAnomaly && (verdict.anomalyType || gemini?.anomalyType) && (
                  <p className="text-base font-medium text-[var(--text-primary)] mt-1">
                    {verdict.anomalyType || gemini?.anomalyType}
                  </p>
                )}
                {isAnomaly && verdict.anomalyPercentage !== undefined && (
                  <p className="text-sm text-[var(--text-secondary)] mt-1">
                    {verdict.anomalyPercentage.toFixed(1)}% of video flagged
                  </p>
                )}
              </div>
            </div>
          </div>

          {/* Stats Grid */}
          {isAnomaly && (
            <div className="grid grid-cols-3 gap-4 p-5 border-t border-white/5">
              <div className="text-center">
                <div className="text-2xl font-bold text-[var(--accent-danger)]">
                  {verdict.numEvents || 0}
                </div>
                <div className="text-xs text-[var(--text-muted)]">Events</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-[var(--text-primary)]">
                  {formatTime(verdict.totalAnomalyDuration || 0)}
                </div>
                <div className="text-xs text-[var(--text-muted)]">Duration</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-orange-400">
                  {((verdict.peakScore || 0) * 100).toFixed(1)}
                </div>
                <div className="text-xs text-[var(--text-muted)]">Peak Score</div>
              </div>
            </div>
          )}
        </div>

        {/* AI Reasoning (from Gemini) */}
        {gemini?.reasoning && (
          <div className="glass-card p-5">
            <div className="flex items-center gap-2 mb-3">
              <Activity className="w-4 h-4 text-[var(--accent-danger)]" />
              <span className="text-xs font-medium text-[var(--text-muted)] uppercase tracking-wider">
                AI Analysis
              </span>
              {gemini.confidence !== undefined && (
                <span className={`ml-auto px-2 py-0.5 rounded-full text-xs font-medium ${
                  isAnomaly 
                    ? 'bg-[var(--accent-danger)]/20 text-[var(--accent-danger)]' 
                    : 'bg-[var(--accent-primary)]/20 text-[var(--accent-primary)]'
                }`}>
                  {Math.round(gemini.confidence * 100)}% confidence
                </span>
              )}
            </div>
            <p className="text-sm text-[var(--text-secondary)] leading-relaxed">
              {gemini.reasoning}
            </p>
          </div>
        )}

        {/* Key Observations (from Gemini) */}
        {gemini?.keyObservations && gemini.keyObservations.length > 0 && (
          <div className="glass-card p-5">
            <div className="flex items-center gap-2 mb-3">
              <Eye className="w-4 h-4 text-[var(--accent-primary)]" />
              <span className="text-xs font-medium text-[var(--text-muted)] uppercase tracking-wider">
                Key Observations
              </span>
            </div>
            <ul className="space-y-2">
              {gemini.keyObservations.map((obs: string, i: number) => (
                <li key={i} className="flex items-start gap-2 text-sm text-[var(--text-secondary)]">
                  <span className="text-[var(--accent-primary)] mt-0.5">•</span>
                  {obs}
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* Events Timeline */}
        {events.length > 0 && (
          <div className="glass-card p-5">
            <div className="flex items-center gap-2 mb-4">
              <Clock className="w-4 h-4 text-[var(--accent-primary)]" />
              <span className="text-xs font-medium text-[var(--text-muted)] uppercase tracking-wider">
                Detected Events
              </span>
            </div>
            <div className="space-y-3">
              {events.map((event: HysteresisEvent, i: number) => (
                <div key={i} className="flex items-center gap-4 p-3 rounded-lg bg-[var(--bg-tertiary)]">
                  <div className="w-10 h-10 rounded-lg bg-[var(--accent-danger)]/20 flex items-center justify-center">
                    <AlertTriangle className="w-5 h-5 text-[var(--accent-danger)]" />
                  </div>
                  <div className="flex-1">
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-medium text-[var(--text-primary)]">
                        Event {i + 1}
                      </span>
                      <span className="text-xs text-[var(--text-muted)]">
                        {formatTime(event.start)} → {formatTime(event.end)}
                      </span>
                    </div>
                    <div className="text-xs text-[var(--text-secondary)] mt-0.5">
                      Duration: {formatTime(event.duration)} • Peak: {(event.peakScore * 100).toFixed(1)}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Scene Analysis */}
        {result.promptBanks?.sceneAnalysis && (
          <div className="glass-card p-5">
            <div className="flex items-center gap-2 mb-3">
              <Eye className="w-4 h-4 text-[var(--accent-primary)]" />
              <span className="text-xs font-medium text-[var(--text-muted)] uppercase tracking-wider">
                Scene Analysis
              </span>
            </div>
            <p className="text-sm text-[var(--text-secondary)] leading-relaxed">
              {result.promptBanks.sceneAnalysis}
            </p>
          </div>
        )}

        {/* Video Info */}
        <div className="glass-card p-4">
          <div className="flex items-center justify-between text-xs text-[var(--text-muted)]">
            <span>Mode: Hysteresis</span>
            <span>Duration: {formatTime(result.videoDuration || 0)}</span>
          </div>
        </div>
      </div>
    );
  }

  // Handle BATCH mode (original behavior)
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

        {/* Reasoning */}
        {gemini.reasoning && (
          <div className="p-5 border-t border-white/5">
            <h4 className="text-xs font-medium text-[var(--text-muted)] uppercase tracking-wider mb-3">
              Analysis Reasoning
            </h4>
            <p className="text-sm text-[var(--text-secondary)] leading-relaxed">
              {gemini.reasoning}
            </p>
          </div>
        )}

        {/* Key Observations (if available) */}
        {gemini.keyObservations && gemini.keyObservations.length > 0 && (
          <div className="p-5 border-t border-white/5">
            <h4 className="text-xs font-medium text-[var(--text-muted)] uppercase tracking-wider mb-3">
              Key Observations
            </h4>
            <ul className="space-y-2">
              {gemini.keyObservations.map((obs: string, i: number) => (
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
