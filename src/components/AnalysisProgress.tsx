'use client';

import { Loader2, CheckCircle2, AlertCircle, Upload, Cpu, Brain, Sparkles } from 'lucide-react';
import { AnalysisStatus } from '@/types';

interface AnalysisProgressProps {
  status: AnalysisStatus;
  progress?: number;
  currentStep?: string;
}

const steps = [
  { id: 'uploading', label: 'Uploading Video', icon: Upload },
  { id: 'queued', label: 'Queued for Processing', icon: Cpu },
  { id: 'processing', label: 'AI Analysis', icon: Brain },
  { id: 'completed', label: 'Analysis Complete', icon: Sparkles },
];

export function AnalysisProgress({ status, progress = 0, currentStep }: AnalysisProgressProps) {
  const getStepStatus = (stepId: string) => {
    const statusOrder = ['idle', 'uploading', 'queued', 'processing', 'completed'];
    const currentIndex = statusOrder.indexOf(status);
    const stepIndex = statusOrder.indexOf(stepId);
    
    if (status === 'failed') return 'error';
    if (stepIndex < currentIndex) return 'completed';
    if (stepIndex === currentIndex) return 'active';
    return 'pending';
  };

  if (status === 'idle') return null;

  return (
    <div className="glass-card p-6 animate-slide-up">
      <div className="space-y-6">
        {/* Progress steps */}
        <div className="flex items-center justify-between">
          {steps.map((step, index) => {
            const stepStatus = getStepStatus(step.id);
            const Icon = step.icon;
            
            return (
              <div key={step.id} className="flex items-center">
                <div className="flex flex-col items-center">
                  <div className={`
                    w-10 h-10 rounded-full flex items-center justify-center transition-all
                    ${stepStatus === 'completed' ? 'bg-[var(--accent-primary)] text-[var(--bg-primary)]' : ''}
                    ${stepStatus === 'active' ? 'bg-[var(--accent-secondary)] text-white' : ''}
                    ${stepStatus === 'pending' ? 'bg-[var(--bg-tertiary)] text-[var(--text-muted)]' : ''}
                    ${stepStatus === 'error' ? 'bg-[var(--accent-danger)] text-white' : ''}
                  `}>
                    {stepStatus === 'completed' ? (
                      <CheckCircle2 className="w-5 h-5" />
                    ) : stepStatus === 'active' ? (
                      <Loader2 className="w-5 h-5 animate-spin" />
                    ) : stepStatus === 'error' ? (
                      <AlertCircle className="w-5 h-5" />
                    ) : (
                      <Icon className="w-5 h-5" />
                    )}
                  </div>
                  <span className={`text-xs mt-2 ${stepStatus === 'active' ? 'text-[var(--text-primary)]' : 'text-[var(--text-muted)]'}`}>
                    {step.label}
                  </span>
                </div>
                {index < steps.length - 1 && (
                  <div className={`
                    w-16 h-0.5 mx-2 -mt-6
                    ${getStepStatus(steps[index + 1].id) !== 'pending' ? 'bg-[var(--accent-primary)]' : 'bg-[var(--border-color)]'}
                  `} />
                )}
              </div>
            );
          })}
        </div>

        {/* Progress bar for active step */}
        {(status === 'uploading' || status === 'processing') && (
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span className="text-[var(--text-secondary)]">{currentStep || 'Processing...'}</span>
              <span className="text-[var(--accent-primary)] font-medium">{Math.round(progress)}%</span>
            </div>
            <div className="progress-bar">
              <div className="progress-fill" style={{ width: `${progress}%` }} />
            </div>
          </div>
        )}

        {/* Status message */}
        {status === 'failed' && (
          <div className="flex items-center gap-3 p-4 rounded-lg bg-[var(--accent-danger)]/10 border border-[var(--accent-danger)]/30">
            <AlertCircle className="w-5 h-5 text-[var(--accent-danger)]" />
            <span className="text-[var(--accent-danger)]">Analysis failed. Please try again.</span>
          </div>
        )}

        {status === 'completed' && (
          <div className="flex items-center gap-3 p-4 rounded-lg bg-[var(--accent-primary)]/10 border border-[var(--accent-primary)]/30">
            <CheckCircle2 className="w-5 h-5 text-[var(--accent-primary)]" />
            <span className="text-[var(--accent-primary)]">Analysis complete! Review detected events below.</span>
          </div>
        )}
      </div>
    </div>
  );
}
