'use client';

import { Loader2, CheckCircle2, AlertCircle, Upload, Cpu, Brain, Sparkles } from 'lucide-react';
import { AnalysisStatus } from '@/types';

interface AnalysisProgressProps {
  status: AnalysisStatus;
  progress?: number;
  currentStep?: string;
}

const steps = [
  { id: 'uploading', label: 'Upload', icon: Upload },
  { id: 'queued', label: 'Queue', icon: Cpu },
  { id: 'processing', label: 'Analyze', icon: Brain },
  { id: 'completed', label: 'Done', icon: Sparkles },
];

export function AnalysisProgress({ status, progress = 0, currentStep }: AnalysisProgressProps) {
  const getStepStatus = (stepId: string) => {
    const statusOrder = ['idle', 'uploading', 'queued', 'processing', 'completed'];
    const currentIndex = statusOrder.indexOf(status);
    const stepIndex = statusOrder.indexOf(stepId);
    
    if (status === 'failed') return 'error';
    if (status === 'completed') return 'completed'; // All steps complete when done
    if (stepIndex < currentIndex) return 'completed';
    if (stepIndex === currentIndex) return 'active';
    return 'pending';
  };

  if (status === 'idle') return null;

  return (
    <div className="glass-card p-5 animate-slide-up">
      <div className="space-y-5">
        {/* Progress steps - minimal */}
        <div className="flex items-center justify-between">
          {steps.map((step, index) => {
            const stepStatus = getStepStatus(step.id);
            const Icon = step.icon;
            
            return (
              <div key={step.id} className="flex items-center">
                <div className="flex flex-col items-center">
                  <div className={`
                    w-9 h-9 rounded-full flex items-center justify-center transition-all duration-300
                    ${stepStatus === 'completed' ? 'bg-[var(--accent-primary)]' : ''}
                    ${stepStatus === 'active' ? 'bg-[var(--bg-tertiary)] ring-2 ring-[var(--accent-primary)]' : ''}
                    ${stepStatus === 'pending' ? 'bg-[var(--bg-tertiary)]' : ''}
                    ${stepStatus === 'error' ? 'bg-[var(--accent-danger)]' : ''}
                  `}>
                    {stepStatus === 'completed' ? (
                      <CheckCircle2 className="w-4 h-4 text-[var(--bg-primary)]" />
                    ) : stepStatus === 'active' ? (
                      <Loader2 className="w-4 h-4 text-[var(--accent-primary)] animate-spin" />
                    ) : stepStatus === 'error' ? (
                      <AlertCircle className="w-4 h-4 text-white" />
                    ) : (
                      <Icon className="w-4 h-4 text-[var(--text-muted)]" />
                    )}
                  </div>
                  <span className={`text-[10px] mt-1.5 font-medium tracking-wide uppercase ${
                    stepStatus === 'active' || stepStatus === 'completed' 
                      ? 'text-[var(--text-primary)]' 
                      : 'text-[var(--text-muted)]'
                  }`}>
                    {step.label}
                  </span>
                </div>
                {index < steps.length - 1 && (
                  <div className={`
                    w-12 h-[2px] mx-1.5 -mt-5 rounded-full transition-all duration-300
                    ${getStepStatus(steps[index + 1].id) !== 'pending' ? 'bg-[var(--accent-primary)]' : 'bg-[var(--border-color)]'}
                  `} />
                )}
              </div>
            );
          })}
        </div>

        {/* Progress bar - only during upload/processing */}
        {(status === 'uploading' || status === 'processing' || status === 'queued') && (
          <div className="space-y-2">
            <div className="flex justify-between text-xs">
              <span className="text-[var(--text-muted)]">{currentStep || 'Processing...'}</span>
              <span className="text-[var(--text-secondary)] font-mono">{Math.round(progress)}%</span>
            </div>
            <div className="progress-bar">
              <div className="progress-fill" style={{ width: `${progress}%` }} />
            </div>
          </div>
        )}

        {/* Status messages */}
        {status === 'failed' && (
          <div className="flex items-center gap-2 text-sm text-[var(--accent-danger)]">
            <AlertCircle className="w-4 h-4" />
            <span>Analysis failed. Please try again.</span>
          </div>
        )}

        {status === 'completed' && (
          <div className="flex items-center gap-2 text-sm text-[var(--accent-primary)]">
            <CheckCircle2 className="w-4 h-4" />
            <span>Complete — review results below</span>
          </div>
        )}
      </div>
    </div>
  );
}
