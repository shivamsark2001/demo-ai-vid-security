'use client';

import { Camera, Target } from 'lucide-react';

interface ContextInputProps {
  cameraContext: string;
  detectionTargets: string;
  onCameraContextChange: (value: string) => void;
  onDetectionTargetsChange: (value: string) => void;
  disabled?: boolean;
}

export function ContextInput({ 
  cameraContext, 
  detectionTargets, 
  onCameraContextChange, 
  onDetectionTargetsChange, 
  disabled 
}: ContextInputProps) {
  return (
    <div className="space-y-5">
      {/* Camera Context */}
      <div className="space-y-2">
        <div className="flex items-center gap-2">
          <Camera className="w-4 h-4 text-[var(--accent-secondary)]" />
          <label className="text-sm font-medium text-[var(--text-secondary)]">
            Camera Context
          </label>
        </div>
        <textarea
          className="input-field min-h-[80px]"
          placeholder="Where is the camera? What does it normally see?&#10;&#10;Example: Industrial site surveillance camera overlooking equipment yard. Normal activity includes workers, trucks loading/unloading, forklifts operating."
          value={cameraContext}
          onChange={(e) => onCameraContextChange(e.target.value)}
          disabled={disabled}
        />
        <p className="text-xs text-[var(--text-muted)]">
          Describe the camera location and typical scene activity
        </p>
      </div>

      {/* Detection Targets */}
      <div className="space-y-2">
        <div className="flex items-center gap-2">
          <Target className="w-4 h-4 text-[var(--accent-primary)]" />
          <label className="text-sm font-medium text-[var(--text-secondary)]">
            Detection Targets
          </label>
        </div>
        <textarea
          className="input-field min-h-[80px]"
          placeholder="What anomalies should be detected?&#10;&#10;Example: Fire, smoke, explosions, equipment damage, unauthorized personnel, safety violations"
          value={detectionTargets}
          onChange={(e) => onDetectionTargetsChange(e.target.value)}
          disabled={disabled}
        />
        <p className="text-xs text-[var(--text-muted)]">
          List specific threats or behaviors to detect
        </p>
      </div>
    </div>
  );
}
