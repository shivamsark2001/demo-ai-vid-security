'use client';

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
    <div className="space-y-4">
      <div className="space-y-1.5">
        <label className="text-xs font-medium text-[var(--text-muted)] uppercase tracking-wider">
          Scene Context
        </label>
        <textarea
          className="input-field min-h-[70px]"
          placeholder="Describe the camera location and normal activity..."
          value={cameraContext}
          onChange={(e) => onCameraContextChange(e.target.value)}
          disabled={disabled}
        />
      </div>

      <div className="space-y-1.5">
        <label className="text-xs font-medium text-[var(--text-muted)] uppercase tracking-wider">
          Detect
        </label>
        <textarea
          className="input-field min-h-[70px]"
          placeholder="What should be flagged? e.g. fire, smoke, intrusion..."
          value={detectionTargets}
          onChange={(e) => onDetectionTargetsChange(e.target.value)}
          disabled={disabled}
        />
      </div>
    </div>
  );
}
