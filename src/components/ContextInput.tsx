'use client';

import { MessageSquare } from 'lucide-react';

interface ContextInputProps {
  value: string;
  onChange: (value: string) => void;
  disabled?: boolean;
}

export function ContextInput({ value, onChange, disabled }: ContextInputProps) {
  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2">
        <MessageSquare className="w-4 h-4 text-[var(--accent-secondary)]" />
        <label className="text-sm font-medium text-[var(--text-secondary)]">
          Analysis Context
        </label>
      </div>
      <textarea
        className="input-field min-h-[120px]"
        placeholder="Describe what to look for...&#10;&#10;Examples:&#10;• Detect unauthorized access or trespassing&#10;• Monitor for suspicious loitering near entrance&#10;• Identify aggressive behavior or physical altercations&#10;• Track package theft or delivery tampering"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        disabled={disabled}
      />
      <p className="text-xs text-[var(--text-muted)]">
        Be specific about the threats or behaviors you want to detect. This helps the AI focus on relevant events.
      </p>
    </div>
  );
}
