'use client';

import { TimelineEvent } from '@/types';

interface TimelineProps {
  events: TimelineEvent[];
  duration: number;
  currentTime: number;
  onEventClick: (event: TimelineEvent) => void;
}

export function Timeline({ events, duration, currentTime, onEventClick }: TimelineProps) {
  if (duration === 0) return null;

  const getEventPosition = (time: number) => (time / duration) * 100;
  const getEventWidth = (t0: number, t1: number) => ((t1 - t0) / duration) * 100;

  return (
    <div className="timeline-container">
      {/* Time markers */}
      <div className="flex justify-between text-xs text-[var(--text-muted)] mb-2">
        <span>0:00</span>
        <span>{formatTime(duration / 4)}</span>
        <span>{formatTime(duration / 2)}</span>
        <span>{formatTime((duration * 3) / 4)}</span>
        <span>{formatTime(duration)}</span>
      </div>
      
      {/* Track */}
      <div className="timeline-track">
        {/* Events */}
        {events.map((event) => (
          <div
            key={event.id}
            className={`timeline-event ${event.severity}`}
            style={{
              left: `${getEventPosition(event.t0)}%`,
              width: `${Math.max(getEventWidth(event.t0, event.t1), 1)}%`,
            }}
            onClick={() => onEventClick(event)}
            title={`${event.label} (${formatTime(event.t0)} - ${formatTime(event.t1)})`}
          />
        ))}
        
        {/* Current time indicator */}
        <div
          className="absolute top-0 bottom-0 w-0.5 bg-white z-20 transition-all duration-100"
          style={{ left: `${getEventPosition(currentTime)}%` }}
        >
          <div className="absolute -top-1 left-1/2 -translate-x-1/2 w-2 h-2 bg-white rounded-full" />
        </div>
      </div>

      {/* Legend */}
      <div className="flex items-center gap-4 mt-3 text-xs">
        <div className="flex items-center gap-1.5">
          <div className="w-3 h-3 rounded bg-[var(--accent-danger)]" />
          <span className="text-[var(--text-secondary)]">High Risk</span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="w-3 h-3 rounded bg-[var(--accent-warning)]" />
          <span className="text-[var(--text-secondary)]">Medium Risk</span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="w-3 h-3 rounded bg-[var(--accent-primary)]" />
          <span className="text-[var(--text-secondary)]">Low Risk</span>
        </div>
      </div>
    </div>
  );
}

function formatTime(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, '0')}`;
}
