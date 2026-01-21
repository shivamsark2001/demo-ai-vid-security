'use client';

import { TimelineEvent } from '@/types';
import { Clock, AlertTriangle, CheckCircle, XCircle, ChevronRight } from 'lucide-react';

interface EventListProps {
  events: TimelineEvent[];
  onEventClick: (event: TimelineEvent) => void;
  selectedEventId?: string;
}

export function EventList({ events, onEventClick, selectedEventId }: EventListProps) {
  if (events.length === 0) {
    return (
      <div className="text-center py-12 text-[var(--text-secondary)]">
        <CheckCircle className="w-12 h-12 mx-auto mb-4 text-[var(--accent-primary)] opacity-50" />
        <p>No security events detected</p>
        <p className="text-sm text-[var(--text-muted)] mt-1">The video appears safe</p>
      </div>
    );
  }

  const sortedEvents = [...events].sort((a, b) => {
    // Sort by severity first (high > medium > low), then by time
    const severityOrder = { high: 0, medium: 1, low: 2 };
    if (severityOrder[a.severity] !== severityOrder[b.severity]) {
      return severityOrder[a.severity] - severityOrder[b.severity];
    }
    return a.t0 - b.t0;
  });

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between mb-4">
        <h3 className="font-semibold text-[var(--text-primary)]">
          Detected Events
        </h3>
        <span className="text-sm text-[var(--text-muted)]">
          {events.length} event{events.length !== 1 ? 's' : ''}
        </span>
      </div>
      
      {sortedEvents.map((event, index) => (
        <div
          key={event.id}
          className={`event-card ${event.severity} ${selectedEventId === event.id ? 'ring-2 ring-[var(--accent-primary)]' : ''} animate-slide-up`}
          style={{ animationDelay: `${index * 0.1}s` }}
          onClick={() => onEventClick(event)}
        >
          <div className="flex items-start justify-between gap-4">
            <div className="flex-1 min-w-0">
              {/* Header */}
              <div className="flex items-center gap-2 mb-2">
                <SeverityIcon severity={event.severity} />
                <span className="font-semibold text-[var(--text-primary)] truncate">
                  {event.label}
                </span>
              </div>
              
              {/* Timestamp */}
              <div className="flex items-center gap-1.5 text-sm text-[var(--text-secondary)] mb-2">
                <Clock className="w-3.5 h-3.5" />
                <span>{formatTime(event.t0)} - {formatTime(event.t1)}</span>
                <span className="text-[var(--text-muted)]">
                  ({Math.round(event.t1 - event.t0)}s)
                </span>
              </div>
              
              {/* Gemini explanation */}
              <p className="text-sm text-[var(--text-secondary)] line-clamp-2">
                {event.geminiExplanation}
              </p>
              
              {/* Confidence meters */}
              <div className="flex items-center gap-4 mt-3">
                <div className="flex-1">
                  <div className="flex items-center justify-between text-xs mb-1">
                    <span className="text-[var(--text-muted)]">Detection</span>
                    <span className="text-[var(--text-secondary)]">{Math.round(event.score * 100)}%</span>
                  </div>
                  <div className="confidence-meter">
                    <div 
                      className={`confidence-fill ${event.severity}`}
                      style={{ width: `${event.score * 100}%` }}
                    />
                  </div>
                </div>
                <div className="flex-1">
                  <div className="flex items-center justify-between text-xs mb-1">
                    <span className="text-[var(--text-muted)]">AI Confidence</span>
                    <span className="text-[var(--text-secondary)]">{Math.round(event.geminiConfidence * 100)}%</span>
                  </div>
                  <div className="confidence-meter">
                    <div 
                      className={`confidence-fill ${event.severity}`}
                      style={{ width: `${event.geminiConfidence * 100}%` }}
                    />
                  </div>
                </div>
              </div>
            </div>
            
            {/* Verdict badge */}
            <div className="flex flex-col items-end gap-2">
              <div className={`
                px-2 py-1 rounded text-xs font-medium
                ${event.geminiVerdict 
                  ? 'bg-[var(--accent-danger)]/20 text-[var(--accent-danger)]' 
                  : 'bg-[var(--accent-primary)]/20 text-[var(--accent-primary)]'
                }
              `}>
                {event.geminiVerdict ? 'Confirmed' : 'Unconfirmed'}
              </div>
              <ChevronRight className="w-5 h-5 text-[var(--text-muted)]" />
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}

function SeverityIcon({ severity }: { severity: 'high' | 'medium' | 'low' }) {
  switch (severity) {
    case 'high':
      return <XCircle className="w-4 h-4 text-[var(--accent-danger)]" />;
    case 'medium':
      return <AlertTriangle className="w-4 h-4 text-[var(--accent-warning)]" />;
    case 'low':
      return <CheckCircle className="w-4 h-4 text-[var(--accent-primary)]" />;
  }
}

function formatTime(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, '0')}`;
}
