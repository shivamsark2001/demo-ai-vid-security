'use client';

import { useState } from 'react';
import { TimelineEvent } from '@/types';
import { Clock, AlertTriangle, CheckCircle, XCircle, ChevronDown, ChevronUp } from 'lucide-react';

interface EventListProps {
  events: TimelineEvent[];
  onEventClick: (event: TimelineEvent) => void;
  selectedEventId?: string;
}

export function EventList({ events, onEventClick, selectedEventId }: EventListProps) {
  const [expandedId, setExpandedId] = useState<string | null>(null);

  if (events.length === 0) {
    return (
      <div className="text-center py-10 text-[var(--text-secondary)]">
        <CheckCircle className="w-10 h-10 mx-auto mb-3 text-[var(--accent-primary)] opacity-50" />
        <p className="text-sm">No events detected</p>
        <p className="text-xs text-[var(--text-muted)] mt-1">Video appears normal</p>
      </div>
    );
  }

  const sortedEvents = [...events].sort((a, b) => {
    const severityOrder = { high: 0, medium: 1, low: 2 };
    if (severityOrder[a.severity] !== severityOrder[b.severity]) {
      return severityOrder[a.severity] - severityOrder[b.severity];
    }
    return a.t0 - b.t0;
  });

  const toggleExpand = (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    setExpandedId(expandedId === id ? null : id);
  };

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <h3 className="text-xs font-medium text-[var(--text-muted)] uppercase tracking-wider">
          Events
        </h3>
        <span className="text-xs text-[var(--text-muted)]">
          {events.length} found
        </span>
      </div>
      
      {sortedEvents.map((event, index) => {
        const isExpanded = expandedId === event.id;
        const isSelected = selectedEventId === event.id;
        
        return (
          <div
            key={event.id}
            className={`event-card ${event.severity} ${isSelected ? 'selected' : ''} animate-slide-up`}
            style={{ animationDelay: `${index * 0.05}s` }}
            onClick={() => onEventClick(event)}
          >
            {/* Header row */}
            <div className="flex items-start justify-between gap-3">
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 mb-1">
                  <SeverityIcon severity={event.severity} />
                  <span className="font-medium text-sm text-[var(--text-primary)] truncate">
                    {event.label}
                  </span>
                </div>
                
                <div className="flex items-center gap-1.5 text-xs text-[var(--text-muted)]">
                  <Clock className="w-3 h-3" />
                  <span>{formatTime(event.t0)} - {formatTime(event.t1)}</span>
                </div>
              </div>
              
              <div className="flex items-center gap-2">
                <span className={`
                  text-xs font-medium px-2 py-0.5 rounded
                  ${event.severity === 'high' ? 'bg-[var(--accent-danger)]/15 text-[var(--accent-danger)]' : ''}
                  ${event.severity === 'medium' ? 'bg-[var(--accent-warning)]/15 text-[var(--accent-warning)]' : ''}
                  ${event.severity === 'low' ? 'bg-[var(--accent-primary)]/15 text-[var(--accent-primary)]' : ''}
                `}>
                  {Math.round(event.geminiConfidence * 100)}%
                </span>
              </div>
            </div>

            {/* Reasoning preview / expanded */}
            {event.geminiExplanation && (
              <div className="mt-3">
                <p className={`text-xs text-[var(--text-secondary)] leading-relaxed ${!isExpanded ? 'line-clamp-2' : ''}`}>
                  {event.geminiExplanation}
                </p>
                
                {event.geminiExplanation.length > 100 && (
                  <button
                    onClick={(e) => toggleExpand(event.id, e)}
                    className="flex items-center gap-1 text-xs text-[var(--accent-primary)] mt-2 hover:underline"
                  >
                    {isExpanded ? (
                      <>
                        <ChevronUp className="w-3 h-3" />
                        Show less
                      </>
                    ) : (
                      <>
                        <ChevronDown className="w-3 h-3" />
                        Show full reasoning
                      </>
                    )}
                  </button>
                )}
              </div>
            )}

            {/* Confidence bars - only when expanded */}
            {isExpanded && (
              <div className="flex items-center gap-4 mt-3 pt-3 border-t border-[var(--border-color)]">
                <div className="flex-1">
                  <div className="flex items-center justify-between text-[10px] mb-1">
                    <span className="text-[var(--text-muted)]">Detection Score</span>
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
                  <div className="flex items-center justify-between text-[10px] mb-1">
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
            )}
          </div>
        );
      })}
    </div>
  );
}

function SeverityIcon({ severity }: { severity: 'high' | 'medium' | 'low' }) {
  const size = "w-3.5 h-3.5";
  switch (severity) {
    case 'high':
      return <XCircle className={`${size} text-[var(--accent-danger)]`} />;
    case 'medium':
      return <AlertTriangle className={`${size} text-[var(--accent-warning)]`} />;
    case 'low':
      return <CheckCircle className={`${size} text-[var(--accent-primary)]`} />;
  }
}

function formatTime(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, '0')}`;
}
