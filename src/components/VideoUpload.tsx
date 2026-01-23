'use client';

import { useCallback, useState } from 'react';
import { Upload, Film, X, FileVideo, ShieldAlert, Link } from 'lucide-react';

// Sample videos for demo
export const SAMPLE_VIDEOS = [
  {
    id: 'robbery1',
    name: 'Robbery #1',
    url: 'https://82qj2yqd8lzq6kc0.public.blob.vercel-storage.com/Robbery069_x264.mp4',
    size: '5.4 MB',
    icon: ShieldAlert,
    color: 'text-red-500',
    bgColor: 'bg-red-500/20',
    suggestedContext: 'Store or bank surveillance camera',
    suggestedTargets: 'Robbery, theft, armed threat, aggressive behavior',
  },
  {
    id: 'robbery2',
    name: 'Robbery #2',
    url: 'https://82qj2yqd8lzq6kc0.public.blob.vercel-storage.com/Robbery004_x264.mp4',
    size: '8.2 MB',
    icon: ShieldAlert,
    color: 'text-red-400',
    bgColor: 'bg-red-500/20',
    suggestedContext: 'Store or bank surveillance camera',
    suggestedTargets: 'Robbery, theft, armed threat, aggressive behavior',
  },
  {
    id: 'fighting1',
    name: 'Fighting #1',
    url: 'https://82qj2yqd8lzq6kc0.public.blob.vercel-storage.com/Fighting046_x264.mp4',
    size: '24 MB',
    icon: ShieldAlert,
    color: 'text-orange-500',
    bgColor: 'bg-orange-500/20',
    suggestedContext: 'Public area or street surveillance camera',
    suggestedTargets: 'Fighting, physical assault, violence, aggressive behavior',
  },
  {
    id: 'fighting2',
    name: 'Fighting #2',
    url: 'https://82qj2yqd8lzq6kc0.public.blob.vercel-storage.com/Fighting020_x264.mp4',
    size: '20 MB',
    icon: ShieldAlert,
    color: 'text-orange-400',
    bgColor: 'bg-orange-500/20',
    suggestedContext: 'Public area or street surveillance camera',
    suggestedTargets: 'Fighting, physical assault, violence, aggressive behavior',
  },
  {
    id: 'vandalism',
    name: 'Vandalism',
    url: 'https://82qj2yqd8lzq6kc0.public.blob.vercel-storage.com/Vandalism010_x264.mp4',
    size: '25 MB',
    icon: ShieldAlert,
    color: 'text-yellow-500',
    bgColor: 'bg-yellow-500/20',
    suggestedContext: 'Property or building exterior surveillance camera',
    suggestedTargets: 'Vandalism, property damage, graffiti, destruction',
  },
  {
    id: 'theft',
    name: 'Theft',
    url: 'https://82qj2yqd8lzq6kc0.public.blob.vercel-storage.com/Stealing006_x264.mp4',
    size: '27 MB',
    icon: ShieldAlert,
    color: 'text-purple-500',
    bgColor: 'bg-purple-500/20',
    suggestedContext: 'Store or parking area surveillance camera',
    suggestedTargets: 'Theft, stealing, shoplifting, unauthorized removal',
  },
  {
    id: 'abuse',
    name: 'Abuse',
    url: 'https://82qj2yqd8lzq6kc0.public.blob.vercel-storage.com/Abuse001_x264.mp4',
    size: '15 MB',
    icon: ShieldAlert,
    color: 'text-pink-500',
    bgColor: 'bg-pink-500/20',
    suggestedContext: 'Indoor or facility surveillance camera',
    suggestedTargets: 'Abuse, physical harm, assault, mistreatment',
  },
  {
    id: 'fire',
    name: 'Fire Incident',
    url: 'https://82qj2yqd8lzq6kc0.public.blob.vercel-storage.com/WhatsApp%20Video%202026-01-21%20at%2018.54.28.mp4',
    size: '6.2 MB',
    icon: ShieldAlert,
    color: 'text-amber-500',
    bgColor: 'bg-amber-500/20',
    suggestedContext: 'Building or facility surveillance camera',
    suggestedTargets: 'Fire, smoke, flames, burning, safety hazard',
  },
];

interface VideoUploadProps {
  onFileSelect: (file: File) => void;
  onSampleSelect?: (sample: typeof SAMPLE_VIDEOS[0]) => void;
  selectedFile: File | null;
  selectedSample?: typeof SAMPLE_VIDEOS[0] | null;
  onClear: () => void;
  disabled?: boolean;
}

export function VideoUpload({ 
  onFileSelect, 
  onSampleSelect, 
  selectedFile, 
  selectedSample, 
  onClear, 
  disabled 
}: VideoUploadProps) {
  const [isDragOver, setIsDragOver] = useState(false);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    
    if (disabled) return;
    
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('video/')) {
      onFileSelect(file);
    }
  }, [onFileSelect, disabled]);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    if (!disabled) setIsDragOver(true);
  }, [disabled]);

  const handleDragLeave = useCallback(() => {
    setIsDragOver(false);
  }, []);

  const handleFileInput = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      onFileSelect(file);
    }
  }, [onFileSelect]);

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024 * 1024) {
      return `${(bytes / 1024).toFixed(1)} KB`;
    }
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  // Show selected file
  if (selectedFile) {
    return (
      <div className="upload-zone has-file">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="w-14 h-14 rounded-xl bg-[var(--accent-primary)]/20 flex items-center justify-center">
              <FileVideo className="w-7 h-7 text-[var(--accent-primary)]" />
            </div>
            <div className="text-left">
              <p className="font-semibold text-[var(--text-primary)] truncate max-w-[300px]">
                {selectedFile.name}
              </p>
              <p className="text-sm text-[var(--text-secondary)]">
                {formatFileSize(selectedFile.size)} • {selectedFile.type.split('/')[1].toUpperCase()}
              </p>
            </div>
          </div>
          {!disabled && (
            <button
              onClick={(e) => {
                e.stopPropagation();
                onClear();
              }}
              className="w-10 h-10 rounded-lg bg-[var(--bg-tertiary)] flex items-center justify-center hover:bg-[var(--accent-danger)]/20 transition-colors group"
            >
              <X className="w-5 h-5 text-[var(--text-secondary)] group-hover:text-[var(--accent-danger)]" />
            </button>
          )}
        </div>
      </div>
    );
  }

  // Show selected sample video
  if (selectedSample) {
    const Icon = selectedSample.icon;
    return (
      <div className="upload-zone has-file">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className={`w-14 h-14 rounded-xl ${selectedSample.bgColor} flex items-center justify-center`}>
              <Icon className={`w-7 h-7 ${selectedSample.color}`} />
            </div>
            <div className="text-left">
              <div className="flex items-center gap-2">
                <p className="font-semibold text-[var(--text-primary)]">
                  {selectedSample.name}
                </p>
                <span className="px-2 py-0.5 text-[10px] font-medium bg-[var(--accent-primary)]/20 text-[var(--accent-primary)] rounded-full">
                  SAMPLE
                </span>
              </div>
              <p className="text-sm text-[var(--text-secondary)]">
                {selectedSample.size} • Pre-loaded demo video
              </p>
            </div>
          </div>
          {!disabled && (
            <button
              onClick={(e) => {
                e.stopPropagation();
                onClear();
              }}
              className="w-10 h-10 rounded-lg bg-[var(--bg-tertiary)] flex items-center justify-center hover:bg-[var(--accent-danger)]/20 transition-colors group"
            >
              <X className="w-5 h-5 text-[var(--text-secondary)] group-hover:text-[var(--accent-danger)]" />
            </button>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Upload zone */}
      <label
        className={`upload-zone block ${isDragOver ? 'dragover' : ''} ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
      >
        <input
          type="file"
          accept="video/*"
          onChange={handleFileInput}
          className="hidden"
          disabled={disabled}
        />
        <div className="flex flex-col items-center gap-4">
          <div className="w-16 h-16 rounded-2xl bg-[var(--accent-primary)]/10 flex items-center justify-center">
            {isDragOver ? (
              <Film className="w-8 h-8 text-[var(--accent-primary)] animate-pulse-glow" />
            ) : (
              <Upload className="w-8 h-8 text-[var(--accent-primary)]" />
            )}
          </div>
          <div>
            <p className="font-semibold text-[var(--text-primary)]">
              {isDragOver ? 'Drop your video here' : 'Upload surveillance footage'}
            </p>
            <p className="text-sm text-[var(--text-secondary)] mt-1">
              Drag & drop or click to browse • MP4, MOV, AVI supported
            </p>
          </div>
        </div>
      </label>

      {/* Sample videos section */}
      {onSampleSelect && (
        <div className="pt-2">
          <div className="flex items-center gap-2 mb-3">
            <Link className="w-4 h-4 text-[var(--text-muted)]" />
            <p className="text-xs font-medium text-[var(--text-muted)] uppercase tracking-wider">
              Or try a sample video
            </p>
          </div>
          <div className="grid grid-cols-2 gap-2">
            {SAMPLE_VIDEOS.map((sample) => {
              const Icon = sample.icon;
              return (
                <button
                  key={sample.id}
                  onClick={() => !disabled && onSampleSelect(sample)}
                  disabled={disabled}
                  className={`
                    flex items-center gap-3 p-3 rounded-xl border border-[var(--border-subtle)]
                    bg-[var(--bg-secondary)] hover:bg-[var(--bg-tertiary)] 
                    hover:border-[var(--accent-primary)]/30
                    transition-all duration-200 text-left group
                    ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}
                  `}
                >
                  <div className={`w-10 h-10 rounded-lg ${sample.bgColor} flex items-center justify-center flex-shrink-0 
                    group-hover:scale-110 transition-transform`}>
                    <Icon className={`w-5 h-5 ${sample.color}`} />
                  </div>
                  <div className="min-w-0">
                    <p className="text-sm font-medium text-[var(--text-primary)] truncate">
                      {sample.name}
                    </p>
                    <p className="text-xs text-[var(--text-muted)]">
                      {sample.size}
                    </p>
                  </div>
                </button>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
