'use client';

import { useCallback, useState } from 'react';
import { Upload, Film, X, FileVideo } from 'lucide-react';

interface VideoUploadProps {
  onFileSelect: (file: File) => void;
  selectedFile: File | null;
  onClear: () => void;
  disabled?: boolean;
}

export function VideoUpload({ onFileSelect, selectedFile, onClear, disabled }: VideoUploadProps) {
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

  return (
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
  );
}
