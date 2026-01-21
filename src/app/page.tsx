'use client';

import { useState, useRef, useCallback } from 'react';
import { Shield, Zap, Eye } from 'lucide-react';
import { upload } from '@vercel/blob/client';
import { VideoUpload } from '@/components/VideoUpload';
import { ContextInput } from '@/components/ContextInput';
import { AnalysisProgress } from '@/components/AnalysisProgress';
import { VideoPlayer, VideoPlayerRef } from '@/components/VideoPlayer';
import { Timeline } from '@/components/Timeline';
import { EventList } from '@/components/EventList';
import { AnalysisStatus, AnalysisResult, TimelineEvent } from '@/types';

export default function Home() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [cameraContext, setCameraContext] = useState('');
  const [detectionTargets, setDetectionTargets] = useState('');
  const [status, setStatus] = useState<AnalysisStatus>('idle');
  const [progress, setProgress] = useState(0);
  const [currentStep, setCurrentStep] = useState('');
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [videoDuration, setVideoDuration] = useState(0);
  const [selectedEventId, setSelectedEventId] = useState<string | undefined>();
  
  const videoRef = useRef<VideoPlayerRef>(null);

  const handleFileSelect = (file: File) => {
    setSelectedFile(file);
    // Create a local URL for preview
    const url = URL.createObjectURL(file);
    setVideoUrl(url);
    // Reset state
    setResult(null);
    setStatus('idle');
  };

  const handleClearFile = () => {
    if (videoUrl) {
      URL.revokeObjectURL(videoUrl);
    }
    setSelectedFile(null);
    setVideoUrl(null);
    setResult(null);
    setStatus('idle');
  };

  const pollJobStatus = useCallback(async (jobId: string) => {
    const maxAttempts = 120; // 10 minutes max
    let attempts = 0;

    const poll = async (): Promise<AnalysisResult | null> => {
      attempts++;
      if (attempts > maxAttempts) {
        throw new Error('Analysis timed out');
      }

      const response = await fetch(`/api/runpod/status?jobId=${jobId}`);
      const data = await response.json();

      if (data.status === 'COMPLETED') {
        return data.output;
      } else if (data.status === 'FAILED') {
        throw new Error(data.error || 'Analysis failed');
      } else {
        // Update progress based on status
        if (data.status === 'IN_QUEUE') {
          setCurrentStep('Waiting in queue...');
          setProgress(Math.min(10 + attempts, 30));
        } else if (data.status === 'IN_PROGRESS') {
          setStatus('processing');
          setCurrentStep(data.progress?.step || 'Processing video...');
          setProgress(Math.min(30 + (attempts * 2), 90));
        }
        
        await new Promise(resolve => setTimeout(resolve, 5000));
        return poll();
      }
    };

    return poll();
  }, []);

  // Delete blob after analysis to save storage costs
  const deleteBlob = async (url: string) => {
    try {
      await fetch('/api/blob-delete', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url }),
      });
      console.log('Blob deleted:', url);
    } catch (e) {
      console.warn('Failed to delete blob:', e);
    }
  };

  const handleAnalyze = async () => {
    if (!selectedFile) return;

    let blobUrlToDelete: string | null = null;

    try {
      // Step 1: Upload video using client-side upload
      setStatus('uploading');
      setProgress(0);
      setCurrentStep('Uploading video...');

      let blobUrl: string;

      try {
        // Client-side upload directly to Vercel Blob (bypasses 4.5MB limit)
        const blob = await upload(selectedFile.name, selectedFile, {
          access: 'public',
          handleUploadUrl: '/api/upload',
          addRandomSuffix: true, // Prevent "blob already exists" errors
          onUploadProgress: (progressEvent) => {
            const percent = Math.round((progressEvent.loaded / progressEvent.total) * 100);
            setProgress(percent);
            setCurrentStep(`Uploading video... ${percent}%`);
          },
        });
        blobUrl = blob.url;
        blobUrlToDelete = blob.url; // Save for cleanup later
        console.log('Video uploaded to:', blobUrl);
      } catch (uploadError) {
        console.error('Upload failed:', uploadError);
        setStatus('failed');
        setCurrentStep('Upload failed. Please try again.');
        return;
      }
      
      setProgress(100);

      // Step 2: Try RunPod first, fallback to mock
      setStatus('queued');
      setProgress(0);
      setCurrentStep('Starting analysis...');

      let useMock = false;
      let jobId = '';

      try {
        const runpodResponse = await fetch('/api/runpod/run', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            video_url: blobUrl,
            camera_context: cameraContext || 'Security surveillance camera',
            detection_targets: detectionTargets || 'Suspicious or threatening behavior, unauthorized access, safety hazards',
          }),
        });

        if (runpodResponse.ok) {
          const runpodData = await runpodResponse.json();
          jobId = runpodData.id;
        } else {
          useMock = true;
        }
      } catch {
        useMock = true;
      }

      if (useMock) {
        // Use mock endpoint for demo
        setStatus('processing');
        setCurrentStep('Running AI analysis (demo mode)...');
        
        // Simulate progress
        for (let i = 0; i <= 100; i += 10) {
          setProgress(i);
          await new Promise(resolve => setTimeout(resolve, 200));
        }

        const mockResponse = await fetch('/api/mock-analyze', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            video_url: blobUrl,
            camera_context: cameraContext,
            detection_targets: detectionTargets,
          }),
        });

        const mockResult = await mockResponse.json();
        setResult(mockResult);
        setStatus('completed');
        setProgress(100);
        return;
      }

      setProgress(10);

      // Step 3: Poll for results
      const analysisResult = await pollJobStatus(jobId);
      
      if (analysisResult) {
        setResult(analysisResult);
        setStatus('completed');
        setProgress(100);
      }
    } catch (error) {
      console.error('Analysis error:', error);
      setStatus('failed');
    } finally {
      // Clean up blob after analysis (success or failure) - saves storage costs
      if (blobUrlToDelete) {
        deleteBlob(blobUrlToDelete);
      }
    }
  };

  const handleEventClick = (event: TimelineEvent) => {
    setSelectedEventId(event.id);
    videoRef.current?.seekTo(event.t0);
    videoRef.current?.play();
  };

  const isAnalyzing = status !== 'idle' && status !== 'completed' && status !== 'failed';

  return (
    <main className="min-h-screen p-6 md:p-10">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <header className="text-center mb-12 animate-slide-up">
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-[var(--accent-primary)]/10 border border-[var(--accent-primary)]/30 mb-6">
            <Shield className="w-4 h-4 text-[var(--accent-primary)]" />
            <span className="text-sm text-[var(--accent-primary)] font-medium">AI-Powered Security</span>
          </div>
          <h1 className="text-4xl md:text-5xl font-bold mb-4 bg-gradient-to-r from-[var(--text-primary)] via-[var(--accent-primary)] to-[var(--accent-secondary)] bg-clip-text text-transparent">
            Video Security Analyzer
          </h1>
          <p className="text-[var(--text-secondary)] max-w-2xl mx-auto">
            Upload surveillance footage for intelligent threat detection. 
            Get timestamped alerts with confidence scores.
          </p>
        </header>

        {/* Main content */}
        <div className="grid lg:grid-cols-2 gap-8">
          {/* Left column - Upload & Controls */}
          <div className="space-y-6">
            {/* Upload section */}
            <div className="glass-card p-6 animate-slide-up stagger-1">
              <div className="flex items-center gap-2 mb-4">
                <Eye className="w-5 h-5 text-[var(--accent-primary)]" />
                <h2 className="font-semibold text-lg">Upload Footage</h2>
              </div>
              <VideoUpload
                onFileSelect={handleFileSelect}
                selectedFile={selectedFile}
                onClear={handleClearFile}
                disabled={isAnalyzing}
              />
            </div>

            {/* Context input */}
            <div className="glass-card p-6 animate-slide-up stagger-2">
              <ContextInput
                cameraContext={cameraContext}
                detectionTargets={detectionTargets}
                onCameraContextChange={setCameraContext}
                onDetectionTargetsChange={setDetectionTargets}
                disabled={isAnalyzing}
              />
            </div>

            {/* Analyze button */}
            <button
              onClick={handleAnalyze}
              disabled={!selectedFile || isAnalyzing}
              className="btn-primary w-full flex items-center justify-center gap-2 text-lg animate-slide-up stagger-3"
            >
              <Zap className="w-5 h-5" />
              {isAnalyzing ? 'Analyzing...' : 'Analyze Video'}
            </button>

            {/* Progress */}
            <AnalysisProgress
              status={status}
              progress={progress}
              currentStep={currentStep}
            />
          </div>

          {/* Right column - Video & Results */}
          <div className="space-y-6">
            {/* Video player */}
            {videoUrl && (
              <div className="glass-card p-4 animate-slide-up stagger-2">
                <VideoPlayer
                  ref={videoRef}
                  src={videoUrl}
                  onTimeUpdate={setCurrentTime}
                  onDurationChange={setVideoDuration}
                />
              </div>
            )}

            {/* Timeline */}
            {result && result.events.length > 0 && (
              <div className="glass-card p-6 animate-slide-up">
                <h3 className="font-semibold mb-4 flex items-center gap-2">
                  <div className="w-2 h-2 rounded-full bg-[var(--accent-primary)] animate-pulse-glow" />
                  Event Timeline
                </h3>
                <Timeline
                  events={result.events}
                  duration={videoDuration}
                  currentTime={currentTime}
                  onEventClick={handleEventClick}
                />
              </div>
            )}

            {/* Event list */}
            {result && (
              <div className="glass-card p-6 animate-slide-up">
                <EventList
                  events={result.events}
                  onEventClick={handleEventClick}
                  selectedEventId={selectedEventId}
                />
              </div>
            )}

            {/* Empty state */}
            {!videoUrl && (
              <div className="glass-card p-12 text-center animate-slide-up stagger-2">
                <div className="w-20 h-20 rounded-2xl bg-[var(--bg-tertiary)] flex items-center justify-center mx-auto mb-4">
                  <Eye className="w-10 h-10 text-[var(--text-muted)]" />
                </div>
                <h3 className="font-semibold text-[var(--text-secondary)] mb-2">
                  No Video Selected
                </h3>
                <p className="text-sm text-[var(--text-muted)]">
                  Upload a video to begin security analysis
                </p>
              </div>
            )}
          </div>
        </div>

      </div>
    </main>
  );
}
