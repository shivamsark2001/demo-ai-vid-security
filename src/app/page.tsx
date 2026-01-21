'use client';

import { useState, useRef, useCallback } from 'react';
import { Scan, Play } from 'lucide-react';
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
    const url = URL.createObjectURL(file);
    setVideoUrl(url);
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
    const maxAttempts = 120;
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
        if (data.status === 'IN_QUEUE') {
          setCurrentStep('Waiting in queue...');
          setProgress(Math.min(10 + attempts, 30));
        } else if (data.status === 'IN_PROGRESS') {
          setStatus('processing');
          setCurrentStep(data.progress?.step || 'Processing...');
          setProgress(Math.min(30 + (attempts * 2), 90));
        }
        
        await new Promise(resolve => setTimeout(resolve, 5000));
        return poll();
      }
    };

    return poll();
  }, []);

  const deleteBlob = async (url: string) => {
    try {
      await fetch('/api/blob-delete', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url }),
      });
    } catch (e) {
      console.warn('Failed to delete blob:', e);
    }
  };

  const handleAnalyze = async () => {
    if (!selectedFile) return;

    let blobUrlToDelete: string | null = null;

    try {
      setStatus('uploading');
      setProgress(0);
      setCurrentStep('Uploading...');

      let blobUrl: string;

      try {
        const blob = await upload(selectedFile.name, selectedFile, {
          access: 'public',
          handleUploadUrl: '/api/upload',
          onUploadProgress: (progressEvent) => {
            const percent = Math.round((progressEvent.loaded / progressEvent.total) * 100);
            setProgress(percent);
            setCurrentStep(`Uploading ${percent}%`);
          },
        });
        blobUrl = blob.url;
        blobUrlToDelete = blob.url;
      } catch (uploadError) {
        console.error('Upload failed:', uploadError);
        setStatus('failed');
        setCurrentStep('Upload failed');
        return;
      }
      
      setProgress(100);
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
        setStatus('processing');
        setCurrentStep('Analyzing...');
        
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
    <main className="min-h-screen p-4 md:p-8">
      <div className="max-w-6xl mx-auto">
        {/* Header - Minimal */}
        <header className="mb-10 animate-slide-up">
          <div className="flex items-center gap-3 mb-2">
            <div className="w-8 h-8 rounded-lg bg-[var(--accent-primary)] flex items-center justify-center">
              <Scan className="w-4 h-4 text-[var(--bg-primary)]" />
            </div>
            <h1 className="text-2xl font-semibold tracking-tight">truvo</h1>
          </div>
          <p className="text-sm text-[var(--text-muted)] ml-11">
            Video intelligence for security
          </p>
        </header>

        {/* Main content */}
        <div className="grid lg:grid-cols-2 gap-6">
          {/* Left column */}
          <div className="space-y-4">
            <div className="glass-card p-5 animate-slide-up stagger-1">
              <VideoUpload
                onFileSelect={handleFileSelect}
                selectedFile={selectedFile}
                onClear={handleClearFile}
                disabled={isAnalyzing}
              />
            </div>

            <div className="glass-card p-5 animate-slide-up stagger-2">
              <ContextInput
                cameraContext={cameraContext}
                detectionTargets={detectionTargets}
                onCameraContextChange={setCameraContext}
                onDetectionTargetsChange={setDetectionTargets}
                disabled={isAnalyzing}
              />
            </div>

            <button
              onClick={handleAnalyze}
              disabled={!selectedFile || isAnalyzing}
              className="btn-primary w-full flex items-center justify-center gap-2 animate-slide-up stagger-3"
            >
              <Play className="w-4 h-4" />
              {isAnalyzing ? 'Analyzing...' : 'Analyze'}
            </button>

            <AnalysisProgress
              status={status}
              progress={progress}
              currentStep={currentStep}
            />
          </div>

          {/* Right column */}
          <div className="space-y-4">
            {videoUrl && (
              <div className="glass-card p-3 animate-slide-up stagger-2">
                <VideoPlayer
                  ref={videoRef}
                  src={videoUrl}
                  onTimeUpdate={setCurrentTime}
                  onDurationChange={setVideoDuration}
                />
              </div>
            )}

            {result && result.events.length > 0 && (
              <div className="glass-card p-5 animate-slide-up">
                <h3 className="text-xs font-medium text-[var(--text-muted)] uppercase tracking-wider mb-3">
                  Timeline
                </h3>
                <Timeline
                  events={result.events}
                  duration={videoDuration}
                  currentTime={currentTime}
                  onEventClick={handleEventClick}
                />
              </div>
            )}

            {result && (
              <div className="glass-card p-5 animate-slide-up">
                <EventList
                  events={result.events}
                  onEventClick={handleEventClick}
                  selectedEventId={selectedEventId}
                />
              </div>
            )}

            {!videoUrl && (
              <div className="glass-card p-10 text-center animate-slide-up stagger-2">
                <div className="w-14 h-14 rounded-xl bg-[var(--bg-tertiary)] flex items-center justify-center mx-auto mb-3">
                  <Scan className="w-6 h-6 text-[var(--text-muted)]" />
                </div>
                <p className="text-sm text-[var(--text-muted)]">
                  Upload video to start
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </main>
  );
}
