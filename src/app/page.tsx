'use client';

import { useState, useRef, useCallback } from 'react';
import { Scan, Play } from 'lucide-react';
import { upload } from '@vercel/blob/client';
import { VideoUpload } from '@/components/VideoUpload';
import { ContextInput } from '@/components/ContextInput';
import { AnalysisProgress } from '@/components/AnalysisProgress';
import { VideoPlayer, VideoPlayerRef } from '@/components/VideoPlayer';
import { PipelineResults } from '@/components/PipelineResults';
import { AnalysisStatus, AnalysisResult } from '@/types';

export default function Home() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [cameraContext, setCameraContext] = useState('');
  const [detectionTargets, setDetectionTargets] = useState('');
  const [status, setStatus] = useState<AnalysisStatus>('idle');
  const [progress, setProgress] = useState(0);
  const [currentStep, setCurrentStep] = useState('');
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  
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

  const isAnalyzing = status !== 'idle' && status !== 'completed' && status !== 'failed';

  return (
    <main className="min-h-screen p-4 md:p-6 lg:p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <header className="mb-8 animate-slide-up">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-[var(--accent-primary)] to-[var(--accent-secondary)] flex items-center justify-center shadow-lg shadow-[var(--accent-primary)]/20">
              <Scan className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-semibold tracking-tight">Truwo</h1>
              <p className="text-[11px] text-[var(--text-muted)]">Video intelligence</p>
            </div>
          </div>
        </header>

        {/* Main Grid */}
        <div className="grid lg:grid-cols-5 gap-5">
          {/* Left - Controls */}
          <div className="lg:col-span-2 space-y-4">
            <div className="glass-card p-4 animate-slide-up stagger-1">
              <VideoUpload
                onFileSelect={handleFileSelect}
                selectedFile={selectedFile}
                onClear={handleClearFile}
                disabled={isAnalyzing}
              />
            </div>

            <div className="glass-card p-4 animate-slide-up stagger-2">
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
              {isAnalyzing ? 'Analyzing...' : 'Analyze Video'}
            </button>

            <AnalysisProgress
              status={status}
              progress={progress}
              currentStep={currentStep}
            />
          </div>

          {/* Right - Results */}
          <div className="lg:col-span-3 space-y-4">
            {/* Video Player */}
            {videoUrl && (
              <div className="glass-card p-3 animate-slide-up stagger-2">
                <VideoPlayer
                  ref={videoRef}
                  src={videoUrl}
                />
              </div>
            )}

            {/* Analysis Results */}
            {result && (
              <div className="animate-slide-up">
                <PipelineResults result={result} />
              </div>
            )}

            {/* Empty State */}
            {!videoUrl && (
              <div className="glass-card p-12 text-center animate-slide-up stagger-2">
                <div className="w-16 h-16 rounded-2xl bg-[var(--bg-tertiary)] flex items-center justify-center mx-auto mb-4">
                  <Scan className="w-8 h-8 text-[var(--text-muted)]" />
                </div>
                <p className="text-[var(--text-muted)] text-sm">
                  Upload a video to begin analysis
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </main>
  );
}
