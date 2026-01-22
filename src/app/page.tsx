'use client';

import { useState, useRef, useCallback } from 'react';
import { Scan, Play } from 'lucide-react';
import { upload } from '@vercel/blob/client';
import { VideoUpload, SAMPLE_VIDEOS } from '@/components/VideoUpload';
import { ContextInput } from '@/components/ContextInput';
import { AnalysisProgress } from '@/components/AnalysisProgress';
import { VideoPlayer, VideoPlayerRef } from '@/components/VideoPlayer';
import { PipelineResults } from '@/components/PipelineResults';
import { AnalysisStatus, AnalysisResult } from '@/types';

type SampleVideo = typeof SAMPLE_VIDEOS[0];

export default function Home() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [selectedSample, setSelectedSample] = useState<SampleVideo | null>(null);
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
    setSelectedSample(null);
    const url = URL.createObjectURL(file);
    setVideoUrl(url);
    setResult(null);
    setStatus('idle');
  };

  const handleSampleSelect = (sample: SampleVideo) => {
    setSelectedSample(sample);
    setSelectedFile(null);
    setVideoUrl(sample.url);
    // Pre-fill context with suggested values
    setCameraContext(sample.suggestedContext);
    setDetectionTargets(sample.suggestedTargets);
    setResult(null);
    setStatus('idle');
  };

  const handleClearFile = () => {
    if (videoUrl && selectedFile) {
      URL.revokeObjectURL(videoUrl);
    }
    setSelectedFile(null);
    setSelectedSample(null);
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
    if (!selectedFile && !selectedSample) return;

    let blobUrlToDelete: string | null = null;

    try {
      let blobUrl: string;

      // For sample videos, skip upload - they're already hosted
      if (selectedSample) {
        blobUrl = selectedSample.url;
        setStatus('queued');
        setProgress(0);
        setCurrentStep('Starting analysis...');
      } else if (selectedFile) {
        // Regular file upload flow
        setStatus('uploading');
        setProgress(0);
        setCurrentStep('Uploading to cloud...');

        try {
          console.log('Starting upload for:', selectedFile.name, selectedFile.size);
          const blob = await upload(selectedFile.name, selectedFile, {
            access: 'public',
            handleUploadUrl: '/api/upload',
            onUploadProgress: (progressEvent) => {
              const percent = Math.round((progressEvent.loaded / progressEvent.total) * 100);
              setProgress(percent);
              setCurrentStep(`Uploading ${percent}%`);
            },
          });
          
          console.log('Upload response:', blob);
          
          // Validate upload response
          if (!blob || !blob.url) {
            console.error('Upload returned invalid response:', blob);
            setStatus('failed');
            setCurrentStep('Upload failed - no URL returned');
            return;
          }
          
          // Double-check it's not a blob: URL
          if (blob.url.startsWith('blob:')) {
            console.error('Upload returned browser blob URL instead of hosted URL:', blob.url);
            setStatus('failed');
            setCurrentStep('Upload failed - invalid URL');
            return;
          }
          
          blobUrl = blob.url;
          blobUrlToDelete = blob.url;
          console.log('Upload successful, URL:', blobUrl);
        } catch (uploadError) {
          console.error('Upload failed:', uploadError);
          setStatus('failed');
          setCurrentStep('Upload failed - ' + (uploadError instanceof Error ? uploadError.message : 'unknown error'));
          return;
        }
        
        setProgress(100);
        setStatus('queued');
        setProgress(0);
        setCurrentStep('Starting analysis...');
      } else {
        return;
      }

      let useMock = false;
      let jobId = '';

      // FINAL SAFEGUARD: Never send browser blob: URLs to backend
      if (!blobUrl || blobUrl.startsWith('blob:')) {
        console.error('CRITICAL: Invalid URL before sending to backend:', blobUrl);
        setStatus('failed');
        setCurrentStep('Invalid video URL - please try again');
        return;
      }

      console.log('✅ Sending to RunPod:', blobUrl.substring(0, 100));

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
  const hasResults = result !== null;
  const hasVideo = selectedFile !== null || selectedSample !== null;

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
        <div className="grid lg:grid-cols-2 gap-6">
          {/* LEFT - All Inputs */}
          <div className="space-y-4">
            {/* Video Upload */}
            <div className="glass-card p-4 animate-slide-up stagger-1">
              <VideoUpload
                onFileSelect={handleFileSelect}
                onSampleSelect={handleSampleSelect}
                selectedFile={selectedFile}
                selectedSample={selectedSample}
                onClear={handleClearFile}
                disabled={isAnalyzing}
              />
            </div>

            {/* Video Preview */}
            {videoUrl && (
              <div className="glass-card p-3 animate-slide-up">
                <VideoPlayer
                  ref={videoRef}
                  src={videoUrl}
                />
              </div>
            )}

            {/* Context Input */}
            <div className="glass-card p-4 animate-slide-up stagger-2">
              <ContextInput
                cameraContext={cameraContext}
                detectionTargets={detectionTargets}
                onCameraContextChange={setCameraContext}
                onDetectionTargetsChange={setDetectionTargets}
                disabled={isAnalyzing}
              />
            </div>

            {/* Analyze Button */}
            <button
              onClick={handleAnalyze}
              disabled={!hasVideo || isAnalyzing}
              className="btn-primary w-full flex items-center justify-center gap-2 animate-slide-up stagger-3"
            >
              <Play className="w-4 h-4" />
              {isAnalyzing ? 'Analyzing...' : 'Analyze Video'}
            </button>

            {/* Progress */}
            <AnalysisProgress
              status={status}
              progress={progress}
              currentStep={currentStep}
            />
          </div>

          {/* RIGHT - All Outputs */}
          <div className="space-y-4">
            {/* Results */}
            {hasResults ? (
              <div className="animate-slide-up">
                <PipelineResults result={result} />
              </div>
            ) : (
              <div className="glass-card p-12 text-center animate-slide-up stagger-2 h-full flex flex-col items-center justify-center min-h-[400px]">
                <div className="w-16 h-16 rounded-2xl bg-[var(--bg-tertiary)] flex items-center justify-center mx-auto mb-4">
                  <Scan className="w-8 h-8 text-[var(--text-muted)]" />
                </div>
                <p className="text-[var(--text-muted)] text-sm">
                  Analysis results will appear here
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </main>
  );
}
