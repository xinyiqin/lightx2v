import React, { useState, useRef, useEffect } from 'react';
import { X, Play, Pause } from 'lucide-react';
import { Language } from '../../i18n/useTranslation';

interface AudioEditorModalProps {
  nodeId: string;
  audioData: string;
  onClose: () => void;
  onSave: (trimmedAudio: string) => void;
  lang: Language;
}

// Helper functions for audio processing
function audioBufferToWav(buffer: AudioBuffer): ArrayBuffer {
  const length = buffer.length;
  const numberOfChannels = buffer.numberOfChannels;
  const sampleRate = buffer.sampleRate;
  const arrayBuffer = new ArrayBuffer(44 + length * numberOfChannels * 2);
  const view = new DataView(arrayBuffer);

  const writeString = (offset: number, string: string) => {
    for (let i = 0; i < string.length; i++) {
      view.setUint8(offset + i, string.charCodeAt(i));
    }
  };

  writeString(0, 'RIFF');
  view.setUint32(4, 36 + length * numberOfChannels * 2, true);
  writeString(8, 'WAVE');
  writeString(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, numberOfChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * numberOfChannels * 2, true);
  view.setUint16(32, numberOfChannels * 2, true);
  view.setUint16(34, 16, true);
  writeString(36, 'data');
  view.setUint32(40, length * numberOfChannels * 2, true);

  let offset = 44;
  for (let i = 0; i < length; i++) {
    for (let channel = 0; channel < numberOfChannels; channel++) {
      const sample = Math.max(-1, Math.min(1, buffer.getChannelData(channel)[i]));
      view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
      offset += 2;
    }
  }

  return arrayBuffer;
}

function arrayBufferToBase64(buffer: ArrayBuffer): string {
  const bytes = new Uint8Array(buffer);
  let binary = '';
  for (let i = 0; i < bytes.byteLength; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary);
}

export const AudioEditorModal: React.FC<AudioEditorModalProps> = ({
  nodeId,
  audioData,
  onClose,
  onSave,
  lang
}) => {
  const [audioBuffer, setAudioBuffer] = useState<AudioBuffer | null>(null);
  const [startTime, setStartTime] = useState(0);
  const [endTime, setEndTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [waveformData, setWaveformData] = useState<number[]>([]);
  const [draggingStart, setDraggingStart] = useState(false);
  const [draggingEnd, setDraggingEnd] = useState(false);
  const [trimmedAudioUrl, setTrimmedAudioUrl] = useState<string | null>(null);
  const [trimmedDuration, setTrimmedDuration] = useState(0);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const trimmedAudioRef = useRef<HTMLAudioElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const playbackBarRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    // Load audio and create buffer
    const audio = new Audio(audioData);
    audioRef.current = audio;

    const handleLoadedMetadata = () => {
      const dur = audio.duration;
      setDuration(dur);
      setEndTime(dur);
    };

    const handleTimeUpdate = () => {
      setCurrentTime(audio.currentTime);
      if (audio.currentTime >= endTime) {
        audio.pause();
        setIsPlaying(false);
      }
    };

    const handleEnded = () => {
      setIsPlaying(false);
    };

    audio.addEventListener('loadedmetadata', handleLoadedMetadata);
    audio.addEventListener('timeupdate', handleTimeUpdate);
    audio.addEventListener('ended', handleEnded);

    // Load audio buffer for waveform
    fetch(audioData)
      .then(res => res.arrayBuffer())
      .then(arrayBuffer => {
        const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
        return audioContext.decodeAudioData(arrayBuffer);
      })
      .then(buffer => {
        setAudioBuffer(buffer);
        // Generate waveform data
        const samples = 200;
        const sampleRate = buffer.sampleRate;
        const samplesPerBar = Math.floor(buffer.length / samples);
        const waveform: number[] = [];

        for (let i = 0; i < samples; i++) {
          let sum = 0;
          for (let j = 0; j < samplesPerBar; j++) {
            const index = i * samplesPerBar + j;
            if (index < buffer.length) {
              const channelData = buffer.getChannelData(0);
              sum += Math.abs(channelData[index]);
            }
          }
          waveform.push(sum / samplesPerBar);
        }
        setWaveformData(waveform);
      })
      .catch(err => console.error('Failed to load audio buffer:', err));

    return () => {
      audio.removeEventListener('loadedmetadata', handleLoadedMetadata);
      audio.removeEventListener('timeupdate', handleTimeUpdate);
      audio.removeEventListener('ended', handleEnded);
      if (audioRef.current) {
        audioRef.current.pause();
        audioRef.current = null;
      }
    };
  }, [audioData]);

  useEffect(() => {
    // Draw waveform
    if (!canvasRef.current || waveformData.length === 0 || duration === 0) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;
    const barWidth = width / waveformData.length;
    const maxAmplitude = Math.max(...waveformData, 0.001);

    ctx.clearRect(0, 0, width, height);

    waveformData.forEach((amplitude, index) => {
      const barHeight = (amplitude / maxAmplitude) * height * 0.7;
      const x = index * barWidth;
      const y = (height - barHeight) / 2;

      const timePerBar = duration / waveformData.length;
      const barStartTime = index * timePerBar;
      const barEndTime = (index + 1) * timePerBar;

      if (barStartTime >= startTime && barEndTime <= endTime) {
        ctx.fillStyle = '#90dce1';
      } else {
        ctx.fillStyle = '#d1d5db';
      }

      ctx.fillRect(x, y, Math.max(1, barWidth - 1), barHeight);
    });

    if (isPlaying && duration > 0) {
      const timeRatio = trimmedAudioUrl ? currentTime / trimmedDuration : currentTime / duration;
      const indicatorX = timeRatio * width;
      ctx.strokeStyle = '#90dce1';
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(indicatorX, 0);
      ctx.lineTo(indicatorX, height);
      ctx.stroke();
    }

    if (duration > 0 && !trimmedAudioUrl) {
      const startX = (startTime / duration) * width;
      const endX = (endTime / duration) * width;

      ctx.fillStyle = 'rgba(144, 220, 225, 0.15)';
      ctx.fillRect(startX, 0, endX - startX, height);

      ctx.fillStyle = '#90dce1';
      ctx.fillRect(startX - 2, 0, 4, height);
      ctx.beginPath();
      ctx.arc(startX, height / 2, 10, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 3;
      ctx.stroke();

      ctx.fillStyle = '#90dce1';
      ctx.fillRect(endX - 2, 0, 4, height);
      ctx.beginPath();
      ctx.arc(endX, height / 2, 10, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 3;
      ctx.stroke();
    }
  }, [waveformData, startTime, endTime, duration, isPlaying, currentTime, trimmedAudioUrl, trimmedDuration]);

  const handlePlay = () => {
    const audio = trimmedAudioUrl ? trimmedAudioRef.current : audioRef.current;
    if (!audio) return;

    if (isPlaying) {
      audio.pause();
      setIsPlaying(false);
    } else {
      if (trimmedAudioUrl) {
        audio.currentTime = 0;
      } else {
        audio.currentTime = startTime;
      }
      audio.play();
      setIsPlaying(true);
    }
  };

  useEffect(() => {
    if (!trimmedAudioRef.current) return;
    const audio = trimmedAudioRef.current;

    const handleTimeUpdate = () => {
      setCurrentTime(audio.currentTime);
    };

    const handleEnded = () => {
      setIsPlaying(false);
      setCurrentTime(0);
    };

    audio.addEventListener('timeupdate', handleTimeUpdate);
    audio.addEventListener('ended', handleEnded);

    return () => {
      audio.removeEventListener('timeupdate', handleTimeUpdate);
      audio.removeEventListener('ended', handleEnded);
    };
  }, [trimmedAudioUrl]);

  const handleTrim = async () => {
    if (!audioBuffer || startTime >= endTime) return;

    try {
      const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
      const sampleRate = audioBuffer.sampleRate;
      const startSample = Math.floor(startTime * sampleRate);
      const endSample = Math.floor(endTime * sampleRate);
      const length = endSample - startSample;

      if (length <= 0) {
        alert(lang === 'zh' ? '请选择有效的剪辑范围' : 'Please select a valid trim range');
        return;
      }

      const newBuffer = audioContext.createBuffer(audioBuffer.numberOfChannels, length, sampleRate);

      for (let channel = 0; channel < audioBuffer.numberOfChannels; channel++) {
        const oldData = audioBuffer.getChannelData(channel);
        const newData = newBuffer.getChannelData(channel);
        for (let i = 0; i < length; i++) {
          newData[i] = oldData[startSample + i];
        }
      }

      const wav = audioBufferToWav(newBuffer);
      const base64 = arrayBufferToBase64(wav);
      const dataUrl = `data:audio/wav;base64,${base64}`;

      setTrimmedAudioUrl(dataUrl);
      setTrimmedDuration(endTime - startTime);
      setCurrentTime(0);
    } catch (error: any) {
      console.error('Failed to trim audio:', error);
      alert(lang === 'zh' ? '剪辑失败: ' + error.message : 'Trim failed: ' + error.message);
    }
  };

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!canvasRef.current || duration === 0) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const width = rect.width;
    const startX = (startTime / duration) * width;
    const endX = (endTime / duration) * width;

    if (Math.abs(x - startX) < 20) {
      setDraggingStart(true);
      e.preventDefault();
      return;
    }

    if (Math.abs(x - endX) < 20) {
      setDraggingEnd(true);
      e.preventDefault();
      return;
    }
  };

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!canvasRef.current || duration === 0) return;
      const rect = canvasRef.current.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const ratio = Math.max(0, Math.min(1, x / rect.width));
      const time = ratio * duration;

      if (draggingStart) {
        setStartTime(Math.min(Math.max(0, time), endTime));
        if (audioRef.current) {
          audioRef.current.currentTime = time;
        }
      } else if (draggingEnd) {
        setEndTime(Math.min(Math.max(time, startTime), duration));
      }
    };

    const handleMouseUp = () => {
      setDraggingStart(false);
      setDraggingEnd(false);
    };

    if (draggingStart || draggingEnd) {
      window.addEventListener('mousemove', handleMouseMove);
      window.addEventListener('mouseup', handleMouseUp);
      return () => {
        window.removeEventListener('mousemove', handleMouseMove);
        window.removeEventListener('mouseup', handleMouseUp);
      };
    }
  }, [draggingStart, draggingEnd, duration, endTime, startTime]);

  const handlePlaybackBarClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!playbackBarRef.current) return;
    const rect = playbackBarRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const ratio = Math.max(0, Math.min(1, x / rect.width));
    const audio = trimmedAudioUrl ? trimmedAudioRef.current : audioRef.current;
    if (audio) {
      if (trimmedAudioUrl) {
        audio.currentTime = ratio * trimmedDuration;
      } else {
        audio.currentTime = startTime + ratio * (endTime - startTime);
      }
      setCurrentTime(audio.currentTime);
    }
  };

  const displayDuration = trimmedAudioUrl ? trimmedDuration : duration;
  const displayCurrentTime = trimmedAudioUrl ? currentTime : currentTime - startTime;
  const displayStartTime = trimmedAudioUrl ? 0 : startTime;
  const displayEndTime = trimmedAudioUrl ? trimmedDuration : endTime;

  return (
    <div className="fixed inset-0 bg-black/60 backdrop-blur-md z-50 flex items-center justify-center p-4" onClick={onClose}>
      <div className="bg-white rounded-3xl shadow-2xl w-full max-w-3xl overflow-hidden" onClick={e => e.stopPropagation()}>
        <div className="px-8 py-6 border-b border-gray-200">
          <div className="flex items-center justify-between">
            <h2 className="text-2xl font-semibold text-gray-900">{lang === 'zh' ? '音频剪辑' : 'Audio Editor'}</h2>
            <button onClick={onClose} className="p-2 text-gray-400 hover:text-gray-600 transition-colors rounded-full hover:bg-gray-100">
              <X size={20} />
            </button>
          </div>
        </div>

        <div className="px-8 py-6 space-y-8">
          <div className="space-y-3">
            <div ref={containerRef} className="relative">
              <canvas
                ref={canvasRef}
                width={800}
                height={120}
                onMouseDown={handleMouseDown}
                className="w-full h-24 bg-gray-50 rounded-2xl border border-gray-200"
                style={{ cursor: draggingStart || draggingEnd ? 'grabbing' : 'grab' }}
              />
            </div>

            {!trimmedAudioUrl && (
              <div className="flex items-center justify-between text-xs text-gray-500 px-1">
                <span>{formatTime(startTime)}</span>
                <span className="text-gray-400">{formatTime(endTime - startTime)}</span>
                <span>{formatTime(endTime)}</span>
              </div>
            )}
          </div>

          <div className="space-y-2">
            <div
              ref={playbackBarRef}
              onClick={handlePlaybackBarClick}
              className="relative h-2 bg-gray-200 rounded-full cursor-pointer overflow-hidden"
            >
              <div
                className="absolute h-full bg-blue-500 rounded-full transition-all"
                style={{ width: `${displayDuration > 0 ? (displayCurrentTime / displayDuration) * 100 : 0}%` }}
              />
            </div>
            <div className="flex items-center justify-between text-xs text-gray-500">
              <span>{formatTime(displayCurrentTime)}</span>
              <span>{formatTime(displayDuration)}</span>
            </div>
          </div>

          {!trimmedAudioUrl && <audio ref={audioRef} src={audioData} />}
          {trimmedAudioUrl && <audio ref={trimmedAudioRef} src={trimmedAudioUrl} />}

          <div className="flex items-center justify-end gap-3 pt-4 border-t border-gray-200">
            <button
              onClick={onClose}
              className="px-6 py-2.5 text-gray-700 hover:bg-gray-100 rounded-xl font-medium transition-colors"
            >
              {lang === 'zh' ? '取消' : 'Cancel'}
            </button>

            {trimmedAudioUrl ? (
              <>
                <button
                  onClick={() => {
                    setTrimmedAudioUrl(null);
                    setTrimmedDuration(0);
                    setCurrentTime(0);
                    setIsPlaying(false);
                  }}
                  className="px-6 py-2.5 text-gray-700 hover:bg-gray-100 rounded-xl font-medium transition-colors"
                >
                  {lang === 'zh' ? '重新剪辑' : 'Re-edit'}
                </button>
                <button
                  onClick={() => {
                    if (trimmedAudioUrl) {
                      onSave(trimmedAudioUrl);
                      onClose();
                    }
                  }}
                  className="px-6 py-2.5 bg-blue-500 hover:bg-blue-600 text-white rounded-xl font-medium transition-colors shadow-sm"
                >
                  {lang === 'zh' ? '保存' : 'Save'}
                </button>
              </>
            ) : (
              <button
                onClick={handleTrim}
                disabled={startTime >= endTime || Math.abs(endTime - startTime) < 0.1}
                className="px-6 py-2.5 bg-blue-500 hover:bg-blue-600 disabled:bg-gray-300 disabled:cursor-not-allowed text-white rounded-xl font-medium transition-colors shadow-sm"
              >
                {lang === 'zh' ? '应用剪辑' : 'Apply Trim'}
              </button>
            )}

            <button
              onClick={handlePlay}
              className="px-6 py-2.5 bg-gray-900 hover:bg-gray-800 text-white rounded-xl font-medium transition-colors flex items-center gap-2"
            >
              {isPlaying ? <Pause size={16} /> : <Play size={16} />}
              {isPlaying ? (lang === 'zh' ? '暂停' : 'Pause') : (lang === 'zh' ? '播放' : 'Play')}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};


