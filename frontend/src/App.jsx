import { useState, useEffect, useRef } from 'react';
import './App.css';

function App() {
  // Recording State
  const [isRecording, setIsRecording] = useState(false);
  const [recordingStatus, setRecordingStatus] = useState("Ready to record");
  const [audioUrl, setAudioUrl] = useState(null);
  const [audioBlob, setAudioBlob] = useState(null);
  const [voiceName, setVoiceName] = useState('');
  const [isSaving, setIsSaving] = useState(false);
  
  // Precompute State
  const [voices, setVoices] = useState([]);
  const [selectedVoice, setSelectedVoice] = useState('');
  const [isPrecomputing, setIsPrecomputing] = useState(false);
  const [precomputeStatus, setPrecomputeStatus] = useState({ type: '', message: '' });

  // Refs
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const canvasRef = useRef(null);
  const animationRef = useRef(null);
  const audioContextRef = useRef(null);
  const analyserRef = useRef(null);
  const dataArrayRef = useRef(null);
  const streamRef = useRef(null);

  // Initialize Canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (canvas) {
      const resizeCanvas = () => {
        canvas.width = canvas.parentElement.clientWidth;
        canvas.height = canvas.parentElement.clientHeight;
        drawFlatLine();
      };
      
      window.addEventListener('resize', resizeCanvas);
      resizeCanvas();
      
      return () => window.removeEventListener('resize', resizeCanvas);
    }
  }, []);

  // Fetch voices on mount
  useEffect(() => {
    fetchVoices();
  }, []);

  const drawFlatLine = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = 'rgba(0, 0, 0, 0.2)';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.beginPath();
    ctx.moveTo(0, canvas.height / 2);
    ctx.lineTo(canvas.width, canvas.height / 2);
    ctx.strokeStyle = 'rgba(59, 130, 246, 0.5)';
    ctx.lineWidth = 2;
    ctx.stroke();
  };

  const drawVisualizer = () => {
    const canvas = canvasRef.current;
    const analyser = analyserRef.current;
    const dataArray = dataArrayRef.current;
    if (!canvas || !analyser) return;

    animationRef.current = requestAnimationFrame(drawVisualizer);
    analyser.getByteTimeDomainData(dataArray);

    const ctx = canvas.getContext('2d');
    ctx.fillStyle = 'rgba(0, 0, 0, 0.2)';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    ctx.lineWidth = 2;
    ctx.strokeStyle = '#3b82f6';
    ctx.beginPath();

    const sliceWidth = canvas.width * 1.0 / analyser.frequencyBinCount;
    let x = 0;

    for (let i = 0; i < analyser.frequencyBinCount; i++) {
      const v = dataArray[i] / 128.0;
      const y = v * canvas.height / 2;

      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }

      x += sliceWidth;
    }

    ctx.lineTo(canvas.width, canvas.height / 2);
    ctx.stroke();
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      const audioContext = new (window.AudioContext || window.webkitAudioContext)();
      audioContextRef.current = audioContext;
      
      const source = audioContext.createMediaStreamSource(stream);
      const analyser = audioContext.createAnalyser();
      analyser.fftSize = 2048;
      source.connect(analyser);
      analyserRef.current = analyser;
      dataArrayRef.current = new Uint8Array(analyser.frequencyBinCount);

      drawVisualizer();

      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = () => {
        const blob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
        setAudioBlob(blob);
        setAudioUrl(URL.createObjectURL(blob));

        stream.getTracks().forEach(track => track.stop());
        if (audioContextRef.current) {
          audioContextRef.current.close();
        }
        if (animationRef.current) {
          cancelAnimationFrame(animationRef.current);
        }
        drawFlatLine();
      };

      mediaRecorder.start();
      setIsRecording(true);
      setRecordingStatus("🔴 Recording... Speak naturally.");
      setAudioUrl(null);
      setAudioBlob(null);

    } catch (err) {
      console.error("Error accessing microphone:", err);
      alert("Could not access microphone. Please ensure you have granted permission.");
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      setRecordingStatus("✅ Recording stopped. Review and save below.");
    }
  };

  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      setAudioBlob(file);
      setAudioUrl(URL.createObjectURL(file));
      const defaultName = file.name.replace(/\.[^/.]+$/, "");
      setVoiceName(defaultName);
      setRecordingStatus(`📁 Uploaded file: ${file.name}. Review and save.`);
    }
  };

  const saveVoice = async () => {
    if (!audioBlob) return;
    const name = voiceName.trim();
    if (!name) {
      alert("Please enter a name for the voice.");
      return;
    }

    setIsSaving(true);
    const formData = new FormData();
    formData.append('audio', audioBlob, `${name}.wav`);
    formData.append('name', name);

    try {
      const response = await fetch('/api/upload_voice', {
        method: 'POST',
        body: formData
      });
      const data = await response.json();

      if (data.success) {
        setRecordingStatus(`✅ Saved successfully as ${data.filename}`);
        setVoiceName('');
        setAudioUrl(null);
        setAudioBlob(null);
        fetchVoices();
      } else {
        alert(data.error || "Failed to save voice.");
      }
    } catch (err) {
      console.error("Error saving voice:", err);
      alert("Network error while saving voice.");
    } finally {
      setIsSaving(false);
    }
  };

  const fetchVoices = async () => {
    try {
      const response = await fetch('/api/voices');
      const data = await response.json();
      if (data.voices) {
        setVoices(data.voices);
      }
    } catch (err) {
      console.error("Error fetching voices:", err);
    }
  };

  const precomputeVoice = async () => {
    if (!selectedVoice) {
      setPrecomputeStatus({ type: 'error', message: 'Please select a voice first.' });
      return;
    }

    setIsPrecomputing(true);
    setPrecomputeStatus({ type: '', message: '' });

    try {
      const response = await fetch('/api/precompute', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filename: selectedVoice })
      });
      const data = await response.json();

      if (response.ok && data.success) {
        setPrecomputeStatus({ type: 'success', message: `✅ ${data.message}` });
      } else {
        setPrecomputeStatus({ type: 'error', message: `❌ ${data.error || 'Failed to precompute'}` });
      }
    } catch (err) {
      console.error("Error precomputing:", err);
      setPrecomputeStatus({ type: 'error', message: '❌ Network error while precomputing.' });
    } finally {
      setIsPrecomputing(false);
    }
  };

  return (
    <>
      <div className="background-elements">
        <div className="blob blob-1"></div>
        <div className="blob blob-2"></div>
      </div>

      <div className="app-container">
        <header>
          <div className="logo-container">
            <div className="logo-icon">🎙️</div>
            <h1>Voice Studio</h1>
          </div>
          <p className="subtitle">Record and Precompute your AI Voice</p>
        </header>

        <main className="dashboard">
          {/* Recording Section */}
          <section className="card glass-panel">
            <h2>Record New Voice</h2>
            <div className="visualizer-container">
              <canvas ref={canvasRef}></canvas>
            </div>
            
            <div className="controls">
              <button 
                onClick={startRecording} 
                disabled={isRecording} 
                className="btn primary"
              >
                Start Recording
              </button>
              <button 
                onClick={stopRecording} 
                disabled={!isRecording} 
                className="btn danger"
              >
                Stop Recording
              </button>
              <label 
                className="btn accent" 
                style={{ cursor: isRecording ? 'not-allowed' : 'pointer', opacity: isRecording ? 0.5 : 1, margin: 0, width: 'auto' }}
              >
                Upload File
                <input 
                  type="file" 
                  accept="audio/*" 
                  onChange={handleFileUpload} 
                  disabled={isRecording} 
                  style={{ display: 'none' }} 
                />
              </label>
            </div>
            
            <div className="status-text" style={{ color: isRecording ? '#ef4444' : (audioUrl ? '#10b981' : 'var(--text-secondary)') }}>
              {recordingStatus}
            </div>

            {audioUrl && (
              <div className="save-controls">
                <audio src={audioUrl} controls></audio>
                <div className="input-group">
                  <input 
                    type="text" 
                    value={voiceName}
                    onChange={(e) => setVoiceName(e.target.value)}
                    placeholder="Enter voice name (e.g., teacher_v1)" 
                  />
                  <button 
                    onClick={saveVoice} 
                    disabled={isSaving} 
                    className="btn success"
                  >
                    {isSaving ? <><span className="loader"></span></> : "Save Voice"}
                  </button>
                </div>
              </div>
            )}
          </section>

          {/* Precompute Section */}
          <section className="card glass-panel">
            <h2>Precompute Voice</h2>
            <p className="description">Generate speaker embeddings to speed up voice synthesis.</p>
            
            <div className="input-group">
              <select 
                value={selectedVoice} 
                onChange={(e) => setSelectedVoice(e.target.value)}
              >
                <option value="" disabled>Select a voice</option>
                {voices.length === 0 ? (
                  <option value="" disabled>No voices found</option>
                ) : (
                  voices.map((v) => (
                    <option key={v} value={v}>{v}</option>
                  ))
                )}
              </select>
              <button onClick={fetchVoices} className="btn icon-btn" title="Refresh list">
                🔄
              </button>
            </div>

            <button 
              onClick={precomputeVoice} 
              disabled={isPrecomputing || !selectedVoice} 
              className="btn accent"
            >
              {isPrecomputing ? <><span className="loader"></span> Precomputing...</> : "Precompute Embeddings"}
            </button>
            
            {precomputeStatus.message && (
              <div className={`status-message ${precomputeStatus.type}`}>
                {precomputeStatus.message}
              </div>
            )}
          </section>
        </main>
      </div>
    </>
  );
}

export default App;
