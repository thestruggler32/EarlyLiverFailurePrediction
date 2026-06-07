import React, { useState, useRef } from 'react';
import { 
  Activity, UploadCloud, AlertCircle, FileText, User, ActivitySquare, CheckCircle2 
} from 'lucide-react';
import { 
  Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, 
  BarElement, Title, Tooltip, Legend 
} from 'chart.js';
import { Line, Bar } from 'react-chartjs-2';

ChartJS.register(
  CategoryScale, LinearScale, PointElement, LineElement, BarElement, Title, Tooltip, Legend
);

function App() {
  const [age, setAge] = useState(55);
  const [gender, setGender] = useState('M');
  const [imageFile, setImageFile] = useState(null);
  const [csvFile, setCsvFile] = useState(null);
  const [csvText, setCsvText] = useState("");
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  
  const imageInputRef = useRef();
  const csvInputRef = useRef();

  const handleAnalyze = async () => {
    if (!imageFile && !csvFile && !csvText.trim()) {
      setError("Please upload at least one input (Ultrasound Image or EHR CSV data).");
      return;
    }
    
    setError(null);
    setLoading(true);
    
    const formData = new FormData();
    formData.append('age', age);
    formData.append('gender', gender);
    if (imageFile) formData.append('image', imageFile);
    if (csvFile) {
      formData.append('csv_file', csvFile);
    } else if (csvText.trim()) {
      const file = new File([csvText], "pasted.csv", { type: "text/csv" });
      formData.append('csv_file', file);
    }
    
    try {
      // Point this to backend URL where FastAPI is running
      const res = await fetch('http://localhost:8000/api/analyze', {
        method: 'POST',
        body: formData
      });
      
      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || "Analysis failed");
      }
      
      const data = await res.json();
      setResults(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const trendDataChart = {
    labels: results?.trends?.[0]?.dates || [],
    datasets: results?.trends?.map((t, i) => ({
      label: t.test,
      data: t.values,
      borderColor: ['#2dd4bf', '#0ea5e9', '#eab308', '#ec4899', '#a855f7', '#f97316'][i % 6],
      tension: 0.3,
      borderWidth: 2,
      pointRadius: 3
    })) || []
  };

  return (
    <div className="app-container">
      {/* Sidebar */}
      <aside className="sidebar">
        <div className="logo-container">
          <div className="logo-icon">
            <Activity size={24} />
          </div>
          <div className="logo-text">
            <h1>HepSense</h1>
            <p>Clinical Decision Support</p>
          </div>
        </div>

        <div className="card">
          <div className="card-title">
            <User size={16} /> Patient Information
          </div>
          <div className="input-row">
            <div className="input-group">
              <label>Age</label>
              <input type="number" value={age} onChange={e => setAge(e.target.value)} min={18} max={120} />
            </div>
            <div className="input-group">
              <label>Sex</label>
              <select value={gender} onChange={e => setGender(e.target.value)}>
                <option value="M">Male</option>
                <option value="F">Female</option>
              </select>
            </div>
          </div>
        </div>

        <div className="card">
          <div className="card-title">
            <UploadCloud size={16} /> Input Data
          </div>
          
          <div className="input-group">
            <label>Ultrasound Image</label>
            <div 
              className="dropzone" 
              onClick={() => imageInputRef.current.click()}
            >
              <input 
                type="file" 
                hidden 
                ref={imageInputRef} 
                accept="image/*"
                onChange={e => setImageFile(e.target.files[0])}
              />
              <UploadCloud className="dropzone-icon" size={24} />
              <div className="dropzone-text">Click to upload image</div>
              <div className="dropzone-sub">JPG, PNG, BMP</div>
            </div>
            {imageFile && (
              <div className="file-preview">
                {imageFile.name}
                <CheckCircle2 size={14} />
              </div>
            )}
          </div>

          <div className="input-group" style={{marginTop: '20px'}}>
            <label>Lab Results</label>
            <div 
              className="dropzone" 
              onClick={() => csvInputRef.current.click()}
              style={{marginBottom: '10px'}}
            >
              <input 
                type="file" 
                hidden 
                ref={csvInputRef} 
                accept=".csv"
                onChange={e => {
                  setCsvFile(e.target.files[0]);
                  setCsvText('');
                }}
              />
              <FileText className="dropzone-icon" size={24} />
              <div className="dropzone-text">Click to upload EHR CSV</div>
              <div className="dropzone-sub">charttime, lab_test, value</div>
            </div>
            {csvFile && (
              <div className="file-preview">
                {csvFile.name}
                <CheckCircle2 size={14} />
              </div>
            )}
            
            <div style={{ textAlign: 'center', fontSize: '12px', color: 'var(--text-secondary)', margin: '8px 0' }}>OR</div>
            
            <textarea 
              value={csvText}
              onChange={e => {
                setCsvText(e.target.value);
                setCsvFile(null);
              }}
              placeholder="Paste CSV data here..."
              style={{
                width: '100%', height: '80px', padding: '10px', 
                background: 'var(--bg-input)', border: '1px solid var(--border-color)',
                color: 'var(--text-primary)', borderRadius: 'var(--radius-md)',
                fontFamily: 'monospace', fontSize: '12px', resize: 'vertical',
                outline: 'none', transition: 'border-color 0.2s, box-shadow 0.2s'
              }}
              onFocus={e => {
                e.target.style.borderColor = 'var(--accent-primary)';
                e.target.style.boxShadow = '0 0 0 2px rgba(2, 132, 199, 0.2)';
              }}
              onBlur={e => {
                e.target.style.borderColor = 'var(--border-color)';
                e.target.style.boxShadow = 'none';
              }}
            />
          </div>
        </div>

        <button 
          className="btn-primary" 
          onClick={handleAnalyze} 
          disabled={loading}
        >
          {loading ? <ActivitySquare className="spinner" size={18} /> : <ActivitySquare size={18} />}
          {loading ? 'Analyzing...' : 'Run Assessment'}
        </button>

        {error && (
          <div style={{color: '#ef4444', fontSize: '13px', padding: '10px', background: 'rgba(239, 68, 68, 0.1)', borderRadius: '8px'}}>
            {error}
          </div>
        )}
      </aside>

      {/* Main Content */}
      <main className="main-content">
        {!results ? (
          <div className="empty-state">
            <Activity className="empty-state-icon" size={64} />
            <h2>Ready for Analysis</h2>
            <p>Upload an ultrasound image or laboratory CSV to generate risk stratification.</p>
          </div>
        ) : (
          <div className="results-container">
            <div className="header">
              <h2>Assessment Report</h2>
            </div>

            {/* Alert Banner */}
            <div className={`alert-banner alert-${results.recommendation.severity_level.replace(' ', '-')}`}>
              <div className="alert-title">HepSense Risk Classification</div>
              <div className="alert-level">
                <AlertCircle size={28} /> {results.recommendation.severity_level}
              </div>
              <div className="alert-desc">{results.recommendation.recommendation}</div>
            </div>

            {/* Top Metrics */}
            <div className="results-grid">
              <div className="card">
                <div className="card-title">Fibrosis Stage (Vision AI)</div>
                <div className="metric-value">{results.vision.stage}</div>
                <div className="metric-sub">
                  Confidence: {results.vision.confidence ? (results.vision.confidence * 100).toFixed(1) + '%' : 'N/A'}
                </div>
              </div>
              <div className="card">
                <div className="card-title">Decompensation Risk (Clinical AI)</div>
                <div className="metric-value">{results.clinical.risk_label}</div>
                <div className="metric-sub">
                  Risk Probability: {results.clinical.risk_probability ? (results.clinical.risk_probability * 100).toFixed(1) + '%' : 'N/A'}
                </div>
              </div>
              <div className="card" style={{gridColumn: '1 / -1'}}>
                <div className="card-title">Recommended Management Plan</div>
                <ul className="action-list">
                  {results.recommendation.actions.map((act, i) => (
                    <li key={i}>{act}</li>
                  ))}
                </ul>
              </div>
            </div>

            {/* Diagnostics */}
            <div className="diagnostic-grid">
              {results.vision.gradcam_overlay && (
                <div className="card">
                  <div className="card-title">Ultrasound Grad-CAM Analysis</div>
                  <img 
                    src={`data:image/png;base64,${results.vision.gradcam_overlay}`} 
                    alt="GradCAM" 
                    className="cam-image" 
                  />
                </div>
              )}
              
              {results.trends && results.trends.length > 0 && (
                <div className="card" style={{gridColumn: results.vision.gradcam_overlay ? '2' : '1 / -1'}}>
                  <div className="card-title">Normalized Laboratory Trends</div>
                  <div className="chart-container">
                    <Line 
                      data={trendDataChart} 
                      options={{
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                          legend: { labels: { color: '#94a3b8' } }
                        },
                        scales: {
                          x: { grid: { color: '#334155' }, ticks: { color: '#94a3b8' } },
                          y: { grid: { color: '#334155' }, ticks: { color: '#94a3b8' } }
                        }
                      }} 
                    />
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
