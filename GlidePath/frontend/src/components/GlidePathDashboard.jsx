import React, { useState, useRef, useEffect, useCallback } from "react";
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine,
} from "recharts";
import "./GlidePathDashboard.css";

const RAW_BACKEND =
  (process.env.REACT_APP_API_URL || process.env.VITE_API_URL || "").trim() ||
  (process.env.NODE_ENV === "development"
    ? "http://127.0.0.1:8000"
    : "https://glidepath.onrender.com");
const BACKEND = RAW_BACKEND.replace(/\/+$/, "");
const C = {
  sky: "#2DAAE1", yellow: "#F4B400", green: "#2ECC71",
  warning: "#F39C12", danger: "#E74C3C", dim: "#5A7A90",
  text: "#C8D8E8", border: "#1A3448", panel: "#0D1F2D",
};

function abs(p) {
  if (!p) return "";
  return p.startsWith("http") ? p : `${BACKEND}${p}`;
}
function fmt(v, d = 2) {
  const n = Number(v);
  return Number.isFinite(n) ? n.toFixed(d) : "N/A";
}

// ─── Scroll-tracking plane ──────────────────────────────────────────────────
function ScrollPlane() {
  const [pct, setPct] = useState(0);

  useEffect(() => {
    const onScroll = () => {
      const scrollY = window.scrollY;
      const docH = document.documentElement.scrollHeight - window.innerHeight;
      setPct(docH > 0 ? Math.min(scrollY / docH, 1) : 0);
    };
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  const top = pct * (window.innerHeight - 48);

  return (
    <div className="scroll-plane-track">
      {/* Trail behind the plane */}
      <div className="scroll-plane-trail" style={{ height: top }} />
      {/* Plane icon */}
      <svg
        className="scroll-plane-svg"
        style={{ top }}
        viewBox="0 0 36 36"
        fill="none"
      >
        {/* Fuselage */}
        <ellipse cx="18" cy="18" rx="4" ry="12" fill="#2DAAE1" />
        {/* Wings */}
        <path d="M6 21 L18 16 L30 21 Z" fill="#2DAAE1" opacity="0.9" />
        {/* Tail */}
        <path d="M13 28 L18 24 L23 28 Z" fill="#2DAAE1" opacity="0.7" />
        {/* Engine glow */}
        <circle cx="18" cy="30" r="2.5" fill="#F4B400" opacity="0.6" />
      </svg>
    </div>
  );
}

// ─── Upload zone ────────────────────────────────────────────────────────────
function UploadZone({ file, setFile, airportCode, setAirportCode, onAnalyze, loading }) {
  const inputRef = useRef(null);
  const [dragOver, setDragOver] = useState(false);

  const handleDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    const f = e.dataTransfer.files?.[0];
    if (f) setFile(f);
  };

  return (
    <div
      className={`gp-upload-zone ${dragOver ? "drag-over" : ""}`}
      onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
      onDragLeave={() => setDragOver(false)}
      onDrop={handleDrop}
      onClick={() => inputRef.current?.click()}
    >
      <div className="gp-upload-icon">✈️</div>
      <div className="gp-upload-title">DROP LANDING VIDEO HERE</div>
      <div className="gp-upload-sub">Supports MP4 · MOV · AVI</div>

      <input
        ref={inputRef}
        className="gp-file-input"
        type="file"
        accept="video/*,.mp4,.mov,.avi,.mkv,.webm"
        onChange={(e) => setFile(e.target.files?.[0] || null)}
      />

      {file && <div className="gp-file-name">📎 {file.name}</div>}

      <input
        className="gp-airport-input"
        type="text"
        placeholder="KJFK"
        value={airportCode}
        onChange={(e) => setAirportCode(e.target.value)}
        onClick={(e) => e.stopPropagation()}
        style={{ marginTop: 20 }}
      />

      <div className="gp-upload-actions">
        <button
          className="gp-btn-primary"
          disabled={!file || loading}
          onClick={(e) => { e.stopPropagation(); onAnalyze(); }}
        >
          {loading ? "ANALYZING…" : "ANALYZE"}
        </button>
      </div>
    </div>
  );
}

// ─── Offset chart ────────────────────────────────────────────────────────────
function OffsetChart({ offsets }) {
  if (!offsets || offsets.length === 0) return null;

  // Downsample to ~100 points so the chart stays clean
  const raw = offsets.map((v) => Number(v) || 0);
  const MAX_POINTS = 100;
  const step = Math.max(1, Math.floor(raw.length / MAX_POINTS));
  const data = [];
  for (let i = 0; i < raw.length; i += step) {
    const slice = raw.slice(i, Math.min(i + step, raw.length));
    const avg = slice.reduce((a, b) => a + b, 0) / slice.length;
    data.push({ frame: i + 1, offset: parseFloat(avg.toFixed(1)) });
  }

  const CustomTip = ({ active, payload, label }) => {
    if (!active || !payload?.length) return null;
    const val = payload[0]?.value;
    const col = Math.abs(val) < 20 ? C.green : Math.abs(val) < 40 ? C.yellow : C.danger;
    return (
      <div style={{
        background: "#06121C", border: `1px solid ${C.border}`,
        borderRadius: 8, padding: "8px 12px", fontSize: 12,
      }}>
        <div style={{ color: C.dim }}>Frame {label}</div>
        <div style={{ color: col, fontWeight: 700 }}>{val?.toFixed(1)} px</div>
      </div>
    );
  };

  return (
    <div className="gp-card">
      <div className="gp-card-label">CENTERLINE OFFSET · PX OVER TIME</div>
      <ResponsiveContainer width="100%" height={180}>
        <AreaChart data={data} margin={{ top: 4, right: 4, bottom: 0, left: -20 }}>
          <defs>
            <linearGradient id="oGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor={C.sky} stopOpacity={0.3} />
              <stop offset="95%" stopColor={C.sky} stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid stroke={C.border} vertical={false} />
          <XAxis dataKey="frame" tick={{ fill: C.dim, fontSize: 10 }} axisLine={false} tickLine={false}
            tickCount={8} />
          <YAxis tick={{ fill: C.dim, fontSize: 10 }} axisLine={false} tickLine={false} />
          <Tooltip content={<CustomTip />} />
          <ReferenceLine y={0} stroke={C.green} strokeDasharray="4 4" strokeOpacity={0.5}
            label={{ value: "CENTER", fill: C.green, fontSize: 9 }} />
          <Area type="monotone" dataKey="offset" stroke={C.sky} strokeWidth={2}
            fill="url(#oGrad)" dot={false} activeDot={{ r: 3, fill: C.sky }} />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}

// ─── Stability heatmap ──────────────────────────────────────────────────────
function StabilityHeatmap({ offsets }) {
  if (!offsets || offsets.length === 0) return null;

  // Bucket offsets into ~50 segments
  const raw = offsets.map((v) => Math.abs(Number(v) || 0));
  const BUCKETS = Math.min(50, raw.length);
  const bucketSize = Math.max(1, Math.floor(raw.length / BUCKETS));
  const cells = [];
  for (let i = 0; i < raw.length; i += bucketSize) {
    const slice = raw.slice(i, Math.min(i + bucketSize, raw.length));
    const avg = slice.reduce((a, b) => a + b, 0) / slice.length;
    const startFrame = i + 1;
    const endFrame = Math.min(i + bucketSize, raw.length);
    let color, label;
    if (avg < 20) { color = C.green; label = "Aligned"; }
    else if (avg < 35) { color = C.yellow; label = "Caution"; }
    else if (avg < 50) { color = C.warning; label = "Drift"; }
    else { color = C.danger; label = "Unstable"; }
    cells.push({ color, label, startFrame, endFrame, avg });
  }

  return (
    <div className="gp-card">
      <div className="gp-card-label">LANDING STABILITY HEATMAP</div>
      <div className="gp-heatmap-bar">
        {cells.map((c, i) => (
          <div key={i} className="gp-heat-cell"
            style={{ background: c.color }}
            title={`Frames ${c.startFrame}–${c.endFrame}: ${c.label} (avg ${c.avg.toFixed(0)} px)`}
          />
        ))}
      </div>
      <div className="gp-heat-legend">
        {[
          { color: C.green, label: "Aligned (<20 px)" },
          { color: C.yellow, label: "Caution (20–35 px)" },
          { color: C.warning, label: "Drift (35–50 px)" },
          { color: C.danger, label: "Unstable (>50 px)" },
        ].map((l) => (
          <div key={l.label} className="gp-heat-legend-item">
            <div className="gp-heat-legend-dot" style={{ background: l.color }} />
            {l.label}
          </div>
        ))}
      </div>
    </div>
  );
}

// ─── Report card ────────────────────────────────────────────────────────────
function ReportCard({ analysis }) {
  const conf = (analysis.confidence || 0) * 100;
  const avgOff = Math.abs(analysis.average_offset_px || 0);

  // Compute grades from real data
  const alignScore = avgOff < 15 ? 95 : avgOff < 25 ? 80 : avgOff < 40 ? 65 : 40;
  const alignGrade = alignScore >= 90 ? "A" : alignScore >= 80 ? "B+" : alignScore >= 65 ? "B" : alignScore >= 50 ? "C" : "D";

  const stabLabel = analysis.stability || "unknown";
  const stabScore = stabLabel === "stable" ? 92 : stabLabel === "caution" ? 70 : 45;
  const stabGrade = stabScore >= 90 ? "A" : stabScore >= 70 ? "B" : stabScore >= 50 ? "C" : "D";

  const confScore = Math.min(conf, 100);
  const confGrade = confScore >= 90 ? "A" : confScore >= 75 ? "B+" : confScore >= 60 ? "B" : "C";

  const grades = [
    { label: "Runway Alignment", grade: alignGrade, score: alignScore, color: C.sky },
    { label: "Approach Stability", grade: stabGrade, score: stabScore, color: C.yellow },
    { label: "Detection Confidence", grade: confGrade, score: confScore, color: C.green },
  ];

  const statusColor = (s) =>
    s === "good" ? C.green : s === "caution" ? C.yellow : s === "danger" ? C.danger : C.dim;

  const alignStatus = (analysis.alignment || "").includes("aligned") ? "good" :
    (analysis.alignment || "").includes("drift") ? "caution" : "danger";

  const stats = [
    { label: "Avg Offset", value: `${fmt(analysis.average_offset_px)} px`, status: avgOff < 20 ? "good" : avgOff < 35 ? "caution" : "danger" },
    { label: "Frame Count", value: String(analysis.frame_count || 0), status: "neutral" },
    { label: "Confidence", value: `${fmt(conf, 1)}%`, status: conf > 75 ? "good" : conf > 50 ? "caution" : "danger" },
    { label: "Alignment", value: (analysis.alignment || "UNKNOWN").toUpperCase(), status: alignStatus },
    { label: "Stability", value: (analysis.stability || "UNKNOWN").toUpperCase(), status: stabLabel === "stable" ? "good" : stabLabel === "caution" ? "caution" : "danger" },
    { label: "Offset Frames", value: String((analysis.offsets || []).length), status: "neutral" },
  ];

  return (
    <div className="gp-card">
      <div className="gp-card-label">FLIGHT REPORT CARD</div>
      {grades.map((g) => (
        <div key={g.label} className="gp-grade-row">
          <div className="gp-grade-header">
            <span className="gp-grade-label">{g.label}</span>
            <span className="gp-grade-value" style={{ color: g.color }}>{g.grade}</span>
          </div>
          <div className="gp-grade-bar">
            <div className="gp-grade-fill" style={{
              width: `${g.score}%`,
              background: `linear-gradient(90deg, ${g.color}80, ${g.color})`,
            }} />
          </div>
        </div>
      ))}
      <div className="gp-divider" />
      <div className="gp-stat-grid">
        {stats.map((s) => (
          <div key={s.label} className="gp-stat-cell">
            <div className="gp-stat-cell-label">{s.label}</div>
            <div className="gp-stat-cell-value" style={{ color: statusColor(s.status) }}>
              {s.value}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ─── AI summary ─────────────────────────────────────────────────────────────
function AISummary({ analysis }) {
  const [visible, setVisible] = useState(false);
  useEffect(() => { const t = setTimeout(() => setVisible(true), 400); return () => clearTimeout(t); }, []);

  const avgOff = analysis.average_offset_px || 0;
  const conf = ((analysis.confidence || 0) * 100).toFixed(1);
  const align = analysis.alignment || "unknown";
  const drift = align.includes("right") ? "rightward" : align.includes("left") ? "leftward" : "minimal";
  const correction = align.includes("right") ? "left" : align.includes("left") ? "right" : null;

  return (
    <div className="gp-card" style={{
      borderColor: `${C.sky}40`,
      background: `linear-gradient(135deg, ${C.panel}, #0a1929)`,
      opacity: visible ? 1 : 0,
      transform: visible ? "translateY(0)" : "translateY(12px)",
      transition: "opacity 0.6s ease, transform 0.6s ease",
    }}>
      <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 16 }}>
        <div style={{
          width: 8, height: 8, borderRadius: "50%",
          background: C.sky, boxShadow: `0 0 10px ${C.sky}`,
          animation: "pulse 1.5s ease-in-out infinite alternate",
        }} />
        <span className="gp-card-label" style={{ marginBottom: 0, color: C.sky }}>
          AI FLIGHT ANALYSIS
        </span>
      </div>
      <p style={{ color: C.text, fontSize: 14, lineHeight: 1.7 }}>
        The approach showed <strong style={{ color: drift === "minimal" ? C.green : C.yellow }}>{drift} drift</strong> with
        an average offset of <strong style={{ color: Math.abs(avgOff) < 20 ? C.green : C.yellow }}>{fmt(avgOff)} px</strong>.
        Detection confidence held at <strong style={{ color: C.green }}>{conf}%</strong> across {analysis.frame_count || 0} frames.
      </p>
      {correction && (
        <div style={{
          marginTop: 16, padding: "12px 16px",
          background: `${C.yellow}15`, borderLeft: `3px solid ${C.yellow}`,
          borderRadius: "0 8px 8px 0", fontSize: 13, color: C.yellow,
        }}>
          💡 <strong>Suggested correction:</strong> Apply small {correction} rudder input to correct {drift} drift before flare.
        </div>
      )}
    </div>
  );
}

// ─── Weather card ───────────────────────────────────────────────────────────
function WeatherCard({ weather }) {
  if (!weather) return null;
  const items = [
    { label: "Airport", value: weather.airport_code },
    { label: "Wind Dir", value: weather.direction_degrees != null ? `${weather.direction_degrees}°` : "N/A" },
    { label: "Wind Speed", value: weather.speed_kt != null ? `${weather.speed_kt} kt` : "N/A" },
    { label: "Crosswind", value: weather.crosswind_kt != null ? `${weather.crosswind_kt} kt` : "N/A" },
    { label: "Headwind", value: weather.headwind_kt != null ? `${weather.headwind_kt} kt` : "N/A" },
  ];
  return (
    <div className="gp-card" style={{ marginTop: 20 }}>
      <div className="gp-card-label">WEATHER · METAR</div>
      {weather.metar_raw && (
        <div style={{ fontSize: 11, color: C.dim, marginBottom: 12, wordBreak: "break-all" }}>
          {weather.metar_raw}
        </div>
      )}
      <div className="gp-weather-grid">
        {items.map((it) => (
          <div key={it.label} className="gp-stat-cell">
            <div className="gp-stat-cell-label">{it.label}</div>
            <div className="gp-stat-cell-value" style={{ color: C.text }}>{it.value}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════════
// MAIN DASHBOARD
// ═══════════════════════════════════════════════════════════════════════════════
export default function GlidePathDashboard() {
  const [stage, setStage] = useState("hero"); // hero | uploading | results
  const [file, setFile] = useState(null);
  const [airportCode, setAirportCode] = useState("");
  const [error, setError] = useState("");
  const [analysis, setAnalysis] = useState(null);
  const [weather, setWeather] = useState(null);

  const handleAnalyze = useCallback(async () => {
    if (!file) return;
    setError("");
    setAnalysis(null);
    setWeather(null);
    setStage("uploading");

    try {
      const fd = new FormData();
      fd.append("file", file);
      const res = await fetch(`${BACKEND}/analyze-video`, { method: "POST", body: fd });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      setAnalysis(data);

      // Fetch weather if code provided
      const code = airportCode.trim().toUpperCase();
      if (code) {
        try {
          const wr = await fetch(`${BACKEND}/weather/${encodeURIComponent(code)}`);
          if (wr.ok) setWeather(await wr.json());
        } catch {}
      }

      setStage("results");
    } catch (err) {
      setError(err.message || "Analysis failed.");
      setStage("hero");
    }
  }, [file, airportCode]);

  const handleDownload = async () => {
    if (!analysis?.overlay_video) return;
    try {
      const r = await fetch(abs(analysis.overlay_video));
      const blob = await r.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url; a.download = "glidepath_annotated.mp4";
      document.body.appendChild(a); a.click(); a.remove();
      URL.revokeObjectURL(url);
    } catch (e) {
      setError("Download failed.");
    }
  };

  return (
    <div className="gp-root">
      <div className="grid-bg" />
      <ScrollPlane />

      {/* Nav */}
      <nav className="gp-nav">
        <div className="gp-nav-brand">
          <span style={{ fontSize: 20 }}>✈</span>
          <span className="gp-nav-title">GLIDEPATH</span>
          <span className="gp-nav-badge">MVP</span>
        </div>
        <div className="gp-nav-right">RUNWAY ANALYSIS SYSTEM</div>
      </nav>

      <div className="gp-content">

        {/* ── HERO ── */}
        {stage === "hero" && (
          <div className="gp-hero">
            <div className="gp-hero-tag">COMPUTER VISION · FLIGHT ANALYSIS</div>
            <h1 className="gp-hero-title">ANALYZE AIRCRAFT<br />LANDINGS</h1>
            <p className="gp-hero-sub">
              Upload runway approach footage · Get real-time alignment analysis
            </p>
            <UploadZone
              file={file} setFile={setFile}
              airportCode={airportCode} setAirportCode={setAirportCode}
              onAnalyze={handleAnalyze} loading={false}
            />
            {error && <div className="gp-error">{error}</div>}
          </div>
        )}

        {/* ── UPLOADING ── */}
        {stage === "uploading" && (
          <div className="gp-uploading">
            <div className="gp-spinner" />
            <div className="gp-uploading-title">ANALYZING LANDING</div>
            <div className="gp-uploading-sub">
              Processing frames · Detecting runway · Computing offsets
            </div>
            <div className="gp-progress-bar">
              <div className="gp-progress-fill" />
            </div>
          </div>
        )}

        {/* ── RESULTS ── */}
        {stage === "results" && analysis && (
          <div className="gp-results">

            {/* Status banner */}
            <div className="gp-banner">
              <div className="gp-banner-stats">
                {airportCode.trim() && (
                  <>
                    <div>
                      <div className="gp-banner-stat-label">AIRPORT</div>
                      <div className="gp-banner-stat-value" style={{ color: C.sky }}>
                        {airportCode.trim().toUpperCase()}
                      </div>
                    </div>
                    <div className="gp-banner-divider" />
                  </>
                )}
                <div>
                  <div className="gp-banner-stat-label">STATUS</div>
                  <div className="gp-banner-stat-value" style={{
                    color: (analysis.alignment || "").includes("aligned") ? C.green : C.yellow,
                  }}>
                    {(analysis.alignment || "UNKNOWN").toUpperCase().replace("_", " ")}
                  </div>
                </div>
                <div className="gp-banner-divider" />
                <div>
                  <div className="gp-banner-stat-label">CONFIDENCE</div>
                  <div className="gp-banner-stat-value" style={{ color: C.green }}>
                    {fmt((analysis.confidence || 0) * 100, 1)}%
                  </div>
                </div>
                <div className="gp-banner-divider" />
                <div>
                  <div className="gp-banner-stat-label">FRAMES</div>
                  <div className="gp-banner-stat-value" style={{ color: C.text }}>
                    {analysis.frame_count || 0}
                  </div>
                </div>
              </div>
              <button className="gp-btn-ghost" onClick={() => setStage("hero")}>
                ↩ NEW ANALYSIS
              </button>
            </div>

            {/* Main grid */}
            <div className="gp-grid">
              {/* Left column */}
              <div className="gp-col">
                {/* Video player */}
                <div className="gp-card">
                  <div className="gp-card-label">ANNOTATED VIDEO</div>
                  {analysis.overlay_video ? (
                    <>
                      <div className="gp-video-wrap">
                        <video
                          className="gp-video"
                          src={abs(analysis.overlay_video)}
                          controls
                          playsInline
                          preload="metadata"
                        />
                      </div>
                      <button className="gp-download-btn" onClick={handleDownload}>
                        ⤓ DOWNLOAD VIDEO
                      </button>
                    </>
                  ) : (
                    <div className="gp-video-none">No annotated video available</div>
                  )}
                </div>

                {/* Frame previews — right below the video */}
                {analysis.preview_frames?.length > 0 && (
                  <div className="gp-card">
                    <div className="gp-card-label">FRAME PREVIEWS</div>
                    <div className="gp-preview-grid">
                      {analysis.preview_frames.map((p, i) => (
                        <img key={i} src={abs(p)} alt={`Frame ${i + 1}`} className="gp-preview-img" />
                      ))}
                    </div>
                  </div>
                )}

                {/* Offset graph */}
                <OffsetChart offsets={analysis.offsets} />

                {/* Heatmap */}
                <StabilityHeatmap offsets={analysis.offsets} />
              </div>

              {/* Right column */}
              <div className="gp-col">
                <ReportCard analysis={analysis} />
                <AISummary analysis={analysis} />
              </div>
            </div>

            {/* Weather */}
            <WeatherCard weather={weather} />
          </div>
        )}
      </div>
    </div>
  );
}
