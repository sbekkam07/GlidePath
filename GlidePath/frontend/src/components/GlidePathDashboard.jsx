import React, { useState } from "react";
import "./GlidePathDashboard.css";

const BACKEND_BASE = "http://127.0.0.1:8000";

function formatNumber(value, digits = 2) {
  const num = Number(value);
  return Number.isFinite(num) ? num.toFixed(digits) : "N/A";
}

function toAbsoluteUrl(path) {
  if (!path) return "";
  if (path.startsWith("http://") || path.startsWith("https://")) return path;
  return `${BACKEND_BASE}${path}`;
}

export default function GlidePathDashboard() {
  const [file, setFile] = useState(null);
  const [airportCode, setAirportCode] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [analysis, setAnalysis] = useState(null);
  const [weather, setWeather] = useState(null);
  const [weatherError, setWeatherError] = useState("");

  const handleSubmit = async (event) => {
    event.preventDefault();
    setError("");
    setWeatherError("");
    setAnalysis(null);
    setWeather(null);

    if (!file) {
      setError("Please select a video file first.");
      return;
    }

    setLoading(true);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const analyzeResponse = await fetch(`${BACKEND_BASE}/analyze-video`, {
        method: "POST",
        body: formData,
      });

      if (!analyzeResponse.ok) {
        const text = await analyzeResponse.text();
        throw new Error(text || "Unable to analyze video.");
      }

      const analysisData = await analyzeResponse.json();
      setAnalysis(analysisData);

      const trimmedAirport = airportCode.trim().toUpperCase();
      if (trimmedAirport) {
        try {
          const weatherResponse = await fetch(
            `${BACKEND_BASE}/weather/${encodeURIComponent(trimmedAirport)}`
          );
          if (!weatherResponse.ok) {
            throw new Error(`Weather endpoint returned ${weatherResponse.status}`);
          }
          const weatherData = await weatherResponse.json();
          setWeather(weatherData);
        } catch (weatherErr) {
          setWeatherError(`Weather lookup failed: ${weatherErr.message}`);
        }
      }
    } catch (err) {
      setError(err.message || "An unexpected error occurred.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="dashboard">
      <header className="header">
        <h1>GlidePath MVP Dashboard</h1>
        <p>Upload runway approach footage and review a demo readiness report.</p>
      </header>

      <form onSubmit={handleSubmit} className="upload-form">
        <label htmlFor="video-file" className="form-label">
          Landing video
        </label>
        <input
          id="video-file"
          type="file"
          accept="video/*,.mp4,.mov,.avi,.mkv,.webm"
          onChange={(event) => setFile(event.target.files?.[0] || null)}
        />

        <label htmlFor="airport-code" className="form-label">
          Airport code (optional)
        </label>
        <input
          id="airport-code"
          type="text"
          placeholder="KJFK"
          value={airportCode}
          onChange={(event) => setAirportCode(event.target.value)}
        />

        <button type="submit" disabled={loading}>
          {loading ? "Analyzing..." : "Analyze"}
        </button>
      </form>

      {loading && <p className="status">Analyzing video, generating previews...</p>}
      {error && <p className="status error">{error}</p>}

      {analysis && (
        <div className="results-grid">
          <section className="card">
            <h2>Runway Analysis</h2>
            <div className="metric-grid">
              <p>
                <strong>Alignment:</strong> {analysis.alignment}
              </p>
              <p>
                <strong>Stability:</strong> {analysis.stability}
              </p>
              <p>
                <strong>Confidence:</strong> {formatNumber(analysis.confidence * 100)}%
              </p>
              <p>
                <strong>Frame Count:</strong> {analysis.frame_count}
              </p>
              <p>
                <strong>Avg Offset:</strong> {formatNumber(analysis.average_offset_px)} px
              </p>
              {analysis.output_video && <p>
                <strong>Output:</strong>{" "}
                <a href={toAbsoluteUrl(analysis.output_video)} target="_blank" rel="noreferrer">
                  Download/View Video
                </a>
              </p>}
              {analysis.output_dir && (
                <p>
                  <strong>Output folder:</strong> {analysis.output_dir}
                </p>
              )}
            </div>

            {analysis.output_video && (
              <div className="video-preview">
                <h3>Processed Video</h3>
                <video controls preload="metadata">
                  <source src={toAbsoluteUrl(analysis.output_video)} type="video/mp4" />
                </video>
              </div>
            )}

            <h3>Frame previews</h3>
            {analysis.preview_frames?.length > 0 ? (
              <div className="preview-grid">
                {analysis.preview_frames.map((path) => (
                  <img
                    key={path}
                    src={toAbsoluteUrl(path)}
                    alt="Frame preview"
                    className="preview-image"
                  />
                ))}
              </div>
            ) : (
                <p className="muted">No preview frames returned.</p>
            )}
          </section>

          <section className="card">
            <h2>Offsets</h2>
            {analysis.offsets?.length > 0 ? (
              <ul>
                {analysis.offsets.map((offset, index) => (
                  <li key={`${offset}-${index}`}>
                    Frame {index + 1}: {formatNumber(offset)} px
                  </li>
                ))}
              </ul>
            ) : (
              <p className="muted">No offsets returned.</p>
            )}
          </section>
        </div>
      )}

      {(weather || weatherError) && (
        <section className="card weather-card">
          <h2>Weather</h2>
          {weatherError && <p className="status error">{weatherError}</p>}
          {weather && (
            <>
              <p>
                <strong>Airport:</strong> {weather.airport_code}
              </p>
              <p>
                <strong>METAR:</strong> {weather.metar_raw || "N/A"}
              </p>
              <p>
                <strong>Wind Direction:</strong>{" "}
                {weather.direction_degrees != null ? `${weather.direction_degrees}°` : "N/A"}
              </p>
              <p>
                <strong>Wind Speed:</strong>{" "}
                {weather.speed_kt != null ? `${weather.speed_kt} kt` : "N/A"}
              </p>
              <p>
                <strong>Crosswind:</strong>{" "}
                {weather.crosswind_kt != null ? `${weather.crosswind_kt} kt` : "N/A"}
              </p>
              <p>
                <strong>Headwind:</strong>{" "}
                {weather.headwind_kt != null ? `${weather.headwind_kt} kt` : "N/A"}
              </p>
            </>
          )}
        </section>
      )}
    </div>
  );
}
