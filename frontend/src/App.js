import React, { useState } from "react";
import axios from "axios";
import "./App.css";

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e) => {
    const f = e.target.files[0];
    setFile(f);
    setPreview(URL.createObjectURL(f));
  };

  const handlePredict = async () => {
    if (!file) return alert("Upload image first");

    const formData = new FormData();
    formData.append("file", file);

    setLoading(true);

    try {
      const res = await axios.post(
        "http://localhost:5000/predict",
        formData,
        { timeout: 60000 }
      );

      console.log("API Response:", res.data);
      setResult(res.data);

    } catch (err) {
      console.error(err);
      alert("Prediction failed");
    }

    setLoading(false);
  };

  return (
    <div className="app">

      {/* HERO */}
      <div className="hero">
        <h1>🍇 Grape Leaf Disease Detection</h1>
        <p>
          Upload a grape leaf image and our AI system will analyze it using
          multiple deep learning models.
        </p>
      </div>

      {/* UPLOAD */}
      <div className="card">
        <input type="file" onChange={handleFileChange} />

        {preview && <img src={preview} alt="preview" />}

        <button onClick={handlePredict}>
          {loading ? "Analyzing..." : "Predict"}
        </button>

        {loading && <p>Running models... ⏳</p>}
      </div>

      {/* RESULT */}
      {result && (
        <div className="result">
          <h2>Final Prediction</h2>
          <h1 className="highlight">{result.final_prediction}</h1>

          <p>
            Confidence: {(result.final_confidence * 100).toFixed(2)}%
          </p>

          {/* ❌ REMOVED DESCRIPTION & SOLUTION */}

          {/* MODELS */}
          <h3>Model Comparison</h3>
          <div className="models">
            {Object.entries(result.models).map(([name, data]) => (
              <div
                key={name}
                className={
                  name === result.best_model ? "model best" : "model"
                }
              >
                <h4>{name}</h4>
                <p>{data.prediction}</p>
                <p>{(data.confidence * 100).toFixed(2)}%</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* FOOTER */}
      <div className="footer">
        <p>AI Project • Grape Disease Detection • Deep Learning</p>
      </div>
    </div>
  );
}

export default App;