import React, { useState } from "react";
import { predictImage } from "./api";

function Upload() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);

  const handleUpload = async () => {
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);
    formData.append("model", "resnet");

    const res = await predictImage(formData);
    setResult(res.data);
  };

  return (
    <div>
      <h2>Upload Grape Leaf Image</h2>

      <input
        type="file"
        onChange={(e) => setFile(e.target.files[0])}
      />

      <button onClick={handleUpload}>Predict</button>

      {result && (
        <div>
          <h3>Prediction: {result.class}</h3>
          <p>Confidence: {(result.confidence * 100).toFixed(2)}%</p>
        </div>
      )}
    </div>
  );
}

export default Upload;