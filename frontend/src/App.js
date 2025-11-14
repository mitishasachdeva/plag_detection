import React, { useState } from 'react';
import './App.css'; 

function App() {
  const [code1, setCode1] = useState('');
  const [code2, setCode2] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!code1 || !code2) {
      setError('Both code snippets are required');
      return;
    }

    setLoading(true);
    setError('');
    setResult(null);

    try {
      const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';
const response = await fetch(`${API_URL}/compare_code`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ code1, code2 }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'API request failed');
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message || 'Error processing request');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const getColor = (prediction) =>
    prediction === 'Plagiarized' ? '--danger' : '--success';

  return (
    <div className="app">
      <header className="header">
        <h1>Code Plagiarism Detector</h1>
        <p>Compare two code snippets for semantic similarity using a trained Siamese Network</p>
      </header>

      <div className="container">
        <form onSubmit={handleSubmit} className="code-form">
          <div className="code-inputs">
            <div className="code-section">
              <h3>Code 1</h3>
              <textarea
                value={code1}
                onChange={(e) => setCode1(e.target.value)}
                placeholder="Paste the first code snippet here..."
                rows={18}
              />
            </div>

            <div className="code-section">
              <h3>Code 2</h3>
              <textarea
                value={code2}
                onChange={(e) => setCode2(e.target.value)}
                placeholder="Paste the second code snippet here..."
                rows={18}
              />
            </div>
          </div>

          <button
            type="submit"
            disabled={loading || !code1 || !code2}
            className="submit-btn"
          >
            {loading ? 'Analyzing...' : 'Compare Code'}
          </button>
        </form>

        {error && <div className="error">{error}</div>}

        {result && (
          <div className={`result ${result.prediction}`}>
            <h2>Analysis Result</h2>

            <div className="score-container">
              {/* Circular Progress Indicator */}
              <div className="score-circle">
                <div
                  className="circle-progress"
                  style={{
                    ['--ring']: `var(${getColor(result.prediction)})`,
                    ['--progress']: Math.max(
                      0,
                      Math.min(1, result.similarity_score ?? 0)
                    ),
                  }}
                ></div>
                <span>
                  {(result.similarity_score * 100).toFixed(1)}%
                </span>
              </div>

              <div className="score-details">
                <p>
                  Similarity Score:{' '}
                  <strong>{result.similarity_score.toFixed(4)}</strong>
                </p>
                <p>
                  Prediction:{' '}
                  <strong
                    className={
                      result.prediction === 'Plagiarized'
                        ? 'not-similar'
                        : 'similar'
                    }
                  >
                    {result.prediction}
                  </strong>
                </p>
              </div>
            </div>

            <div className="interpretation">
              <p>
                {result.prediction === 'Plagiarized'
                  ? 'These code snippets show a high degree of semantic similarity.'
                  : 'These code snippets appear to be logically distinct.'}
              </p>
            </div>
          </div>
        )}
      </div>

      <footer className="footer">
        <p>Powered by Siamese CodeBERT Network</p>
      </footer>
    </div>
  );
}

export default App;
