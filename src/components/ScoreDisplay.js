import React from 'react';

function ScoreDisplay({ score, bestScore, linesCleared }) {
  return (
    <div className="score-container">
      <div className="score-item">
        <div className="score-label">Score</div>
        <div className="score-value">{score.toLocaleString()}</div>
      </div>
      <div className="score-item">
        <div className="score-label">Best</div>
        <div className="score-value">{bestScore.toLocaleString()}</div>
      </div>
      <div className="score-item">
        <div className="score-label">Lines</div>
        <div className="score-value">{linesCleared}</div>
      </div>
    </div>
  );
}

export default ScoreDisplay; 