import React from 'react';

function ScoreDisplay({ score, bestScore, linesCleared, difficulty }) {
  // Ensure values are numbers and handle undefined cases
  const safeScore = typeof score === 'number' ? score : 0;
  const safeBestScore = typeof bestScore === 'number' ? bestScore : 0;
  const safeLinesCleared = typeof linesCleared === 'number' ? linesCleared : 0;
  
  return (
    <div className="score-container">
      <div className="score-item">
        <div className="score-label">Score</div>
        <div className="score-value">{safeScore.toLocaleString()}</div>
      </div>
      <div className="score-item">
        <div className="score-label">Best</div>
        <div className="score-value">{safeBestScore.toLocaleString()}</div>
      </div>
      <div className="score-item">
        <div className="score-label">Cleared</div>
        <div className="score-value">{safeLinesCleared}</div>
      </div>
      <div className="score-item">
        <div className="score-label">Mode</div>
        <div className="score-value">
          {difficulty === 'hard' ? 'Hard' : 'Normal'}
        </div>
      </div>
    </div>
  );
}

export default ScoreDisplay; 