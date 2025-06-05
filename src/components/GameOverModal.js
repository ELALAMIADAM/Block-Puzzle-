import React from 'react';

function GameOverModal({ score, bestScore, isNewBest, onRestart }) {
  return (
    <div className="game-over-overlay">
      <div className="game-over-modal">
        <div className="game-over-title">
          {isNewBest ? '🎉 New Best Score!' : 'Game Over'}
        </div>
        <div className="final-score">
          Final Score: {score.toLocaleString()}
        </div>
        {isNewBest && (
          <div style={{ fontSize: '18px', color: '#FFD700', marginBottom: '20px' }}>
            You beat your previous best of {bestScore !== score ? bestScore.toLocaleString() : '0'}!
          </div>
        )}
        <button className="btn" onClick={onRestart}>
          🔄 Play Again
        </button>
      </div>
    </div>
  );
}

export default GameOverModal; 