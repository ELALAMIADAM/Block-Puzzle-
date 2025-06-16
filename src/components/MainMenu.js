import React from 'react';

function MainMenu({ onNavigate }) {
  return (
    <div className="main-menu">
      <div className="menu-container">
        <h1 className="game-title">
          🧩 Wood Block Puzzle
        </h1>
        
        <div className="menu-subtitle">
          Master the art of block placement
        </div>
        
        <div className="menu-buttons">
          <button 
            className="menu-btn play-btn"
            onClick={() => onNavigate('game')}
          >
            <div className="btn-icon">🎮</div>
            <div className="btn-content">
              <div className="btn-title">Play Game</div>
              <div className="btn-subtitle">Classic wood block puzzle</div>
            </div>
          </button>
          
          <button 
            className="menu-btn ai-btn"
            onClick={() => onNavigate('ai-learning')}
          >
            <div className="btn-icon">🤖</div>
            <div className="btn-content">
              <div className="btn-title">Watch AI Learn</div>
              <div className="btn-subtitle">Neural network training visualization</div>
            </div>
          </button>
          
          <button 
            className="menu-btn settings-btn"
            onClick={() => onNavigate('settings')}
          >
            <div className="btn-icon">⚙️</div>
            <div className="btn-content">
              <div className="btn-title">Settings</div>
              <div className="btn-subtitle">Game preferences & options</div>
            </div>
          </button>
        </div>
        
        <div className="menu-footer">
          <div className="game-features">
            <div className="feature-item">
              <span className="feature-icon">🧠</span>
              <span>Advanced AI with Deep Q-Learning</span>
            </div>
            <div className="feature-item">
              <span className="feature-icon">📊</span>
              <span>Real-time training visualization</span>
            </div>
            <div className="feature-item">
              <span className="feature-icon">🎯</span>
              <span>Strategic gameplay mechanics</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default MainMenu; 