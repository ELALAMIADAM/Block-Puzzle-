import React, { useState, useEffect } from 'react';

function SettingsView({ onNavigate }) {
  const [settings, setSettings] = useState({
    soundEnabled: true,
    animationsEnabled: true,
    difficulty: 'normal',
    autoSave: true,
    theme: 'dark',
    aiSpeed: 500,
    showHints: true,
    gridSize: 9
  });

  // Load settings from localStorage on mount
  useEffect(() => {
    const savedSettings = localStorage.getItem('woodBlockPuzzleSettings');
    if (savedSettings) {
      setSettings(JSON.parse(savedSettings));
    }
  }, []);

  // Save settings to localStorage whenever they change
  const updateSetting = (key, value) => {
    const newSettings = { ...settings, [key]: value };
    setSettings(newSettings);
    localStorage.setItem('woodBlockPuzzleSettings', JSON.stringify(newSettings));
  };

  const resetSettings = () => {
    const defaultSettings = {
      soundEnabled: true,
      animationsEnabled: true,
      difficulty: 'normal',
      autoSave: true,
      theme: 'dark',
      aiSpeed: 500,
      showHints: true,
      gridSize: 9
    };
    setSettings(defaultSettings);
    localStorage.setItem('woodBlockPuzzleSettings', JSON.stringify(defaultSettings));
  };

  const clearAllData = () => {
    const confirmed = window.confirm(
      'Are you sure you want to clear all data? This will reset your best scores, AI models, and settings.'
    );
    
    if (confirmed) {
      localStorage.clear();
      alert('All data cleared successfully!');
      resetSettings();
    }
  };

  const exportData = () => {
    const data = {
      settings: settings,
      bestScore: localStorage.getItem('woodBlockPuzzleBestScore'),
      timestamp: new Date().toISOString()
    };
    
    const dataStr = JSON.stringify(data, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    
    const link = document.createElement('a');
    link.href = url;
    link.download = 'wood-block-puzzle-data.json';
    link.click();
    
    URL.revokeObjectURL(url);
  };

  const importData = (event) => {
    const file = event.target.files[0];
    if (!file) return;
    
    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const data = JSON.parse(e.target.result);
        
        if (data.settings) {
          setSettings(data.settings);
          localStorage.setItem('woodBlockPuzzleSettings', JSON.stringify(data.settings));
        }
        
        if (data.bestScore) {
          localStorage.setItem('woodBlockPuzzleBestScore', data.bestScore);
        }
        
        alert('Data imported successfully!');
      } catch (error) {
        alert('Error importing data. Please check the file format.');
      }
    };
    reader.readAsText(file);
  };

  return (
    <div className="settings-view">
      <div className="settings-header">
        <button 
          className="back-btn"
          onClick={() => onNavigate('menu')}
        >
          ← Back to Menu
        </button>
        
        <h1 style={{ color: 'white', fontSize: '36px', margin: '0', textAlign: 'center', flex: 1 }}>
          ⚙️ Game Settings
        </h1>
      </div>

      <div className="settings-container">
        {/* Game Settings */}
        <div className="settings-section">
          <h3>🎮 Game Settings</h3>
          
          <div className="setting-item">
            <label htmlFor="difficulty">Default Difficulty:</label>
            <select
              id="difficulty"
              value={settings.difficulty}
              onChange={(e) => updateSetting('difficulty', e.target.value)}
              className="setting-select"
            >
              <option value="normal">Normal</option>
              <option value="hard">Hard (Blocked Center)</option>
            </select>
            <div className="setting-description">
              Choose the default difficulty level for new games
            </div>
          </div>

          <div className="setting-item">
            <label htmlFor="gridSize">Grid Size:</label>
            <select
              id="gridSize"
              value={settings.gridSize}
              onChange={(e) => updateSetting('gridSize', parseInt(e.target.value))}
              className="setting-select"
            >
              <option value={9}>9x9 (Classic)</option>
              <option value={10}>10x10 (Extended)</option>
            </select>
            <div className="setting-description">
              Game board size (9x9 is recommended for optimal AI performance)
            </div>
          </div>

          <div className="setting-item">
            <div className="setting-toggle">
              <input
                type="checkbox"
                id="showHints"
                checked={settings.showHints}
                onChange={(e) => updateSetting('showHints', e.target.checked)}
              />
              <label htmlFor="showHints">Show Placement Hints</label>
            </div>
            <div className="setting-description">
              Highlight valid placement positions when dragging blocks
            </div>
          </div>

          <div className="setting-item">
            <div className="setting-toggle">
              <input
                type="checkbox"
                id="autoSave"
                checked={settings.autoSave}
                onChange={(e) => updateSetting('autoSave', e.target.checked)}
              />
              <label htmlFor="autoSave">Auto-Save Progress</label>
            </div>
            <div className="setting-description">
              Automatically save game progress and best scores
            </div>
          </div>
        </div>

        {/* Audio & Visual Settings */}
        <div className="settings-section">
          <h3>🎨 Audio & Visual</h3>
          
          <div className="setting-item">
            <label htmlFor="theme">Theme:</label>
            <select
              id="theme"
              value={settings.theme}
              onChange={(e) => updateSetting('theme', e.target.value)}
              className="setting-select"
            >
              <option value="dark">Dark Wood</option>
              <option value="light">Light Wood</option>
              <option value="colorful">Colorful</option>
            </select>
            <div className="setting-description">
              Choose your preferred color theme
            </div>
          </div>

          <div className="setting-item">
            <div className="setting-toggle">
              <input
                type="checkbox"
                id="soundEnabled"
                checked={settings.soundEnabled}
                onChange={(e) => updateSetting('soundEnabled', e.target.checked)}
              />
              <label htmlFor="soundEnabled">Sound Effects</label>
            </div>
            <div className="setting-description">
              Enable sound effects for block placement and line clears
            </div>
          </div>

          <div className="setting-item">
            <div className="setting-toggle">
              <input
                type="checkbox"
                id="animationsEnabled"
                checked={settings.animationsEnabled}
                onChange={(e) => updateSetting('animationsEnabled', e.target.checked)}
              />
              <label htmlFor="animationsEnabled">Animations</label>
            </div>
            <div className="setting-description">
              Enable smooth animations for block placement and line clearing
            </div>
          </div>
        </div>

        {/* AI Settings */}
        <div className="settings-section">
          <h3>🤖 AI Settings</h3>
          
          <div className="setting-item">
            <label htmlFor="aiSpeed">AI Play Speed:</label>
            <div className="setting-range">
              <input
                type="range"
                id="aiSpeed"
                min="100"
                max="2000"
                step="100"
                value={settings.aiSpeed}
                onChange={(e) => updateSetting('aiSpeed', parseInt(e.target.value))}
              />
              <span className="range-value">{settings.aiSpeed}ms</span>
            </div>
            <div className="setting-description">
              Speed of AI moves when watching AI play ({settings.aiSpeed < 500 ? 'Fast' : settings.aiSpeed < 1000 ? 'Medium' : 'Slow'})
            </div>
          </div>
        </div>

        {/* Data Management */}
        <div className="settings-section">
          <h3>💾 Data Management</h3>
          
          <div className="data-info">
            <div className="data-item">
              <span className="data-label">Best Score:</span>
              <span className="data-value">
                {localStorage.getItem('woodBlockPuzzleBestScore') || '0'}
              </span>
            </div>
            <div className="data-item">
              <span className="data-label">Saved AI Models:</span>
              <span className="data-value">
                {Object.keys(localStorage).filter(key => key.includes('dqn-model')).length}
              </span>
            </div>
            <div className="data-item">
              <span className="data-label">Storage Used:</span>
              <span className="data-value">
                {Math.round(JSON.stringify(localStorage).length / 1024)} KB
              </span>
            </div>
          </div>

          <div className="data-buttons">
            <button
              onClick={exportData}
              className="btn export-btn"
            >
              📤 Export Data
            </button>
            
            <label className="btn import-btn">
              📥 Import Data
              <input
                type="file"
                accept=".json"
                onChange={importData}
                style={{ display: 'none' }}
              />
            </label>
            
            <button
              onClick={resetSettings}
              className="btn reset-btn"
            >
              🔄 Reset Settings
            </button>
            
            <button
              onClick={clearAllData}
              className="btn danger-btn"
            >
              🗑️ Clear All Data
            </button>
          </div>
        </div>

        {/* About Section */}
        <div className="settings-section">
          <h3>ℹ️ About</h3>
          
          <div className="about-info">
            <div className="about-item">
              <h4>Wood Block Puzzle with AI</h4>
              <p>
                A classic puzzle game enhanced with advanced artificial intelligence capabilities. 
                Watch neural networks learn to play, analyze their strategies, and compete against 
                sophisticated AI opponents.
              </p>
            </div>
            
            <div className="about-item">
              <h4>AI Technology</h4>
              <ul>
                <li>Deep Q-Learning (DQN) with experience replay</li>
                <li>Curriculum learning and meta-learning</li>
                <li>Prioritized experience replay</li>
                <li>Real-time training visualization</li>
                <li>Strategic reward system</li>
              </ul>
            </div>
            
            <div className="about-item">
              <h4>Game Features</h4>
              <ul>
                <li>Classic 9x9 wood block puzzle gameplay</li>
                <li>Multiple difficulty levels</li>
                <li>Advanced scoring system</li>
                <li>Real-time AI training and visualization</li>
                <li>Performance analytics and statistics</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default SettingsView; 