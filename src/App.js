import React, { useState } from 'react';
import MainMenu from './components/MainMenu';
import GameView from './components/GameView';
import AILearningView from './components/AILearningView';
import SettingsView from './components/SettingsView';

function App() {
  const [currentView, setCurrentView] = useState('menu');

  const handleNavigation = (view) => {
    setCurrentView(view);
  };

  const renderCurrentView = () => {
    switch (currentView) {
      case 'game':
        return <GameView onNavigate={handleNavigation} />;
      case 'ai-learning':
        return <AILearningView onNavigate={handleNavigation} />;
      case 'settings':
        return <SettingsView onNavigate={handleNavigation} />;
      case 'menu':
      default:
        return <MainMenu onNavigate={handleNavigation} />;
    }
  };

  return (
    <div className="app">
      {renderCurrentView()}
    </div>
  );
}

export default App; 