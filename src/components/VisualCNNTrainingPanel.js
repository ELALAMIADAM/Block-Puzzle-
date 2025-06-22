import React, { useState, useRef, useEffect } from 'react';
import { ConvDQNAgent } from '../ai/ConvDQNAgent';
import { ConvDQNEnvironment } from '../ai/ConvDQNEnvironment';
import AIVisualization from './AIVisualization';

/**
 * VISUAL CNN TRAINING PANEL - 12x12 Pattern Recognition Mode
 * 
 * Features:
 * - CNN-based reinforcement learning
 * - 12x12 grid with 4-channel visual input
 * - Advanced spatial pattern recognition
 * - Visual intelligence metrics
 * - Symmetry and harmony tracking
 */
function VisualCNNTrainingPanel({ 
  grid, 
  availableBlocks, 
  score, 
  difficulty, 
  onGameStateUpdate,
  gameOver,
  onResetGame 
}) {
  const [isTraining, setIsTraining] = useState(false);
  const [isAIPlaying, setIsAIPlaying] = useState(false);
  const [trainingStats, setTrainingStats] = useState(null);
  const [episodeCount, setEpisodeCount] = useState(0);
  const [trainingSpeed, setTrainingSpeed] = useState(1);
  const [modelStatus, setModelStatus] = useState('Not trained');
  const [visualMetrics, setVisualMetrics] = useState({
    spatialEfficiency: 0,
    visualHarmony: 0,
    symmetryScore: 0,
    patternComplexity: 0
  });

  const agentRef = useRef(null);
  const environmentRef = useRef(null);
  const trainingIntervalRef = useRef(null);
  const aiPlayIntervalRef = useRef(null);

  useEffect(() => {
    // Initialize Visual CNN system
    const visualEnv = new ConvDQNEnvironment();
    environmentRef.current = visualEnv;
    
    const visualStateSize = visualEnv.getVisualStateSize(); // [4, 12, 12]
    const actionSize = visualEnv.getMaxActionSpace();
    
    const agent = new ConvDQNAgent(visualStateSize, actionSize, {
      learningRate: 0.0005,
      epsilon: 0.9,
      epsilonDecay: 0.9975,
      batchSize: 32,
      memorySize: 8000
    });
    
    agentRef.current = agent;
    setModelStatus('CNN initialized - Ready for visual training');
    
    console.log('🎨 VISUAL CNN TRAINING PANEL INITIALIZED');
    console.log(`📐 Grid: 12x12 (144 cells)`);
    console.log(`🖼️  Channels: 4 (grid, blocks, potentials, strategy)`);
    console.log(`🧠 Architecture: Convolutional Neural Network`);

    return () => {
      if (trainingIntervalRef.current) {
        clearInterval(trainingIntervalRef.current);
      }
      if (aiPlayIntervalRef.current) {
        clearInterval(aiPlayIntervalRef.current);
      }
      if (agentRef.current) {
        agentRef.current.dispose();
      }
    };
  }, []);

  const startTraining = async () => {
    if (!environmentRef.current || !agentRef.current) return;
    
    setIsTraining(true);
    setModelStatus('Visual CNN training in progress...');
    
    const runTraining = async () => {
      for (let speedMultiplier = 0; speedMultiplier < trainingSpeed; speedMultiplier++) {
        await runVisualTrainingEpisode();
      }
      
      // Update stats
      const stats = agentRef.current.getStats();
      setTrainingStats(stats);
      setEpisodeCount(stats.episode);
      
      // Update visual metrics
      const env = environmentRef.current;
      if (env.visualMetrics) {
        setVisualMetrics({
          spatialEfficiency: env.visualMetrics.spatialEfficiencyScore || 0,
          visualHarmony: env.visualMetrics.visualHarmonyLevel || 0,
          symmetryScore: env.visualMetrics.symmetryAchievements || 0,
          patternComplexity: env.visualMetrics.complexityHandled || 0
        });
      }
    };
    
    trainingIntervalRef.current = setInterval(runTraining, 100);
  };

  const stopTraining = () => {
    setIsTraining(false);
    setModelStatus('Visual CNN training stopped');
    
    if (trainingIntervalRef.current) {
      clearInterval(trainingIntervalRef.current);
      trainingIntervalRef.current = null;
    }
  };

  const runVisualTrainingEpisode = async () => {
    const environment = environmentRef.current;
    const agent = agentRef.current;
    
    if (!environment || !agent) return;
    
    try {
      // Reset for new episode
      let state = environment.reset();
      agent.startEpisode();
      let totalReward = 0;
      let stepCount = 0;
      const maxSteps = 200; // Increased for larger 12x12 grid
      
      while (!environment.gameOver && stepCount < maxSteps) {
        // Get valid actions
        const validActions = environment.getValidActions();
        
        if (validActions.length === 0) {
          environment.gameOver = true;
          break;
        }
        
        // Agent selects action using visual CNN
        const action = await agent.act(state, validActions, environment);
        
        // Execute action in environment
        const stepResult = environment.step(action);
        
        // Store experience in visual pattern memory
        agent.remember(state, action, stepResult.reward, stepResult.state, stepResult.done);
        
        // Update state
        state.dispose(); // Clean up old tensor
        state = stepResult.state;
        totalReward += stepResult.reward;
        stepCount++;
        
        // Train CNN if enough experience
        if (agent.memory.length >= agent.batchSize) {
          await agent.replay();
        }
        
        if (stepResult.done) {
          environment.gameOver = true;
        }
      }
      
      // End episode
      agent.endEpisode(0, environment.score, environment);
      
      // Clean up final state
      state.dispose();
      
    } catch (error) {
      console.error('🚨 Visual CNN training error:', error);
      stopTraining();
    }
  };

  const startAIPlay = async () => {
    if (!environmentRef.current || !agentRef.current) return;
    
    setIsAIPlaying(true);
    setModelStatus('Visual CNN AI playing...');
    
    // Set current game state in visual environment
    const visualEnv = environmentRef.current;
    visualEnv.setState(grid, availableBlocks, score, 'visual');
    
    const playMove = async () => {
      await makeVisualAIMove();
    };
    
    aiPlayIntervalRef.current = setInterval(playMove, 1000);
  };

  const stopAIPlay = () => {
    setIsAIPlaying(false);
    setModelStatus('Visual CNN AI stopped');
    
    if (aiPlayIntervalRef.current) {
      clearInterval(aiPlayIntervalRef.current);
      aiPlayIntervalRef.current = null;
    }
  };

  const makeVisualAIMove = async () => {
    const environment = environmentRef.current;
    const agent = agentRef.current;
    
    if (!environment || !agent || gameOver) {
      stopAIPlay();
      return;
    }
    
    try {
      // Update environment with current game state
      environment.setState(grid, availableBlocks, score, 'visual');
      
      const validActions = environment.getValidActions();
      if (validActions.length === 0) {
        stopAIPlay();
        return;
      }
      
      // Get visual state
      const state = environment.getVisualState();
      
      // CNN makes decision
      const action = await agent.predictWithCNN(state, validActions);
      
      // Decode action for 12x12 grid
      const { blockIndex, row, col } = environment.decodeAction(action);
      
      // Check if action is valid for current game state
      if (blockIndex < availableBlocks.length) {
        const block = availableBlocks[blockIndex];
        
        // Adapt placement for current grid size (9x9 to 12x12 mapping)
        const adaptedRow = Math.min(row, grid.length - block.length);
        const adaptedCol = Math.min(col, grid[0].length - block[0].length);
        
        // Try to place the block in the actual game
        onGameStateUpdate({
          action: 'placeBlock',
          blockShape: block,
          row: adaptedRow,
          col: adaptedCol,
          blockIndex: blockIndex
        });
        
        console.log(`🎨 Visual CNN AI Move: Block ${blockIndex} at (${adaptedRow}, ${adaptedCol})`);
      }
      
      // Clean up state tensor
      state.dispose();
      
    } catch (error) {
      console.error('🚨 Visual CNN AI play error:', error);
      stopAIPlay();
    }
  };

  const saveModel = async () => {
    if (!agentRef.current) return;
    
    try {
      const success = await agentRef.current.saveModel('visual-cnn-dqn-model');
      if (success) {
        setModelStatus('Visual CNN model saved successfully!');
        setTimeout(() => setModelStatus('Model ready'), 2000);
      } else {
        setModelStatus('Failed to save model');
      }
    } catch (error) {
      console.error('Error saving visual CNN model:', error);
      setModelStatus('Error saving model');
    }
  };

  const loadModel = async () => {
    if (!agentRef.current) return;
    
    try {
      const success = await agentRef.current.loadModel('visual-cnn-dqn-model');
      if (success) {
        const stats = agentRef.current.getStats();
        setTrainingStats(stats);
        setEpisodeCount(stats.episode);
        setModelStatus('Visual CNN model loaded successfully!');
        setTimeout(() => setModelStatus('Model ready'), 2000);
      } else {
        setModelStatus('Failed to load model');
      }
    } catch (error) {
      console.error('Error loading visual CNN model:', error);
      setModelStatus('Error loading model');
    }
  };

  const resetTraining = () => {
    if (agentRef.current) {
      agentRef.current.dispose();
    }
    
    // Reinitialize
    const visualEnv = new ConvDQNEnvironment();
    environmentRef.current = visualEnv;
    
    const visualStateSize = visualEnv.getVisualStateSize();
    const actionSize = visualEnv.getMaxActionSpace();
    
    const agent = new ConvDQNAgent(visualStateSize, actionSize, {
      learningRate: 0.0005,
      epsilon: 0.9,
      epsilonDecay: 0.9975,
      batchSize: 32,
      memorySize: 8000
    });
    
    agentRef.current = agent;
    setTrainingStats(null);
    setEpisodeCount(0);
    setModelStatus('Visual CNN reset - Ready for new training');
    setVisualMetrics({
      spatialEfficiency: 0,
      visualHarmony: 0,
      symmetryScore: 0,
      patternComplexity: 0
    });
  };

  return (
    <div className="visual-cnn-training-panel">
      <div className="training-header">
        <h3>🎨 Visual CNN Training - 12x12 Pattern Recognition</h3>
        <div className="model-info">
          <span className="model-status">{modelStatus}</span>
          <span className="episode-count">Episodes: {episodeCount}</span>
        </div>
      </div>

      {/* Visual Metrics Display */}
      <div className="visual-metrics">
        <h4>🧠 Visual Intelligence Metrics</h4>
        <div className="metrics-grid">
          <div className="metric">
            <span className="metric-label">Spatial Efficiency:</span>
            <span className="metric-value">{(visualMetrics.spatialEfficiency * 100).toFixed(1)}%</span>
          </div>
          <div className="metric">
            <span className="metric-label">Visual Harmony:</span>
            <span className="metric-value">{(visualMetrics.visualHarmony * 100).toFixed(1)}%</span>
          </div>
          <div className="metric">
            <span className="metric-label">Symmetry Score:</span>
            <span className="metric-value">{visualMetrics.symmetryScore}</span>
          </div>
          <div className="metric">
            <span className="metric-label">Pattern Complexity:</span>
            <span className="metric-value">{(visualMetrics.patternComplexity * 100).toFixed(1)}%</span>
          </div>
        </div>
      </div>

      {/* Training Controls */}
      <div className="training-controls">
        <div className="control-group">
          <h4>🏋️ Training Controls</h4>
          <div className="button-row">
            {!isTraining ? (
              <button onClick={startTraining} className="start-btn">
                🎨 Start Visual CNN Training
              </button>
            ) : (
              <button onClick={stopTraining} className="stop-btn">
                ⏸️ Stop Training
              </button>
            )}
            <button onClick={resetTraining} className="reset-btn">
              🔄 Reset CNN
            </button>
          </div>
          
          <div className="speed-control">
            <label>Training Speed: </label>
            <select 
              value={trainingSpeed} 
              onChange={(e) => setTrainingSpeed(Number(e.target.value))}
              disabled={isTraining}
            >
              <option value={1}>1x (Stable)</option>
              <option value={2}>2x (Fast)</option>
              <option value={3}>3x (Rapid)</option>
              <option value={5}>5x (Extreme)</option>
            </select>
          </div>
        </div>

        <div className="control-group">
          <h4>🎮 AI Play Controls</h4>
          <div className="button-row">
            {!isAIPlaying ? (
              <button onClick={startAIPlay} className="play-btn">
                🎯 Start Visual AI Play
              </button>
            ) : (
              <button onClick={stopAIPlay} className="stop-btn">
                ⏹️ Stop AI Play
              </button>
            )}
          </div>
          <div className="ai-info">
            <small>🖼️ Uses 12x12 visual patterns adapted to current 9x9 game</small>
          </div>
        </div>

        <div className="control-group">
          <h4>💾 Model Management</h4>
          <div className="button-row">
            <button onClick={saveModel} className="save-btn">
              💾 Save CNN Model
            </button>
            <button onClick={loadModel} className="load-btn">
              📂 Load CNN Model
            </button>
          </div>
        </div>
      </div>

      {/* CNN Architecture Info */}
      <div className="cnn-info">
        <h4>🏗️ CNN Architecture</h4>
        <div className="architecture-details">
          <div className="arch-item">
            <strong>Input:</strong> 4-channel 12×12 visual state
          </div>
          <div className="arch-item">
            <strong>Channels:</strong> Grid, Blocks, Potentials, Strategy
          </div>
          <div className="arch-item">
            <strong>Conv Layers:</strong> 32→64→128→256→512 filters
          </div>
          <div className="arch-item">
            <strong>Dense Layers:</strong> 1024→512→256→actions
          </div>
          <div className="arch-item">
            <strong>Features:</strong> Batch norm, dropout, global avg pooling
          </div>
        </div>
      </div>

      {/* Training Visualization */}
      {trainingStats && (
        <div className="training-visualization">
          <AIVisualization 
            trainingStats={trainingStats} 
            isTraining={isTraining}
            compact={false}
          />
        </div>
      )}

      {/* Performance Summary */}
      {trainingStats && (
        <div className="performance-summary">
          <h4>📊 Visual CNN Performance</h4>
          <div className="stats-grid">
            <div className="stat">
              <span className="stat-label">Best Score:</span>
              <span className="stat-value">{trainingStats.bestScore}</span>
            </div>
            <div className="stat">
              <span className="stat-label">Avg Score:</span>
              <span className="stat-value">{trainingStats.avgScore?.toFixed(1) || 0}</span>
            </div>
            <div className="stat">
              <span className="stat-label">Pattern Recognition:</span>
              <span className="stat-value">{trainingStats.visualMetrics?.patternRecognitions || 0}</span>
            </div>
            <div className="stat">
              <span className="stat-label">Spatial Success:</span>
              <span className="stat-value">{(trainingStats.spatialSuccessRate * 100).toFixed(1)}%</span>
            </div>
            <div className="stat">
              <span className="stat-label">Training Steps:</span>
              <span className="stat-value">{trainingStats.trainingSteps}</span>
            </div>
            <div className="stat">
              <span className="stat-label">Memory Size:</span>
              <span className="stat-value">{trainingStats.memorySize}</span>
            </div>
          </div>
        </div>
      )}

      <style jsx>{`
        .visual-cnn-training-panel {
          background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
          border-radius: 12px;
          padding: 20px;
          margin: 10px;
          box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        }

        .training-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 20px;
          padding-bottom: 10px;
          border-bottom: 2px solid #e0e0e0;
        }

        .training-header h3 {
          margin: 0;
          color: #2c3e50;
          font-size: 1.4em;
        }

        .model-info {
          display: flex;
          flex-direction: column;
          align-items: flex-end;
          font-size: 0.9em;
        }

        .model-status {
          color: #7f8c8d;
          margin-bottom: 4px;
        }

        .episode-count {
          color: #3498db;
          font-weight: bold;
        }

        .visual-metrics {
          background: white;
          border-radius: 8px;
          padding: 15px;
          margin-bottom: 20px;
          box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        }

        .visual-metrics h4 {
          margin: 0 0 15px 0;
          color: #2c3e50;
        }

        .metrics-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
          gap: 10px;
        }

        .metric {
          display: flex;
          justify-content: space-between;
          padding: 8px 12px;
          background: #f8f9fa;
          border-radius: 6px;
          border-left: 4px solid #3498db;
        }

        .metric-label {
          color: #7f8c8d;
        }

        .metric-value {
          font-weight: bold;
          color: #2c3e50;
        }

        .training-controls {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
          gap: 20px;
          margin-bottom: 20px;
        }

        .control-group {
          background: white;
          border-radius: 8px;
          padding: 15px;
          box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        }

        .control-group h4 {
          margin: 0 0 15px 0;
          color: #2c3e50;
        }

        .button-row {
          display: flex;
          gap: 10px;
          margin-bottom: 10px;
          flex-wrap: wrap;
        }

        button {
          padding: 10px 16px;
          border: none;
          border-radius: 6px;
          font-size: 0.9em;
          cursor: pointer;
          transition: all 0.3s ease;
          font-weight: 500;
        }

        .start-btn {
          background: linear-gradient(45deg, #27ae60, #2ecc71);
          color: white;
        }

        .start-btn:hover {
          transform: translateY(-2px);
          box-shadow: 0 4px 12px rgba(39, 174, 96, 0.3);
        }

        .stop-btn {
          background: linear-gradient(45deg, #e74c3c, #c0392b);
          color: white;
        }

        .stop-btn:hover {
          transform: translateY(-2px);
          box-shadow: 0 4px 12px rgba(231, 76, 60, 0.3);
        }

        .reset-btn {
          background: linear-gradient(45deg, #f39c12, #e67e22);
          color: white;
        }

        .reset-btn:hover {
          transform: translateY(-2px);
          box-shadow: 0 4px 12px rgba(243, 156, 18, 0.3);
        }

        .play-btn {
          background: linear-gradient(45deg, #3498db, #2980b9);
          color: white;
        }

        .play-btn:hover {
          transform: translateY(-2px);
          box-shadow: 0 4px 12px rgba(52, 152, 219, 0.3);
        }

        .save-btn, .load-btn {
          background: linear-gradient(45deg, #9b59b6, #8e44ad);
          color: white;
        }

        .save-btn:hover, .load-btn:hover {
          transform: translateY(-2px);
          box-shadow: 0 4px 12px rgba(155, 89, 182, 0.3);
        }

        .speed-control {
          display: flex;
          align-items: center;
          gap: 10px;
          margin-top: 10px;
        }

        .speed-control select {
          padding: 6px 10px;
          border: 1px solid #ddd;
          border-radius: 4px;
          background: white;
        }

        .ai-info {
          margin-top: 10px;
          color: #7f8c8d;
        }

        .cnn-info {
          background: white;
          border-radius: 8px;
          padding: 15px;
          margin-bottom: 20px;
          box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        }

        .cnn-info h4 {
          margin: 0 0 15px 0;
          color: #2c3e50;
        }

        .architecture-details {
          display: grid;
          gap: 8px;
        }

        .arch-item {
          display: flex;
          align-items: center;
          padding: 8px 12px;
          background: #f8f9fa;
          border-radius: 6px;
          border-left: 4px solid #e74c3c;
        }

        .arch-item strong {
          min-width: 120px;
          color: #2c3e50;
        }

        .performance-summary {
          background: white;
          border-radius: 8px;
          padding: 15px;
          margin-top: 20px;
          box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        }

        .performance-summary h4 {
          margin: 0 0 15px 0;
          color: #2c3e50;
        }

        .stats-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
          gap: 10px;
        }

        .stat {
          display: flex;
          justify-content: space-between;
          padding: 8px 12px;
          background: #f8f9fa;
          border-radius: 6px;
          border-left: 4px solid #9b59b6;
        }

        .stat-label {
          color: #7f8c8d;
        }

        .stat-value {
          font-weight: bold;
          color: #2c3e50;
        }

        .training-visualization {
          margin-top: 20px;
        }
      `}</style>
    </div>
  );
}

export default VisualCNNTrainingPanel; 