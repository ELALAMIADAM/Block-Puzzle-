import React, { useState, useEffect, useRef } from 'react';
import { DQNAgent } from '../ai/DQNAgent';
import { DQNEnvironment } from '../ai/DQNEnvironment';
import { runAITests } from '../ai/AITestRunner';
import AIVisualization from './AIVisualization';
import { generateRandomBlocks, checkGameOver } from '../utils/gameLogic';

function AITrainingPanel({ 
  grid, 
  availableBlocks, 
  score, 
  difficulty, 
  onGameStateUpdate,
  gameOver,
  onResetGame 
}) {
  const [agent, setAgent] = useState(null);
  const [environment, setEnvironment] = useState(null);
  const [isTraining, setIsTraining] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [trainingStats, setTrainingStats] = useState({});
  const [trainingEpisodes, setTrainingEpisodes] = useState(1000);
  const [autoPlay, setAutoPlay] = useState(false);
  const [playSpeed, setPlaySpeed] = useState(500);
  const [continuousPlay, setContinuousPlay] = useState(false);
  const [visualTraining, setVisualTraining] = useState(false);
  const [trainingGrid, setTrainingGrid] = useState([]);
  const [trainingBlocks, setTrainingBlocks] = useState([]);
  const [currentEpisode, setCurrentEpisode] = useState(0);
  const [episodeScore, setEpisodeScore] = useState(0);
  const [episodeSteps, setEpisodeSteps] = useState(0);
  
  const trainingIntervalRef = useRef(null);
  const playIntervalRef = useRef(null);
  const episodeRef = useRef(0);
  const stepRef = useRef(0);

  useEffect(() => {
    // Initialize AI components
    const env = new DQNEnvironment();
    const dqnAgent = new DQNAgent(env.getStateSize(), env.getMaxActionSpace(), {
      learningRate: 0.0005, // Improved stability
      epsilon: 0.9, // Better initial exploration
      epsilonMin: 0.05, // Continued exploration
      epsilonDecay: 0.998, // Slower decay
      gamma: 0.97, // Better long-term planning
      batchSize: 64, // Larger batches for stability
      memorySize: 15000, // More experience memory
      targetUpdateFreq: 150 // More stable target updates
    });
    
    setEnvironment(env);
    setAgent(dqnAgent);
    
    return () => {
      if (dqnAgent) dqnAgent.dispose();
    };
  }, []);

  const startTraining = async () => {
    if (!agent || !environment || isTraining) return;
    
    setIsTraining(true);
    episodeRef.current = 0;
    
    trainingIntervalRef.current = setInterval(async () => {
      await runTrainingEpisode();
      episodeRef.current++;
      
      if (episodeRef.current >= trainingEpisodes) {
        stopTraining();
      }
    }, visualTraining ? 100 : 10); // Slower when visual training is enabled
  };

  const stopTraining = () => {
    setIsTraining(false);
    if (trainingIntervalRef.current) {
      clearInterval(trainingIntervalRef.current);
      trainingIntervalRef.current = null;
    }
  };

  const runTrainingEpisode = async () => {
    if (!agent || !environment) return;
    
    // Reset environment for new episode
    environment.reset();
    environment.availableBlocks = generateRandomBlocks();
    environment.difficulty = difficulty;
    
    // Update visual training state
    if (visualTraining) {
      setTrainingGrid(environment.grid.map(row => [...row]));
      setTrainingBlocks(environment.availableBlocks.map(block => block.map(row => [...row])));
      setCurrentEpisode(episodeRef.current);
      setEpisodeScore(0);
      setEpisodeSteps(0);
      stepRef.current = 0;
    }
    
    agent.startEpisode();
    let state = environment.getState();
    let done = false;
    let stepCount = 0;
    const maxSteps = 100; // Prevent infinite episodes
    
    while (!done && stepCount < maxSteps) {
      const validActions = environment.getValidActions();
      
      if (validActions.length === 0) {
        done = true;
        break;
      }
      
      const action = await agent.act(state, validActions);
      const stepResult = environment.step(action);
      
      // Visual feedback for training
      if (visualTraining) {
        setTrainingGrid(environment.grid.map(row => [...row]));
        setEpisodeScore(environment.score);
        setEpisodeSteps(stepCount + 1);
        stepRef.current = stepCount + 1;
        
        // Small delay to show the move visually (reduced for responsiveness)
        if (visualTraining) {
          await new Promise(resolve => setTimeout(resolve, 10));
        }
      }
      
      agent.remember(
        state,
        action,
        stepResult.reward,
        stepResult.state,
        stepResult.done
      );
      
      state = stepResult.state;
      done = stepResult.done;
      stepCount++;
      
      // Train the agent periodically
      if (agent.memory.length > agent.batchSize && stepCount % 4 === 0) {
        await agent.replay();
      }
      
      // Add new blocks when tray is empty and check for game over
      if (environment.availableBlocks.length === 0 && !done) {
        environment.availableBlocks = generateRandomBlocks();
        // Check if game over after generating new blocks
        done = checkGameOver(environment.grid, environment.availableBlocks, environment.difficulty);
        if (visualTraining) {
          setTrainingBlocks(environment.availableBlocks.map(block => block.map(row => [...row])));
        }
      }
    }
    
    agent.endEpisode(0, environment.score);
    
    // Update stats more frequently for better visualization - every 5 episodes
    if (episodeRef.current % 5 === 0) {
      const newStats = agent.getStats();
      setTrainingStats(newStats);
      
      // Log progress to console for real-time feedback
      console.log(`📈 Episode ${episodeRef.current}: Reward: ${newStats.totalReward.toFixed(2)}, ` +
                  `Avg: ${newStats.avgReward.toFixed(2)}, Best Score: ${newStats.bestScore}, ` +
                  `Epsilon: ${(newStats.epsilon * 100).toFixed(1)}%, Rewards Array Length: ${newStats.rewards?.length || 0}`);
    }
  };

  const startAIPlay = async () => {
    if (!agent || !environment) {
      console.log('🤖 Cannot start AI play: Agent or environment not ready');
      alert('Please start training first to initialize the AI agent!');
      return;
    }
    
    if (isPlaying) {
      console.log('🤖 AI is already playing');
      return;
    }
    
    console.log('🎮 Starting AI play...');
    setIsPlaying(true);
    setContinuousPlay(true);
    
    // Set environment to current game state
    try {
      environment.setState(grid, availableBlocks, score, difficulty);
      console.log('🎮 Environment state updated successfully');
      
      playIntervalRef.current = setInterval(async () => {
        await makeAIMove();
      }, playSpeed);
      
      console.log(`🎮 AI play started with ${playSpeed}ms interval`);
    } catch (error) {
      console.error('🎮 Error starting AI play:', error);
      setIsPlaying(false);
      setContinuousPlay(false);
    }
  };

  const stopAIPlay = () => {
    setIsPlaying(false);
    setContinuousPlay(false);
    if (playIntervalRef.current) {
      clearInterval(playIntervalRef.current);
      playIntervalRef.current = null;
    }
  };

  const makeAIMove = async () => {
    if (!agent || !environment) {
      console.log('🤖 AI or environment not ready');
      return;
    }
    
    // If game is over, handle based on auto-play setting
    if (gameOver) {
      if (autoPlay) {
        console.log('🎮 Game over - Auto-restarting in 2 seconds...');
        setTimeout(() => {
          onResetGame();
          // The useEffect will restart AI play after game reset
        }, 2000);
      } else {
        console.log('🎮 Game over - Stopping AI play');
        stopAIPlay();
      }
      return;
    }
    
    try {
      // Always update environment with current game state before making decisions
      environment.setState(grid, availableBlocks, score, difficulty);
      console.log(`🤖 Game state updated - Score: ${score}, Blocks: ${availableBlocks.length}`);
      
      const validActions = environment.getValidActions();
      console.log(`🤖 Found ${validActions.length} valid actions`);
      
      // If no valid actions, wait for new blocks
      if (validActions.length === 0) {
        console.log('🤖 No valid moves available - waiting for new blocks...');
        return;
      }
      
      // Get AI action (exploitation only - no exploration during play)
      const state = environment.getState();
      const action = await agent.predict(state, validActions);
      
      // Execute action in the game
      const { blockIndex, row, col } = environment.decodeAction(action);
      console.log(`🤖 AI selected action: block ${blockIndex} at position (${row}, ${col})`);
      
      // Double-check the action is still valid
      if (blockIndex >= 0 && blockIndex < availableBlocks.length && availableBlocks[blockIndex]) {
        const blockShape = availableBlocks[blockIndex];
        console.log(`🤖 Placing block with shape:`, blockShape);
        
        // Execute the move in the actual game
        const moveSuccess = onGameStateUpdate({
          blockIndex,
          row,
          col,
          blockShape: blockShape
        });
        
        if (moveSuccess !== false) {
          console.log(`🤖 Move executed successfully`);
        } else {
          console.log(`🤖 Move failed to execute - stopping AI play`);
          stopAIPlay();
          return;
        }
      } else {
        console.log(`🤖 Invalid block selection: blockIndex=${blockIndex}, availableBlocks.length=${availableBlocks.length}`);
      }
      
      // Clean up state tensor
      state.dispose();
      
    } catch (error) {
      console.error('🤖 AI move error:', error);
      // Don't stop on errors, just continue
    }
  };

  const saveModel = async () => {
    if (!agent) return;
    const success = await agent.saveModel(`wood-block-dqn-${difficulty}`);
    if (success) {
      alert('Model saved successfully!');
    } else {
      alert('Failed to save model.');
    }
  };

  const loadModel = async () => {
    if (!agent) return;
    const success = await agent.loadModel(`wood-block-dqn-${difficulty}`);
    if (success) {
      alert('Model loaded successfully!');
      setTrainingStats(agent.getStats());
    } else {
      alert('Failed to load model.');
    }
  };

  const resetTraining = () => {
    if (agent) {
      agent.dispose();
    }
    
    const env = new DQNEnvironment();
    const dqnAgent = new DQNAgent(env.getStateSize(), env.getActionSpace());
    
    setEnvironment(env);
    setAgent(dqnAgent);
    setTrainingStats({});
    episodeRef.current = 0;
  };

  const runTests = async () => {
    console.log('🧪 Running AI system tests...');
    try {
      const results = await runAITests();
      const allPassed = Object.values(results).every(result => result);
      
      if (allPassed) {
        alert('✅ All AI tests passed! System is working correctly.');
      } else {
        alert('❌ Some tests failed. Check the browser console for details.');
      }
    } catch (error) {
      console.error('Test error:', error);
      alert('❌ Test execution failed. Check the browser console for details.');
    }
  };

  useEffect(() => {
    // Cleanup intervals on unmount
    return () => {
      if (trainingIntervalRef.current) clearInterval(trainingIntervalRef.current);
      if (playIntervalRef.current) clearInterval(playIntervalRef.current);
    };
  }, []);

  useEffect(() => {
    // Auto-restart AI play if enabled and game resets
    if (autoPlay && !gameOver && !isPlaying && agent && continuousPlay) {
      console.log('🎮 Auto-restarting AI play after game reset...');
      setTimeout(() => startAIPlay(), 1000);
    }
  }, [gameOver, autoPlay, agent]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    // Update play speed dynamically and restart AI play if needed
    if (isPlaying && playIntervalRef.current) {
      clearInterval(playIntervalRef.current);
      playIntervalRef.current = setInterval(async () => {
        await makeAIMove();
      }, playSpeed);
    }
  }, [playSpeed, makeAIMove]); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="ai-training-panel" style={{ minHeight: '800px' }}>
      <h3>🤖 AI Training Center</h3>
      
      {/* Quick Start Guide */}
      <div style={{ 
        background: 'linear-gradient(135deg, rgba(255, 215, 0, 0.1), rgba(255, 165, 0, 0.1))', 
        padding: '20px', 
        borderRadius: '10px', 
        marginBottom: '20px',
        border: '2px solid #FFD700'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '15px' }}>
          <span style={{ fontSize: '24px' }}>🚀</span>
          <strong style={{ fontSize: '18px', color: '#FFD700' }}>Quick Start: Watch AI Learn Visually!</strong>
        </div>
        <div style={{ 
          display: 'grid', 
          gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', 
          gap: '15px',
          marginBottom: '15px'
        }}>
          <div style={{ 
            background: 'rgba(139, 69, 19, 0.3)',
            padding: '12px',
            borderRadius: '8px',
            border: '1px solid #8B4513'
          }}>
            <div style={{ color: '#FFD700', fontWeight: 'bold', marginBottom: '5px' }}>
              1️⃣ Enable Visual Training
            </div>
            <div style={{ color: '#D2B48C', fontSize: '13px' }}>
              ✅ Check "Visual Training Mode" below to watch AI play on the game board
            </div>
          </div>
          <div style={{ 
            background: 'rgba(139, 69, 19, 0.3)',
            padding: '12px',
            borderRadius: '8px',
            border: '1px solid #8B4513'
          }}>
            <div style={{ color: '#FFD700', fontWeight: 'bold', marginBottom: '5px' }}>
              2️⃣ Start Training
            </div>
            <div style={{ color: '#D2B48C', fontSize: '13px' }}>
              🧠 Click "Start Training" to see live episode-by-episode gameplay
            </div>
          </div>
          <div style={{ 
            background: 'rgba(139, 69, 19, 0.3)',
            padding: '12px',
            borderRadius: '8px',
            border: '1px solid #8B4513'
          }}>
            <div style={{ color: '#FFD700', fontWeight: 'bold', marginBottom: '5px' }}>
              3️⃣ Watch Charts Update
            </div>
            <div style={{ color: '#D2B48C', fontSize: '13px' }}>
              📊 See real-time learning progress in D3.js charts below
            </div>
          </div>
        </div>
        
        {/* Game Rules Reminder */}
        <div style={{ 
          background: 'rgba(255, 215, 0, 0.1)', 
          padding: '15px', 
          borderRadius: '8px', 
          marginBottom: '10px',
          border: '1px solid #FFD700'
        }}>
          <div style={{ color: '#FFD700', fontWeight: 'bold', marginBottom: '8px', fontSize: '14px' }}>
            🎮 Game Rules (AI follows exactly):
          </div>
          <div style={{ 
            display: 'grid', 
            gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', 
            gap: '10px',
            fontSize: '12px',
            color: '#D2B48C'
          }}>
            <div>• Place blocks to fill rows, columns, or 3x3 squares</div>
            <div>• Game ends when no blocks can be placed</div>
            <div>• Score increases from block placement + line clears</div>
            <div>• New blocks appear when tray is empty</div>
          </div>
        </div>
      </div>
      
      {/* Status Overview */}
      <div style={{ 
        background: 'rgba(255, 215, 0, 0.1)', 
        padding: '15px', 
        borderRadius: '8px', 
        marginBottom: '20px',
        border: '2px solid #FFD700'
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
          <strong style={{ fontSize: '16px' }}>🎯 AI Status Dashboard</strong>
          <div style={{ display: 'flex', gap: '15px', fontSize: '14px' }}>
            <span>Training: <strong style={{ color: isTraining ? '#28a745' : '#dc3545' }}>
              {isTraining ? 'ACTIVE' : 'STOPPED'}
            </strong></span>
            <span>Playing: <strong style={{ color: isPlaying ? '#007bff' : '#6c757d' }}>
              {isPlaying ? 'ACTIVE' : 'STOPPED'}
            </strong></span>
            <span>Visual: <strong style={{ color: visualTraining ? '#28a745' : '#6c757d' }}>
              {visualTraining ? 'ON' : 'OFF'}
            </strong></span>
          </div>
        </div>
        <div style={{ fontSize: '14px', opacity: 0.9, lineHeight: '1.4' }}>
          <strong>How it works:</strong> Training teaches the AI by playing thousands of practice games. 
          Visual Training Mode lets you watch each episode on the game board in real-time. 
          AI Play uses the trained knowledge to play your current game.
        </div>
        <div style={{ 
          fontSize: '12px', 
          color: '#FFD700', 
          marginTop: '8px',
          padding: '8px',
          background: 'rgba(255, 215, 0, 0.1)',
          borderRadius: '5px',
          border: '1px solid #FFD700'
        }}>
          <strong>🎮 Game Rules:</strong> AI follows exact Wood Block Puzzle rules - place blocks to fill rows/columns/squares, 
          game ends when no blocks can be placed, new blocks appear when tray is empty. 
          <em>Now using 3x3 max blocks for faster training!</em>
        </div>
      </div>
      
      <div className="ai-controls">
        <div className="training-section">
          <h4>🎓 Training (Learning Phase)</h4>
          <div style={{ fontSize: '12px', color: '#D2B48C', marginBottom: '10px' }}>
            The AI learns by playing practice games and improving its strategy
          </div>
          
          <div className="control-group">
            <label>Episodes:</label>
            <input
              type="number"
              value={trainingEpisodes}
              onChange={(e) => setTrainingEpisodes(parseInt(e.target.value))}
              min="10"
              max="10000"
              disabled={isTraining}
            />
            <span style={{ fontSize: '12px', color: '#D2B48C' }}>
              (1 episode = 1 practice game)
            </span>
          </div>
          
          <div className="control-group">
            <label>
              <input
                type="checkbox"
                checked={visualTraining}
                onChange={(e) => setVisualTraining(e.target.checked)}
                disabled={isTraining}
              />
              Visual Training Mode
            </label>
            <span style={{ fontSize: '12px', color: '#D2B48C' }}>
              (Watch AI play on game board during training)
            </span>
          </div>
          
          <div className="button-group">
            <button
              onClick={startTraining}
              disabled={isTraining || !agent}
              className="btn train-btn"
            >
              {isTraining ? `Training... (${episodeRef.current}/${trainingEpisodes})` : 'Start Training'}
            </button>
            
            <button
              onClick={stopTraining}
              disabled={!isTraining}
              className="btn stop-btn"
            >
              Stop Training
            </button>
            
            <button
              onClick={resetTraining}
              disabled={isTraining}
              className="btn reset-btn"
            >
              Reset AI Brain
            </button>
          </div>
        </div>
        
        <div className="play-section">
          <h4>🎮 AI Play (Action Phase)</h4>
          <div style={{ fontSize: '12px', color: '#D2B48C', marginBottom: '10px' }}>
            The trained AI plays your current game using learned strategies
          </div>
          
          <div className="control-group">
            <label>Speed:</label>
            <input
              type="range"
              value={playSpeed}
              onChange={(e) => setPlaySpeed(parseInt(e.target.value))}
              min="100"
              max="2000"
              step="100"
              disabled={isPlaying}
            />
            <span>{playSpeed}ms</span>
            <span style={{ fontSize: '12px', color: '#D2B48C' }}>
              ({playSpeed < 500 ? 'Fast' : playSpeed < 1000 ? 'Medium' : 'Slow'})
            </span>
          </div>
          
          <div className="control-group">
            <label>
              <input
                type="checkbox"
                checked={autoPlay}
                onChange={(e) => setAutoPlay(e.target.checked)}
              />
              Auto-play new games
            </label>
            <span style={{ fontSize: '12px', color: '#D2B48C' }}>
              (Restart automatically when game ends)
            </span>
          </div>
          
          <div className="control-group">
            <label>
              <input
                type="checkbox"
                checked={continuousPlay}
                onChange={(e) => setContinuousPlay(e.target.checked)}
                disabled={!isPlaying}
              />
              Continuous play mode
            </label>
            <span style={{ fontSize: '12px', color: '#D2B48C' }}>
              (Keep playing even when no moves available)
            </span>
          </div>
          
          <div className="button-group">
            <button
              onClick={startAIPlay}
              disabled={isPlaying || !agent}
              className="btn play-btn"
            >
              {isPlaying ? 'AI Playing...' : 'Start AI Play'}
            </button>
            
            <button
              onClick={stopAIPlay}
              disabled={!isPlaying}
              className="btn stop-btn"
            >
              Stop AI
            </button>
          </div>
        </div>
        
        <div className="model-section">
          <h4>💾 Model Management</h4>
          <div style={{ fontSize: '12px', color: '#D2B48C', marginBottom: '10px' }}>
            Save and load trained AI models for different difficulty levels
          </div>
          
          <div className="button-group">
            <button
              onClick={saveModel}
              disabled={!agent}
              className="btn save-btn"
            >
              💾 Save Model
            </button>
            
            <button
              onClick={loadModel}
              disabled={!agent}
              className="btn load-btn"
            >
              📁 Load Model
            </button>
            
            <button
              onClick={runTests}
              className="btn test-btn"
            >
              🧪 Run Tests
            </button>
          </div>
          
          <div style={{ fontSize: '12px', color: '#D2B48C', marginTop: '8px' }}>
            Models are saved per difficulty: wood-block-dqn-{difficulty}
          </div>
        </div>
      </div>
      
      {/* Visual Training Board */}
      {visualTraining && isTraining && (
        <div style={{
          background: 'rgba(139, 69, 19, 0.3)',
          padding: '20px',
          borderRadius: '10px',
          border: '1px solid #8B4513',
          marginBottom: '20px'
        }}>
          <h4 style={{ color: '#FFD700', marginBottom: '15px' }}>
            🎮 Live Training Episode #{currentEpisode}
          </h4>
          
          <div style={{ display: 'flex', gap: '20px', alignItems: 'flex-start' }}>
            {/* Training Game Board */}
            <div style={{ flex: 1 }}>
              <div style={{ marginBottom: '10px', color: '#D2B48C', fontSize: '14px' }}>
                Training Game Board
              </div>
              <div style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(9, 1fr)',
                gap: '2px',
                background: '#654321',
                padding: '10px',
                borderRadius: '10px',
                width: 'fit-content'
              }}>
                {trainingGrid.length > 0 && trainingGrid.map((row, rowIndex) =>
                  row.map((cell, colIndex) => (
                    <div
                      key={`${rowIndex}-${colIndex}`}
                      style={{
                        width: '25px',
                        height: '25px',
                        background: cell 
                          ? 'linear-gradient(145deg, #4b2703, #8B4513)' 
                          : 'linear-gradient(145deg, #D2B48C, #A0522D)',
                        border: '1px solid #8B4513',
                        borderRadius: '3px',
                        boxShadow: cell ? 'inset 0 2px 4px rgba(0, 0, 0, 0.3)' : 'none'
                      }}
                    />
                  ))
                )}
              </div>
            </div>
            
            {/* Training Blocks */}
            <div style={{ flex: '0 0 150px' }}>
              <div style={{ marginBottom: '10px', color: '#D2B48C', fontSize: '14px' }}>
                Available Blocks
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                {trainingBlocks.map((block, blockIndex) => (
                  <div key={blockIndex} style={{
                    padding: '8px',
                    background: 'rgba(139, 69, 19, 0.3)',
                    borderRadius: '5px',
                    border: '1px solid #8B4513'
                  }}>
                    <div style={{
                      display: 'grid',
                      gridTemplateColumns: `repeat(${Math.max(...block.map(row => row.length))}, 1fr)`,
                      gap: '1px'
                    }}>
                      {block.map((row, rowIndex) =>
                        row.map((cell, colIndex) => (
                          <div
                            key={`${rowIndex}-${colIndex}`}
                            style={{
                              width: '15px',
                              height: '15px',
                              background: cell 
                                ? 'linear-gradient(145deg, #CD853F, #8B4513)' 
                                : 'transparent',
                              border: cell ? '1px solid #654321' : 'none',
                              borderRadius: '2px'
                            }}
                          />
                        ))
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
            
            {/* Episode Stats */}
            <div style={{ flex: '0 0 200px' }}>
              <div style={{ marginBottom: '10px', color: '#D2B48C', fontSize: '14px' }}>
                Episode Progress
              </div>
              <div style={{ 
                background: 'rgba(139, 69, 19, 0.3)',
                padding: '15px',
                borderRadius: '8px',
                border: '1px solid #8B4513'
              }}>
                <div style={{ marginBottom: '8px' }}>
                  <span style={{ color: '#FFD700', fontWeight: 'bold' }}>Episode:</span>
                  <span style={{ color: 'white', marginLeft: '8px' }}>{currentEpisode}</span>
                </div>
                <div style={{ marginBottom: '8px' }}>
                  <span style={{ color: '#FFD700', fontWeight: 'bold' }}>Score:</span>
                  <span style={{ color: 'white', marginLeft: '8px' }}>{episodeScore}</span>
                </div>
                <div style={{ marginBottom: '8px' }}>
                  <span style={{ color: '#FFD700', fontWeight: 'bold' }}>Steps:</span>
                  <span style={{ color: 'white', marginLeft: '8px' }}>{episodeSteps}</span>
                </div>
                <div style={{ marginBottom: '8px' }}>
                  <span style={{ color: '#FFD700', fontWeight: 'bold' }}>Blocks:</span>
                  <span style={{ color: 'white', marginLeft: '8px' }}>{trainingBlocks.length}</span>
                </div>
                
                <div style={{ 
                  marginTop: '15px',
                  padding: '8px',
                  background: 'rgba(40, 167, 69, 0.2)',
                  borderRadius: '5px',
                  border: '1px solid #28a745',
                  textAlign: 'center'
                }}>
                  <div style={{ color: '#28a745', fontSize: '12px', fontWeight: 'bold' }}>
                    🧠 AI Learning...
                  </div>
                  <div style={{ color: '#D2B48C', fontSize: '10px', marginTop: '2px' }}>
                    Episode in progress
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
      
      {trainingStats.episode && (
        <div className="training-stats">
          <h4>📊 Training Statistics</h4>
          <div className="stats-grid">
            <div className="stat-item">
              <span className="stat-label">Episode:</span>
              <span className="stat-value">{trainingStats.episode}</span>
            </div>
            <div className="stat-item">
              <span className="stat-label">Avg Reward (100):</span>
              <span className="stat-value">{trainingStats.avgReward?.toFixed(2) || '0'}</span>
            </div>
            <div className="stat-item">
              <span className="stat-label">Avg Reward (10):</span>
              <span className="stat-value">{trainingStats.avgReward10?.toFixed(2) || '0'}</span>
            </div>
            <div className="stat-item">
              <span className="stat-label">Best Score:</span>
              <span className="stat-value" style={{ color: '#28a745', fontWeight: 'bold' }}>
                {trainingStats.bestScore || '0'}
              </span>
            </div>
            <div className="stat-item">
              <span className="stat-label">Epsilon:</span>
              <span className="stat-value">{(trainingStats.epsilon * 100)?.toFixed(1) || '0'}%</span>
            </div>
            <div className="stat-item">
              <span className="stat-label">Memory:</span>
              <span className="stat-value">{trainingStats.memorySize}</span>
            </div>
            <div className="stat-item">
              <span className="stat-label">Training Steps:</span>
              <span className="stat-value">{trainingStats.trainingSteps}</span>
            </div>
            <div className="stat-item">
              <span className="stat-label">Avg Loss:</span>
              <span className="stat-value">{trainingStats.avgLoss?.toFixed(4) || '0'}</span>
            </div>
          </div>
          
          {isTraining && (
            <div className="training-progress">
              <div className="progress-bar">
                <div 
                  className="progress-fill"
                  style={{ width: `${(episodeRef.current / trainingEpisodes) * 100}%` }}
                />
              </div>
              <span>{episodeRef.current} / {trainingEpisodes} episodes</span>
            </div>
          )}
        </div>
      )}

      {/* Always show AI visualization */}
      <AIVisualization 
        trainingStats={trainingStats} 
        isTraining={isTraining}
      />
    </div>
  );
}

export default AITrainingPanel; 