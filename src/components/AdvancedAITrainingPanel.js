import React, { useState, useEffect, useRef, useCallback } from 'react';
import { EliteDQNAgent } from '../ai/EliteDQNAgent';
import { AlgorithmSelector } from '../ai/AdvancedAIAgents';
import { EliteEnvironment } from '../ai/EliteEnvironment';
import AIVisualization from './AIVisualization';

function AdvancedAITrainingPanel({ 
  grid, 
  availableBlocks, 
  score, 
  difficulty, 
  onGameStateUpdate,
  gameOver,
  onResetGame 
}) {
  const [selectedAlgorithm, setSelectedAlgorithm] = useState('heuristic'); // Start with fastest
  const [agent, setAgent] = useState(null);
  const [environment, setEnvironment] = useState(null);
  const [isTraining, setIsTraining] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [trainingStats, setTrainingStats] = useState({});
  const [trainingEpisodes, setTrainingEpisodes] = useState(500); // Increased for better training
  const [autoPlay, setAutoPlay] = useState(false);
  const [playSpeed, setPlaySpeed] = useState(500);
  const [algorithmComparison, setAlgorithmComparison] = useState({});
  
  const trainingIntervalRef = useRef(null);
  const playIntervalRef = useRef(null);
  const episodeRef = useRef(0);

  // Algorithm configurations
  const algorithmConfigs = {
    'dqn': {
      displayName: 'Elite DQN',
      description: 'Deep Q-Network with advanced features',
      options: {
        learningRate: 0.0003,
        epsilon: 1.0,
        epsilonMin: 0.01,
        epsilonDecay: 0.9995,
        gamma: 0.99,
        batchSize: 64,
        memorySize: 20000
      }
    },
    'mcts': {
      displayName: 'Monte Carlo Tree Search',
      description: 'Tree search with intelligent rollouts',
      options: {
        maxSimulations: 500, // Reduced for faster play
        explorationConstant: 1.414,
        maxDepth: 30,
        rolloutPolicy: 'heuristic'
      }
    },
    'policy-gradient': {
      displayName: 'Policy Gradient',
      description: 'Direct policy optimization with action masking',
      options: {
        learningRate: 0.001,
        gamma: 0.99,
        entropy_coeff: 0.01
      }
    },
    'heuristic': {
      displayName: 'Hybrid Heuristic',
      description: 'Smart rules with lookahead',
      options: {
        lookaheadDepth: 2,
        lineCompletionWeight: 1000,
        futureOpportunityWeight: 100
      }
    }
  };

  // Fix the useEffect dependency issue by using useCallback for initializeAgent
  const initializeAgent = useCallback(() => {
    if (selectedAlgorithm && !agent) {
      console.log(`🤖 Initializing ${selectedAlgorithm} agent...`);
      
      const env = new EliteEnvironment();
      const stateSize = env.getStateSize();
      const actionSize = env.getMaxActionSpace();
      
      let newAgent;
      const config = algorithmConfigs[selectedAlgorithm];
      
      if (selectedAlgorithm === 'dqn') {
        newAgent = new EliteDQNAgent(stateSize, actionSize, config.options);
      } else {
        newAgent = AlgorithmSelector.createAgent(selectedAlgorithm, stateSize, actionSize, config.options);
      }
      
      setEnvironment(env);
      setAgent(newAgent);
      setTrainingStats({});
      episodeRef.current = 0;
      
      console.log(`✅ ${config.displayName} initialized successfully!`);
    }
  }, [selectedAlgorithm, agent]);

  useEffect(() => {
    initializeAgent();
  }, [initializeAgent]);

  const startTraining = async () => {
    if (!agent || !environment || isTraining) return;
    
    console.log(`🚀 Starting ${algorithmConfigs[selectedAlgorithm].displayName} training...`);
    console.log(`📈 Target episodes: ${trainingEpisodes}`);
    
    setIsTraining(true);
    episodeRef.current = 0;
    
    // Adjust training speed based on algorithm
    const trainingSpeed = selectedAlgorithm === 'heuristic' ? 50 : 
                         selectedAlgorithm === 'mcts' ? 200 : 100;
    
    trainingIntervalRef.current = setInterval(async () => {
      await runTrainingEpisode();
      episodeRef.current++;
      
      if (episodeRef.current >= trainingEpisodes) {
        stopTraining();
      }
    }, trainingSpeed);
  };

  const stopTraining = () => {
    setIsTraining(false);
    if (trainingIntervalRef.current) {
      clearInterval(trainingIntervalRef.current);
      trainingIntervalRef.current = null;
    }
    
    console.log(`🏁 ${algorithmConfigs[selectedAlgorithm].displayName} training completed`);
    
    // Update comparison data
    if (agent) {
      const stats = agent.getStats();
      setAlgorithmComparison(prev => ({
        ...prev,
        [selectedAlgorithm]: {
          bestScore: stats.bestScore,
          avgScore: stats.scores.length > 0 ? stats.scores.reduce((a, b) => a + b, 0) / stats.scores.length : 0,
          episodes: stats.episode,
          algorithm: stats.algorithm || algorithmConfigs[selectedAlgorithm].displayName
        }
      }));
    }
  };

  const runTrainingEpisode = async () => {
    if (!agent || !environment) return;
    
    try {
      // Reset environment
      environment.reset();
      environment.difficulty = difficulty;
      
      agent.startEpisode();
      let episodeReward = 0;
      let stepCount = 0;
      const maxSteps = 100;
      
      while (!environment.gameOver && stepCount < maxSteps) {
        const validActions = environment.getValidActions();
        if (validActions.length === 0) break;
        
        let action;
        
        // Different action selection based on algorithm
        if (selectedAlgorithm === 'dqn') {
          const state = environment.getState();
          action = await agent.act(state, validActions, environment);
          state.dispose();
        } else if (selectedAlgorithm === 'policy-gradient') {
          const state = environment.getState();
          action = await agent.selectAction(state, validActions);
          state.dispose();
        } else {
          // MCTS and Heuristic use environment directly
          action = await agent.selectAction(environment);
        }
        
        if (action === null) break;
        
        const stepResult = environment.step(action);
        episodeReward += stepResult.reward;
        
        // Store experience for learning algorithms
        if (selectedAlgorithm === 'dqn') {
          // DQN handles its own experience storage
        } else if (selectedAlgorithm === 'policy-gradient') {
          agent.remember(stepResult.reward);
        }
        
        stepCount++;
        
        // Training for DQN
        if (selectedAlgorithm === 'dqn' && agent.memory.length > agent.batchSize && stepCount % 5 === 0) {
          await agent.replay();
        }
        
        if (stepResult.done) break;
        
        // Add new blocks when needed
        if (environment.availableBlocks.length === 0) {
          environment.availableBlocks = environment.generateCurriculumBlocks();
        }
      }
      
      // End episode
      agent.endEpisode(environment.score);
      
      // Update stats
      const newStats = agent.getStats();
      newStats.episodeReward = episodeReward;
      newStats.episodeSteps = stepCount;
      setTrainingStats(newStats);
      
      // Log progress
      if (episodeRef.current % 10 === 0) {
        console.log(`📊 ${algorithmConfigs[selectedAlgorithm].displayName} Episode ${episodeRef.current}: Score=${environment.score}, Best=${newStats.bestScore}`);
      }
      
    } catch (error) {
      console.error(`❌ Training episode error:`, error);
    }
  };

  const startAIPlay = async () => {
    if (!agent || !environment) {
      alert('Please initialize an agent first!');
      return;
    }
    
    console.log(`🎮 Starting ${algorithmConfigs[selectedAlgorithm].displayName} play...`);
    setIsPlaying(true);
    
    // Set environment to current game state
    environment.setState(grid, availableBlocks, score, difficulty);
    
    playIntervalRef.current = setInterval(async () => {
      await makeAIMove();
    }, playSpeed);
  };

  const stopAIPlay = () => {
    setIsPlaying(false);
    if (playIntervalRef.current) {
      clearInterval(playIntervalRef.current);
      playIntervalRef.current = null;
    }
    console.log(`🛑 ${algorithmConfigs[selectedAlgorithm].displayName} play stopped`);
  };

  const makeAIMove = async () => {
    if (!agent || !environment) return;
    
    if (gameOver) {
      if (autoPlay) {
        setTimeout(() => onResetGame(), 1000);
      } else {
        stopAIPlay();
      }
      return;
    }
    
    try {
      // Update environment with current game state
      environment.setState(grid, availableBlocks, score, difficulty);
      
      const validActions = environment.getValidActions();
      if (validActions.length === 0) return;
      
      let action;
      
      // Select action based on algorithm
      if (selectedAlgorithm === 'dqn') {
        const state = environment.getState();
        action = await agent.selectBestAction(state, validActions);
        state.dispose();
      } else if (selectedAlgorithm === 'policy-gradient') {
        const state = environment.getState();
        action = await agent.selectAction(state, validActions);
        state.dispose();
      } else {
        action = await agent.selectAction(environment);
      }
      
      if (action === null) return;
      
      // Execute action in the game
      const { blockIndex, row, col } = environment.decodeAction(action);
      
      if (blockIndex >= 0 && blockIndex < availableBlocks.length && availableBlocks[blockIndex]) {
        const blockShape = availableBlocks[blockIndex];
        
        console.log(`🧠 ${algorithmConfigs[selectedAlgorithm].displayName} move: Block ${blockIndex} at (${row}, ${col})`);
        
        const moveSuccess = onGameStateUpdate({
          blockIndex,
          row,
          col,
          blockShape: blockShape
        });
        
        if (moveSuccess === false) {
          console.log(`❌ Move failed - stopping AI play`);
          stopAIPlay();
        }
      }
      
    } catch (error) {
      console.error(`❌ AI move error:`, error);
    }
  };

  const runAlgorithmComparison = async () => {
    console.log('🏁 Starting algorithm comparison...');
    
    const algorithms = ['heuristic', 'mcts', 'policy-gradient'];
    const comparisonResults = {};
    
    for (const algo of algorithms) {
      console.log(`🧪 Testing ${algorithmConfigs[algo].displayName}...`);
      
      // Create agent and environment
      const env = new EliteEnvironment();
      const testAgent = AlgorithmSelector.createAgent(algo, env.getStateSize(), env.getMaxActionSpace(), algorithmConfigs[algo].options);
      
      const scores = [];
      const testEpisodes = 10;
      
      for (let episode = 0; episode < testEpisodes; episode++) {
        env.reset();
        testAgent.startEpisode();
        
        let stepCount = 0;
        const maxSteps = 50;
        
        while (!env.gameOver && stepCount < maxSteps) {
          const validActions = env.getValidActions();
          if (validActions.length === 0) break;
          
          const action = await testAgent.selectAction(env);
          if (action === null) break;
          
          env.step(action);
          stepCount++;
          
          if (env.availableBlocks.length === 0) {
            env.availableBlocks = env.generateCurriculumBlocks();
          }
        }
        
        scores.push(env.score);
        testAgent.endEpisode(env.score);
      }
      
      comparisonResults[algo] = {
        algorithm: algorithmConfigs[algo].displayName,
        avgScore: scores.reduce((a, b) => a + b, 0) / scores.length,
        bestScore: Math.max(...scores),
        scores: scores
      };
      
      // Cleanup
      if (testAgent.dispose) testAgent.dispose();
    }
    
    setAlgorithmComparison(comparisonResults);
    console.log('🏁 Algorithm comparison completed:', comparisonResults);
    
    // Show results
    const results = Object.entries(comparisonResults)
      .map(([algo, data]) => `${data.algorithm}: Avg=${data.avgScore.toFixed(1)}, Best=${data.bestScore}`)
      .join('\n');
    
    alert(`🏁 Algorithm Comparison Results:\n\n${results}\n\nCheck console for detailed data.`);
  };

  return (
    <div className="advanced-ai-training-panel" style={{ minHeight: '800px' }}>
      <h3>🧠 Advanced AI Algorithm Center</h3>
      
      {/* Algorithm Selection */}
      <div style={{ 
        background: 'linear-gradient(135deg, rgba(255, 215, 0, 0.15), rgba(255, 165, 0, 0.15))', 
        padding: '20px', 
        borderRadius: '10px', 
        marginBottom: '20px',
        border: '3px solid #FFD700'
      }}>
        <h4 style={{ color: '#FFD700', marginBottom: '15px' }}>🎯 Algorithm Selection</h4>
        
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '15px', marginBottom: '20px' }}>
          {Object.entries(algorithmConfigs).map(([key, config]) => (
            <div
              key={key}
              onClick={() => !isTraining && !isPlaying && setSelectedAlgorithm(key)}
              style={{
                padding: '15px',
                borderRadius: '10px',
                border: selectedAlgorithm === key ? '3px solid #FFD700' : '2px solid #8B4513',
                background: selectedAlgorithm === key ? 'rgba(255, 215, 0, 0.2)' : 'rgba(139, 69, 19, 0.1)',
                cursor: isTraining || isPlaying ? 'not-allowed' : 'pointer',
                opacity: isTraining || isPlaying ? 0.6 : 1,
                transition: 'all 0.3s ease'
              }}
            >
              <div style={{ fontWeight: 'bold', color: '#FFD700', marginBottom: '8px' }}>
                {config.displayName}
              </div>
              <div style={{ fontSize: '12px', color: '#D2B48C', lineHeight: '1.4' }}>
                {config.description}
              </div>
              {selectedAlgorithm === key && (
                <div style={{ 
                  marginTop: '10px', 
                  padding: '8px', 
                  background: 'rgba(255, 215, 0, 0.1)', 
                  borderRadius: '5px',
                  fontSize: '11px',
                  color: '#FFD700'
                }}>
                  ✅ SELECTED
                </div>
              )}
            </div>
          ))}
        </div>
        
        <div style={{ textAlign: 'center', fontSize: '14px', color: '#D2B48C' }}>
          <strong>Current:</strong> {algorithmConfigs[selectedAlgorithm].displayName}
          {(isTraining || isPlaying) && <span style={{ color: '#ffc107' }}> (Cannot change during training/play)</span>}
        </div>
      </div>
      
      {/* Training Controls */}
      <div className="training-section" style={{ marginBottom: '20px' }}>
        <h4>🎓 Training & Testing</h4>
        
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px', marginBottom: '15px' }}>
          <div>
            <label>Episodes:</label>
            <input
              type="number"
              value={trainingEpisodes}
              onChange={(e) => setTrainingEpisodes(parseInt(e.target.value))}
              min="10"
              max="1000"
              disabled={isTraining}
              style={{ marginLeft: '10px', padding: '5px' }}
            />
          </div>
          
          <div>
            <label>Play Speed:</label>
            <input
              type="range"
              value={playSpeed}
              onChange={(e) => setPlaySpeed(parseInt(e.target.value))}
              min="100"
              max="2000"
              step="100"
              disabled={isPlaying}
              style={{ marginLeft: '10px' }}
            />
            <span style={{ marginLeft: '10px' }}>{playSpeed}ms</span>
          </div>
        </div>
        
        <div className="button-group" style={{ display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
          <button
            onClick={startTraining}
            disabled={isTraining || !agent}
            className="btn"
            style={{ background: 'linear-gradient(145deg, #28a745, #20c997)' }}
          >
            {isTraining ? `Training... (${episodeRef.current}/${trainingEpisodes})` : '🎓 Start Training'}
          </button>
          
          <button
            onClick={stopTraining}
            disabled={!isTraining}
            className="btn"
            style={{ background: 'linear-gradient(145deg, #dc3545, #c82333)' }}
          >
            🛑 Stop Training
          </button>
          
          <button
            onClick={startAIPlay}
            disabled={isPlaying || !agent}
            className="btn"
            style={{ background: 'linear-gradient(145deg, #007bff, #0056b3)' }}
          >
            {isPlaying ? '🎮 Playing...' : '🎮 Start AI Play'}
          </button>
          
          <button
            onClick={stopAIPlay}
            disabled={!isPlaying}
            className="btn"
            style={{ background: 'linear-gradient(145deg, #6c757d, #545b62)' }}
          >
            🛑 Stop AI
          </button>
          
          <button
            onClick={runAlgorithmComparison}
            disabled={isTraining || isPlaying}
            className="btn"
            style={{ background: 'linear-gradient(145deg, #6f42c1, #5a32a3)' }}
          >
            🏁 Compare Algorithms
          </button>
        </div>
        
        <div style={{ marginTop: '10px' }}>
          <label>
            <input
              type="checkbox"
              checked={autoPlay}
              onChange={(e) => setAutoPlay(e.target.checked)}
            />
            <span style={{ marginLeft: '8px' }}>Auto-restart games</span>
          </label>
        </div>
      </div>
      
      {/* Algorithm Comparison Results */}
      {Object.keys(algorithmComparison).length > 0 && (
        <div style={{ 
          background: 'rgba(139, 69, 19, 0.2)', 
          padding: '20px', 
          borderRadius: '10px', 
          marginBottom: '20px',
          border: '2px solid #8B4513'
        }}>
          <h4 style={{ color: '#FFD700', marginBottom: '15px' }}>🏁 Algorithm Performance Comparison</h4>
          
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '15px' }}>
            {Object.entries(algorithmComparison).map(([algo, data]) => (
              <div key={algo} style={{
                background: 'rgba(255, 215, 0, 0.1)',
                padding: '15px',
                borderRadius: '8px',
                border: '1px solid #FFD700'
              }}>
                <div style={{ fontWeight: 'bold', color: '#FFD700', marginBottom: '10px' }}>
                  {data.algorithm}
                </div>
                <div style={{ fontSize: '14px', color: '#D2B48C' }}>
                  <div>Best Score: <strong style={{ color: '#28a745' }}>{data.bestScore}</strong></div>
                  <div>Avg Score: <strong style={{ color: '#ffc107' }}>{data.avgScore?.toFixed(1)}</strong></div>
                  <div>Episodes: <strong>{data.episodes || 'N/A'}</strong></div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
      
      {/* Current Algorithm Stats */}
      {trainingStats.episode && (
        <div style={{ 
          background: 'rgba(139, 69, 19, 0.2)', 
          padding: '20px', 
          borderRadius: '10px', 
          marginBottom: '20px',
          border: '2px solid #8B4513'
        }}>
          <h4 style={{ color: '#FFD700', marginBottom: '15px' }}>
            📊 {algorithmConfigs[selectedAlgorithm].displayName} Statistics
          </h4>
          
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: '15px' }}>
            <div style={{ textAlign: 'center' }}>
              <div style={{ color: '#D2B48C', fontSize: '12px' }}>Episodes</div>
              <div style={{ color: '#FFD700', fontSize: '18px', fontWeight: 'bold' }}>{trainingStats.episode}</div>
            </div>
            <div style={{ textAlign: 'center' }}>
              <div style={{ color: '#D2B48C', fontSize: '12px' }}>Best Score</div>
              <div style={{ color: '#28a745', fontSize: '18px', fontWeight: 'bold' }}>{trainingStats.bestScore}</div>
            </div>
            {trainingStats.avgLoss !== undefined && (
              <div style={{ textAlign: 'center' }}>
                <div style={{ color: '#D2B48C', fontSize: '12px' }}>Avg Loss</div>
                <div style={{ color: '#FF6B6B', fontSize: '18px', fontWeight: 'bold' }}>{trainingStats.avgLoss?.toFixed(4)}</div>
              </div>
            )}
            {trainingStats.avgDecisionTime !== undefined && (
              <div style={{ textAlign: 'center' }}>
                <div style={{ color: '#D2B48C', fontSize: '12px' }}>Decision Time</div>
                <div style={{ color: '#00CED1', fontSize: '18px', fontWeight: 'bold' }}>{trainingStats.avgDecisionTime?.toFixed(1)}ms</div>
              </div>
            )}
            {trainingStats.totalSimulations !== undefined && (
              <div style={{ textAlign: 'center' }}>
                <div style={{ color: '#D2B48C', fontSize: '12px' }}>Total Simulations</div>
                <div style={{ color: '#ffc107', fontSize: '18px', fontWeight: 'bold' }}>{trainingStats.totalSimulations}</div>
              </div>
            )}
          </div>
        </div>
      )}
      
      {/* Visualization */}
      <div style={{
        maxHeight: '600px',
        overflowY: 'auto',
        overflowX: 'hidden',
        border: '2px solid #8B4513',
        borderRadius: '10px',
        padding: '15px',
        background: 'rgba(139, 69, 19, 0.1)',
        marginTop: '20px'
      }}>
        <AIVisualization 
          trainingStats={trainingStats} 
          isTraining={isTraining}
        />
      </div>
    </div>
  );
}

export default AdvancedAITrainingPanel; 