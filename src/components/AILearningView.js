import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { DndProvider } from 'react-dnd';
import { HTML5Backend } from 'react-dnd-html5-backend';
import { DQNAgent } from '../ai/DQNAgent';
import { DQNEnvironment } from '../ai/DQNEnvironment';
import { EliteDQNAgent } from '../ai/EliteDQNAgent';
import { ConvDQNAgent } from '../ai/ConvDQNAgent';
import { ConvDQNEnvironment } from '../ai/ConvDQNEnvironment';
import { AlgorithmSelector } from '../ai/AdvancedAIAgents';
import AIVisualization from './AIVisualization';
import GameBoard from './GameBoard';
import BlockTray from './BlockTray';
import ScoreDisplay from './ScoreDisplay';

function AILearningView({ onNavigate }) {
  // Algorithm Selection
  const [selectedAlgorithm, setSelectedAlgorithm] = useState('dqn');
  
  // Training State  
  const [isTraining, setIsTraining] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [visualTraining, setVisualTraining] = useState(true);
  const [trainingSpeed, setTrainingSpeed] = useState(50);
  const [aiPlayInterval, setAiPlayInterval] = useState(null);
  
  // Game State
  const [grid, setGrid] = useState(Array(9).fill(null).map(() => Array(9).fill(false)));
  const [availableBlocks, setAvailableBlocks] = useState([]);
  const [score, setScore] = useState(0);
  const [bestScore, setBestScore] = useState(0);
  const [difficulty] = useState('normal');
  
  // Training Progress
  const [currentEpisode, setCurrentEpisode] = useState(0);
  const [episodeScore, setEpisodeScore] = useState(0);
  const [episodeSteps, setEpisodeSteps] = useState(0);
  const [trainingStats, setTrainingStats] = useState({
    losses: [],
    rewards: [],
    scores: [],
    epsilon: 1.0
  });

  // AI Components
  const [agent, setAgent] = useState(null);
  const [environment, setEnvironment] = useState(null);
  
  // UI State
  const [showTips, setShowTips] = useState(false);
  const [testResults] = useState({});
  
  // Refs for training control
  const trainingRef = useRef(false);
  const episodeRef = useRef(0);
  const stepRef = useRef(0);

  // Algorithm configurations - ALL USE SAME ENVIRONMENT FOR FAIR COMPARISON
  const algorithmConfigs = useMemo(() => ({
    'dqn': {
      name: 'Original DQN',
      description: 'Deep Q-Network with experience replay',
      agentClass: DQNAgent,
      environmentClass: DQNEnvironment, // Same environment for all
      options: {
        learningRate: 0.001,
        epsilon: 0.95,
        epsilonDecay: 0.995,
        gamma: 0.99,
        batchSize: 32
      }
    },
    'elite-dqn': {
      name: 'Elite DQN',
      description: 'Advanced DQN with prioritized replay and sophisticated strategies',
      agentClass: EliteDQNAgent,
      environmentClass: DQNEnvironment, // CHANGED: Use same environment for fair comparison
      options: {
        learningRate: 0.0003,
        epsilon: 1.0,
        epsilonDecay: 0.9995,
        gamma: 0.99,
        batchSize: 64
      }
    },
    'visual-cnn': {
      name: 'Visual CNN DQN',
      description: 'Revolutionary CNN-based DQN with 12×12 grid and 4-channel visual intelligence',
      agentClass: ConvDQNAgent,
      environmentClass: ConvDQNEnvironment,
      options: {
        learningRate: 0.0005,
        epsilon: 0.8,
        epsilonDecay: 0.996,
        gamma: 0.99,
        batchSize: 32,
        patternGuidedRate: 0.4
      }
    },
    'mcts': {
      name: 'Monte Carlo Tree Search',
      description: 'Tree search with intelligent rollouts and UCB1 exploration',
      agentClass: null, // Will use AlgorithmSelector
      environmentClass: DQNEnvironment, // CHANGED: Use same environment for fair comparison
      options: {
        simulations: 50,
        explorationConstant: 1.414,
        maxDepth: 15
      }
    },
    'policy-gradient': {
      name: 'Policy Gradient (REINFORCE)',
      description: 'Direct policy optimization with action masking',
      agentClass: null, // Will use AlgorithmSelector
      environmentClass: DQNEnvironment, // CHANGED: Use same environment for fair comparison
      options: {
        learningRate: 0.001,
        gamma: 0.99,
        entropyCoeff: 0.01
      }
    },
    'heuristic': {
      name: 'Hybrid Heuristic',
      description: 'Hand-crafted rules with 2-step lookahead (no training required)',
      agentClass: null, // Will use AlgorithmSelector
      environmentClass: DQNEnvironment, // CHANGED: Use same environment for fair comparison
      options: {
        lookaheadDepth: 2,
        lineCompletionWeight: 1000,
        futureOpportunityWeight: 100
      }
    }
  }), []);

  // Initialize agent and environment when algorithm changes
  const initializeAgent = useCallback(() => {
    console.log(`🤖 Initializing ${algorithmConfigs[selectedAlgorithm].name}...`);
    
    const config = algorithmConfigs[selectedAlgorithm];
    const env = new config.environmentClass();
    
    let newAgent;
    if (config.agentClass) {
      // Use specific agent class (DQN variants)
      let stateSize;
      if (selectedAlgorithm === 'visual-cnn') {
        // ConvDQNAgent expects visual state size as array [channels, height, width]
        stateSize = env.getVisualStateSize();
      } else {
        // Other agents expect state size as number
        stateSize = env.getStateSize();
      }
      
      newAgent = new config.agentClass(stateSize, env.getMaxActionSpace(), config.options);
    } else {
      // Use AlgorithmSelector for advanced algorithms
      newAgent = AlgorithmSelector.createAgent(selectedAlgorithm, env.getStateSize(), env.getMaxActionSpace(), config.options);
    }
    
    setAgent(newAgent);
    setEnvironment(env);
    
    // Reset game state
    env.reset();
    setGrid(env.grid.map(row => [...row]));
    setAvailableBlocks(env.availableBlocks.map(block => block.map(row => [...row])));
    setScore(env.score);
    setBestScore(Math.max(bestScore, env.score));
    
    console.log(`✅ ${config.name} initialized successfully!`);
  }, [selectedAlgorithm, algorithmConfigs, bestScore]);

  useEffect(() => {
    initializeAgent();
  }, [initializeAgent]);

  const startTraining = async () => {
    if (!agent || !environment) {
      console.error('Agent or environment not initialized');
      return;
    }

    setIsTraining(true);
    setIsPaused(false);
    trainingRef.current = true;
    episodeRef.current = 0;

    console.log(`🚀 Starting ${algorithmConfigs[selectedAlgorithm].name} training...`);

    while (trainingRef.current && !isPaused) {
      await runTrainingEpisode();
      
      // Update stats after each episode
      if (agent.getStats && typeof agent.getStats === 'function') {
        const stats = agent.getStats();
        setTrainingStats(stats);
        setBestScore(Math.max(bestScore, stats.bestScore || 0));
      }
      
      // Small delay for UI updates
      await new Promise(resolve => setTimeout(resolve, 50));
      
      // Check if we've reached the episode limit
      if (episodeRef.current >= 500) {
        break;
      }
    }

    setIsTraining(false);
  };

  const pauseTraining = () => {
    setIsPaused(true);
  };

  const resumeTraining = () => {
    setIsPaused(false);
    if (trainingRef.current) {
      startTraining();
    }
  };

  const stopTraining = () => {
    trainingRef.current = false;
    setIsTraining(false);
    setIsPaused(false);
    
    console.log(`🛑 ${algorithmConfigs[selectedAlgorithm].name} training stopped`);
  };

  const resetTraining = () => {
    stopTraining();
    
    // Reset agent
    if (agent) {
      agent.dispose();
    }
    
    // Reinitialize
    initializeAgent();
    
    // Reset UI state
    setCurrentEpisode(0);
    setEpisodeScore(0);
    setEpisodeSteps(0);
    setTrainingStats({});
    episodeRef.current = 0;
    stepRef.current = 0;
    
    console.log(`🔄 ${algorithmConfigs[selectedAlgorithm].name} training reset`);
  };

  const runTrainingEpisode = async () => {
    if (!agent || !environment) return;
    
    try {
      // Reset environment for new episode
      environment.reset();
      
      // Store initial score for fair comparison
      const initialScore = environment.score;
      
      // Update visual state
      if (visualTraining) {
        setGrid(environment.grid.map(row => [...row]));
        setAvailableBlocks(environment.availableBlocks.map(block => block.map(row => [...row])));
        setCurrentEpisode(episodeRef.current);
        setEpisodeScore(initialScore);
        setEpisodeSteps(0);
        stepRef.current = 0;
      }

      // Start episode for agent
      if (agent.startEpisode && typeof agent.startEpisode === 'function') {
        agent.startEpisode();
      }
      
      // Get initial state using DQNEnvironment method
      let state = environment.getState(); // DQNEnvironment always has getState() method
      
      let done = false;
      let stepCount = 0;
      const maxSteps = selectedAlgorithm === 'heuristic' ? 50 : 100;
      let totalReward = 0; // Track internal AI rewards separately from game score
      
      while (!done && stepCount < maxSteps && trainingRef.current) {
        const validActions = environment.getValidActions();
        
        if (validActions.length === 0) {
          done = true;
          break;
        }
        
        let action;
        
        // Algorithm-specific action selection
        switch (selectedAlgorithm) {
          case 'mcts':
            action = await agent.selectAction(environment);
            break;
          case 'policy-gradient':
            action = await agent.selectAction(state, validActions);
            break;
          case 'heuristic':
            action = await agent.selectAction(environment);
            break;
          case 'visual-cnn':
          case 'elite-dqn':
          case 'dqn':
          default:
            action = await agent.act(state, validActions, environment);
            break;
        }
        
        if (action === null || action === undefined) {
          console.warn(`⚠️ ${selectedAlgorithm} returned null action, using random`);
          action = validActions[Math.floor(Math.random() * validActions.length)];
        }
        
        const stepResult = environment.step(action);
        
        // IMPORTANT: Separate reward (AI motivation) from score (game performance)
        totalReward += stepResult.reward; // Internal AI reward for learning
        const actualGameScore = environment.score; // Real game score for fair comparison
        
        // Visual feedback - show REAL game score, not AI rewards
        if (visualTraining) {
          setGrid(environment.grid.map(row => [...row]));
          setAvailableBlocks(environment.availableBlocks.map(block => block.map(row => [...row])));
          setEpisodeScore(actualGameScore); // Show real game score
          setEpisodeSteps(stepCount + 1);
          stepRef.current = stepCount + 1;
          
          // Visual delay for training observation and UI responsiveness
          if (stepCount % 3 === 0) {
            await new Promise(resolve => setTimeout(resolve, Math.max(10, 100 - trainingSpeed)));
          }
        } else {
          // Even without visual training, yield periodically to prevent UI freezing
          if (stepCount % 10 === 0) {
            await new Promise(resolve => setTimeout(resolve, 5));
          }
        }
        
        // Algorithm-specific experience storage and training
        switch (selectedAlgorithm) {
          case 'mcts':
            // MCTS doesn't need experience storage, it learns through simulations
            break;
          case 'policy-gradient':
            // Store reward for policy gradient training
            if (agent.remember && typeof agent.remember === 'function') {
              agent.remember(stepResult.reward);
            }
            break;
          case 'heuristic':
            // Heuristic doesn't need training, it's rule-based
            break;
          case 'visual-cnn':
          case 'elite-dqn':
          case 'dqn':
          default:
            // Store experience for neural network agents
            if (agent.remember && typeof agent.remember === 'function') {
              agent.remember(state, action, stepResult.reward, stepResult.state, stepResult.done);
            }
            
            // Training step for neural network agents
            if (agent.replay && typeof agent.replay === 'function' && 
                agent.memory && agent.memory.length > (agent.batchSize || 32) && 
                stepCount % 4 === 0) {
              await agent.replay();
            }
            break;
        }
        
        state = stepResult.state;
        done = stepResult.done;
        stepCount++;
        
        // Add new blocks when needed - use environment-specific block generation
        if (environment.availableBlocks.length === 0 && !done) {
          if (selectedAlgorithm === 'visual-cnn') {
            environment.availableBlocks = environment.generateVisualBlocks(); // ConvDQNEnvironment method
          } else {
            environment.availableBlocks = environment.generateCurriculumBlocks(); // DQNEnvironment method
          }
          done = environment.checkGameOver();
          
          if (visualTraining) {
            setAvailableBlocks(environment.availableBlocks.map(block => block.map(row => [...row])));
          }
        }
      }
      
      // CRITICAL: Use REAL game score for episode ending, not AI rewards
      const finalGameScore = environment.score;
      
      // Algorithm-specific episode ending
      switch (selectedAlgorithm) {
        case 'policy-gradient':
          // Train policy gradient at end of episode
          if (agent.train && typeof agent.train === 'function') {
            await agent.train();
          }
          if (agent.endEpisode && typeof agent.endEpisode === 'function') {
            await agent.endEpisode(finalGameScore); // Use real game score
          }
          break;
        case 'mcts':
        case 'heuristic':
          // End episode for non-neural agents
          if (agent.endEpisode && typeof agent.endEpisode === 'function') {
            agent.endEpisode(finalGameScore); // Use real game score
          }
          break;
        case 'visual-cnn':
        case 'elite-dqn':
        case 'dqn':
        default:
          // End episode for neural network agents
          if (agent.endEpisode && typeof agent.endEpisode === 'function') {
            agent.endEpisode(totalReward, finalGameScore, environment); // Reward for learning, score for comparison
          }
          break;
      }
      
      episodeRef.current++;
      
      // Update training stats for visualization - ensure we track REAL performance
      if (agent.getStats && typeof agent.getStats === 'function') {
        const stats = agent.getStats();
        
        // ENSURE STATS REFLECT REAL GAME PERFORMANCE
        stats.lastGameScore = finalGameScore;
        stats.totalReward = totalReward; // Keep reward separate for AI analysis
        stats.actualPerformance = finalGameScore; // Real game performance
        
        setTrainingStats(stats);
        setBestScore(Math.max(bestScore, finalGameScore)); // Use real game score for best score
        
        // Log progress every 10 episodes - show REAL performance
        if (episodeRef.current % 10 === 0) {
          console.log(`📊 Episode ${episodeRef.current}: REAL Score=${finalGameScore}, AI Reward=${totalReward.toFixed(1)}, Best=${Math.max(bestScore, finalGameScore)}, Algorithm=${selectedAlgorithm}`);
        }
      }
      
      // Clean up state tensor
      if (state && state.dispose && typeof state.dispose === 'function') {
        state.dispose();
      }
      
    } catch (error) {
      console.error(`❌ Training episode error for ${algorithmConfigs[selectedAlgorithm].name}:`, error);
    }
  };

  const loadModel = async () => {
    if (!agent) {
      alert('No agent available to load into!');
      return;
    }
    
    try {
      const algorithmName = algorithmConfigs[selectedAlgorithm].name;
      console.log(`📁 Loading ${algorithmName} model...`);
      
      let success = false;
      
      switch (selectedAlgorithm) {
        case 'mcts':
        case 'heuristic':
          // For non-neural algorithms, try to load saved stats
          const savedKeys = Object.keys(localStorage).filter(key => 
            key.includes(selectedAlgorithm) && key.includes('model')
          );
          
          if (savedKeys.length === 0) {
            throw new Error(`No saved ${algorithmName} models found`);
          }
          
          // Load the most recent model
          const latestKey = savedKeys.sort().pop();
          const savedData = JSON.parse(localStorage.getItem(latestKey));
          
          if (savedData.algorithm === selectedAlgorithm) {
            // Restore stats if possible
            if (agent.loadStats && typeof agent.loadStats === 'function') {
              agent.loadStats(savedData.stats);
            }
            success = true;
            console.log(`✅ ${algorithmName} configuration loaded`);
          }
          break;
          
        case 'policy-gradient':
          // Load policy gradient model
          if (agent.loadModel && typeof agent.loadModel === 'function') {
            // Try to find a saved model
            const modelKeys = Object.keys(localStorage).filter(key => 
              key.includes('policy-gradient') && key.includes('model')
            );
            
            if (modelKeys.length > 0) {
              const latestModelKey = modelKeys.sort().pop();
              success = await agent.loadModel(latestModelKey);
            } else {
              throw new Error('No saved Policy Gradient models found');
            }
          } else {
            throw new Error('Policy Gradient agent does not support model loading');
          }
          break;
          
        case 'visual-cnn':
        case 'elite-dqn':
        case 'dqn':
        default:
          // Load neural network models
          if (agent.loadModel && typeof agent.loadModel === 'function') {
            // Try to find a saved model for this algorithm
            const modelKeys = Object.keys(localStorage).filter(key => 
              key.includes(selectedAlgorithm) && key.includes('model')
            );
            
            if (modelKeys.length > 0) {
              const latestModelKey = modelKeys.sort().pop();
              success = await agent.loadModel(latestModelKey);
            } else {
              throw new Error(`No saved ${algorithmName} models found`);
            }
          } else {
            throw new Error('Agent does not support model loading');
          }
          break;
      }
      
      if (success) {
        // Update training stats
        if (agent.getStats && typeof agent.getStats === 'function') {
          const stats = agent.getStats();
          setTrainingStats(stats);
          setBestScore(stats.bestScore || 0);
        }
        
        alert(`✅ ${algorithmName} model loaded successfully!\n\nBest Score: ${agent.getStats().bestScore || 0}\nEpisodes: ${agent.getStats().episode || 0}`);
      } else {
        throw new Error('Model loading failed');
      }
      
    } catch (error) {
      console.error('❌ Model load error:', error);
      alert(`❌ Failed to load ${algorithmConfigs[selectedAlgorithm].name} model: ${error.message}`);
    }
  };

  const downloadModel = async () => {
    if (!agent) {
      alert('No agent available to download!');
      return;
    }

    try {
      const algorithmName = algorithmConfigs[selectedAlgorithm].name;
      const timestamp = new Date().toISOString().replace(/[:.]/g, '-').split('.')[0];
      const modelName = `${selectedAlgorithm}-model-${timestamp}`;
      
      console.log(`🔽 Downloading ${algorithmName} model...`);
      
      switch (selectedAlgorithm) {
        case 'mcts':
        case 'heuristic':
          // For non-neural algorithms, download configuration and stats as JSON
          const configData = {
            algorithm: selectedAlgorithm,
            algorithmName: algorithmName,
            stats: agent.getStats(),
            timestamp: timestamp,
            version: '1.0',
            configuration: {
              stateSize: agent.stateSize,
              actionSize: agent.actionSize,
              // Add algorithm-specific config
              ...(selectedAlgorithm === 'mcts' ? {
                explorationConstant: agent.explorationConstant,
                maxSimulations: agent.maxSimulations,
                maxDepth: agent.maxDepth,
                rolloutPolicy: agent.rolloutPolicy
              } : {}),
              ...(selectedAlgorithm === 'heuristic' ? {
                weights: agent.weights,
                lookaheadDepth: agent.lookaheadDepth
              } : {})
            }
          };
          
          const configBlob = new Blob([JSON.stringify(configData, null, 2)], {
            type: 'application/json'
          });
          const configUrl = URL.createObjectURL(configBlob);
          const configLink = document.createElement('a');
          configLink.href = configUrl;
          configLink.download = `${modelName}-config.json`;
          document.body.appendChild(configLink);
          configLink.click();
          document.body.removeChild(configLink);
          URL.revokeObjectURL(configUrl);
          
          alert(`🔽 ${algorithmName} Configuration Downloaded!\n\nFile: ${modelName}-config.json\nAlgorithm: ${algorithmName}\nBest Score: ${agent.getStats().bestScore || 0}\nEpisodes: ${agent.getStats().episode || 0}`);
          break;
          
        case 'policy-gradient':
          // For Policy Gradient, try to download neural network if available
          if (agent.policyNetwork && agent.policyNetwork.save) {
            await agent.policyNetwork.save(`downloads://${modelName}`);
            
            // Also save agent state
            const pgState = {
              algorithm: selectedAlgorithm,
              algorithmName: algorithmName,
              stats: agent.getStats(),
              timestamp: timestamp,
              version: '1.0',
              hyperparameters: {
                learningRate: agent.learningRate,
                gamma: agent.gamma,
                entropy_coeff: agent.entropy_coeff
              }
            };
            
            const pgBlob = new Blob([JSON.stringify(pgState, null, 2)], {
              type: 'application/json'
            });
            const pgUrl = URL.createObjectURL(pgBlob);
            const pgLink = document.createElement('a');
            pgLink.href = pgUrl;
            pgLink.download = `${modelName}-state.json`;
            document.body.appendChild(pgLink);
            pgLink.click();
            document.body.removeChild(pgLink);
            URL.revokeObjectURL(pgUrl);
            
            alert(`🔽 ${algorithmName} Model Downloaded!\n\nFiles:\n• ${modelName}.json (model)\n• ${modelName}.bin (weights)\n• ${modelName}-state.json (training state)\n\nBest Score: ${agent.getStats().bestScore || 0}`);
          } else {
            throw new Error('Policy Gradient model does not support downloading');
          }
          break;
          
        case 'visual-cnn':
        case 'elite-dqn':
        case 'dqn':
        default:
          // For neural network models, download TensorFlow.js model
          if (agent.qNetwork && agent.qNetwork.save) {
            await agent.qNetwork.save(`downloads://${modelName}`);
            
            // Create and download agent state
            const stats = agent.getStats();
            const agentState = {
              algorithm: selectedAlgorithm,
              algorithmName: algorithmName,
              timestamp: timestamp,
              version: '1.0',
              stats: stats,
              hyperparameters: {
                learningRate: agent.learningRate,
                epsilon: agent.epsilon,
                epsilonMin: agent.epsilonMin,
                epsilonDecay: agent.epsilonDecay,
                gamma: agent.gamma,
                batchSize: agent.batchSize,
                memorySize: agent.memorySize,
                targetUpdateFreq: agent.targetUpdateFreq
              }
            };

            const stateBlob = new Blob([JSON.stringify(agentState, null, 2)], {
              type: 'application/json'
            });
            const stateUrl = URL.createObjectURL(stateBlob);
            const stateLink = document.createElement('a');
            stateLink.href = stateUrl;
            stateLink.download = `${modelName}-state.json`;
            document.body.appendChild(stateLink);
            stateLink.click();
            document.body.removeChild(stateLink);
            URL.revokeObjectURL(stateUrl);

            alert(`🔽 ${algorithmName} Model Downloaded!\n\nFiles:\n• ${modelName}.json (model architecture)\n• ${modelName}.bin (model weights)\n• ${modelName}-state.json (training progress)\n\nBest Score: ${stats.bestScore || 0}\nEpisodes: ${stats.episode || 0}`);
          } else {
            throw new Error('Neural network model does not support downloading');
          }
          break;
      }
      
    } catch (error) {
      console.error('❌ Model download error:', error);
      alert(`❌ Failed to download ${algorithmConfigs[selectedAlgorithm].name} model: ${error.message}`);
    }
  };

  const TipsModal = () => (
    <div className="tips-modal-overlay" onClick={() => setShowTips(false)}>
      <div className="tips-modal" onClick={(e) => e.stopPropagation()}>
        <div className="tips-header">
          <h3>🎓 AI Learning Tutorial</h3>
          <button className="close-btn" onClick={() => setShowTips(false)}>×</button>
        </div>
        <div className="tips-content">
          <div className="tip-section">
            <h4>🚀 Getting Started</h4>
            <ul>
              <li><strong>Start Training:</strong> Click the "Start Learning" button to begin AI training</li>
              <li><strong>Watch Live:</strong> Observe the AI playing in real-time on the center game board</li>
              <li><strong>Monitor Progress:</strong> Check the graphs on the right panel for learning progress</li>
            </ul>
          </div>
          
          <div className="tip-section">
            <h4>🎮 Training Controls</h4>
            <ul>
              <li><strong>Start Training:</strong> Begin AI learning process</li>
              <li><strong>Pause/Resume:</strong> Temporarily halt training and continue later</li>
              <li><strong>Stop:</strong> End training session completely</li>
              <li><strong>Reset All Stats:</strong> Clear all progress and start fresh training</li>
              <li><strong>Speed Control:</strong> Adjust how fast the AI trains during learning</li>
            </ul>
          </div>
          
          <div className="tip-section">
            <h4>💾 Model Management</h4>
            <ul>
              <li><strong>Download Model:</strong> Save your trained AI to your computer</li>
              <li><strong>Load Model:</strong> Restore a previously saved AI model</li>
            </ul>
          </div>
          
          <div className="tip-section">
            <h4>📊 Understanding the Graphs</h4>
            <ul>
              <li><strong>Gold Line (Reward):</strong> Higher = AI is getting better</li>
              <li><strong>Cyan Line (Epsilon):</strong> Lower = AI is more confident</li>
              <li><strong>Red Area (Loss):</strong> Lower = Better learning stability</li>
              <li><strong>Network Diagram:</strong> Shows AI brain architecture</li>
            </ul>
          </div>
          
          <div className="tip-section">
            <h4>💡 Training Tips</h4>
            <ul>
              <li>Start with <strong>100-500 episodes</strong> for initial learning</li>
              <li>Look for <strong>upward trending rewards</strong> as progress indicator</li>
              <li>If performance plateaus, try <strong>1000+ episodes</strong></li>
              <li>Good performance: average rewards above <strong>200 points</strong></li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );

  return (
    <div className="ai-learning-view">
      <div className="header">
        <button 
          className="back-btn"
          onClick={() => onNavigate('menu')}
        >
          ← Back to Menu
        </button>
        
        <h2>🧠 AI Learning Laboratory</h2>
        
        <button 
          className="tips-btn"
          onClick={() => setShowTips(true)}
        >
          💡 Tips
        </button>
      </div>

      <div className="main-content">
        {/* Algorithm Selection */}
        <div className="algorithm-selection" style={{
          background: 'rgba(139, 69, 19, 0.3)',
          padding: '20px',
          borderRadius: '12px',
          border: '2px solid #8B4513',
          marginBottom: '20px'
        }}>
          <h3 style={{ color: '#FFD700', marginBottom: '15px', textAlign: 'center' }}>
            🤖 Select AI Algorithm
          </h3>
          
          <div style={{ display: 'flex', gap: '15px', marginBottom: '15px' }}>
            <select
              value={selectedAlgorithm}
              onChange={(e) => setSelectedAlgorithm(e.target.value)}
              disabled={isTraining || isPlaying}
              style={{
                flex: 1,
                padding: '12px',
                borderRadius: '8px',
                border: '2px solid #8B4513',
                background: 'rgba(255, 215, 0, 0.1)',
                color: '#FFD700',
                fontSize: '16px',
                fontWeight: 'bold'
              }}
            >
              {Object.entries(algorithmConfigs).map(([key, config]) => (
                <option key={key} value={key} style={{ background: '#2C1810', color: '#FFD700' }}>
                  {config.name}
                </option>
              ))}
            </select>
          </div>
          
          <div style={{
            background: 'rgba(255, 215, 0, 0.1)',
            padding: '12px',
            borderRadius: '8px',
            border: '1px solid #FFD700'
          }}>
            <div style={{ color: '#FFD700', fontWeight: 'bold', marginBottom: '5px' }}>
              {algorithmConfigs[selectedAlgorithm].name}
            </div>
            <div style={{ color: '#D2B48C', fontSize: '14px' }}>
              {algorithmConfigs[selectedAlgorithm].description}
            </div>
          </div>
        </div>

        {/* Performance Dashboard */}
        <div className="performance-dashboard" style={{
          background: 'rgba(139, 69, 19, 0.3)',
          padding: '20px',
          borderRadius: '12px',
          border: '2px solid #8B4513',
          marginBottom: '20px'
        }}>
          <h3 style={{ color: '#FFD700', marginBottom: '15px', textAlign: 'center' }}>
            📊 Performance Dashboard
          </h3>
          
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '15px' }}>
            {/* Current Algorithm Stats */}
            <div style={{
              background: 'rgba(255, 215, 0, 0.1)',
              padding: '15px',
              borderRadius: '8px',
              border: '1px solid #FFD700'
            }}>
              <div style={{ color: '#FFD700', fontWeight: 'bold', marginBottom: '10px' }}>
                🤖 Current: {algorithmConfigs[selectedAlgorithm].name}
              </div>
              <div style={{ color: '#D2B48C', fontSize: '14px', marginBottom: '8px' }}>
                Episodes: {trainingStats.episode || 0}
              </div>
              <div style={{ color: '#D2B48C', fontSize: '14px', marginBottom: '8px' }}>
                Best Score: <span style={{ color: '#28a745', fontWeight: 'bold' }}>
                  {trainingStats.bestScore || 0}
                </span>
              </div>
              <div style={{ color: '#D2B48C', fontSize: '14px' }}>
                Avg Score: <span style={{ color: '#17a2b8', fontWeight: 'bold' }}>
                  {trainingStats.avgScore?.toFixed(1) || '0.0'}
                </span>
              </div>
            </div>

            {/* Test Results Comparison */}
            <div style={{
              background: 'rgba(255, 215, 0, 0.1)',
              padding: '15px',
              borderRadius: '8px',
              border: '1px solid #FFD700'
            }}>
              <div style={{ color: '#FFD700', fontWeight: 'bold', marginBottom: '10px' }}>
                🏆 Best Test Results
              </div>
              {Object.keys(testResults).length > 0 ? (
                Object.entries(testResults)
                  .sort((a, b) => b[1].bestScore - a[1].bestScore)
                  .slice(0, 3)
                  .map(([algorithm, results], index) => (
                    <div key={algorithm} style={{ 
                      color: '#D2B48C', 
                      fontSize: '12px', 
                      marginBottom: '4px',
                      display: 'flex',
                      justifyContent: 'space-between'
                    }}>
                      <span>
                        {index === 0 ? '🥇' : index === 1 ? '🥈' : '🥉'} {results.algorithmName}
                      </span>
                      <span style={{ color: '#28a745', fontWeight: 'bold' }}>
                        {results.bestScore}
                      </span>
                    </div>
                  ))
              ) : (
                <div style={{ color: '#D2B48C', fontSize: '12px', fontStyle: 'italic' }}>
                  No test results yet. Click "🎯 Test Performance" to compare algorithms.
                </div>
              )}
            </div>

            {/* Global Best Score */}
            <div style={{
              background: 'rgba(255, 215, 0, 0.1)',
              padding: '15px',
              borderRadius: '8px',
              border: '1px solid #FFD700'
            }}>
              <div style={{ color: '#FFD700', fontWeight: 'bold', marginBottom: '10px' }}>
                🌟 Global Best
              </div>
              <div style={{ 
                color: '#28a745', 
                fontSize: '24px', 
                fontWeight: 'bold',
                textAlign: 'center'
              }}>
                {Math.max(
                  bestScore,
                  trainingStats.bestScore || 0,
                  ...Object.values(testResults).map(r => r.bestScore || 0)
                )}
              </div>
              <div style={{ 
                color: '#D2B48C', 
                fontSize: '12px', 
                textAlign: 'center',
                marginTop: '5px'
              }}>
                Highest score achieved
              </div>
            </div>
          </div>
        </div>

        {/* Training Controls */}
        <div className="training-controls" style={{
          display: 'grid',
          gridTemplateColumns: '1fr 1fr 1fr',
          gap: '15px',
          marginBottom: '20px'
        }}>
          <div className="control-group">
            <label style={{ color: '#FFD700', fontWeight: 'bold', marginBottom: '8px', display: 'block' }}>
              Training Controls
            </label>
            <div style={{ display: 'flex', gap: '10px' }}>
              {!isTraining ? (
                <button
                  onClick={startTraining}
                  disabled={!agent || isPlaying}
                  className="btn primary"
                  style={{
                    flex: 1,
                    padding: '12px',
                    borderRadius: '8px',
                    border: 'none',
                    background: 'linear-gradient(145deg, #28a745, #20c997)',
                    color: 'white',
                    fontWeight: 'bold'
                  }}
                >
                  🚀 Start Training
                </button>
              ) : (
                <>
                  {!isPaused ? (
                    <button
                      onClick={pauseTraining}
                      className="btn warning"
                      style={{
                        flex: 1,
                        padding: '12px',
                        borderRadius: '8px',
                        border: 'none',
                        background: 'linear-gradient(145deg, #ffc107, #fd7e14)',
                        color: 'white',
                        fontWeight: 'bold'
                      }}
                    >
                      ⏸️ Pause
                    </button>
                  ) : (
                    <button
                      onClick={resumeTraining}
                      className="btn success"
                      style={{
                        flex: 1,
                        padding: '12px',
                        borderRadius: '8px',
                        border: 'none',
                        background: 'linear-gradient(145deg, #28a745, #20c997)',
                        color: 'white',
                        fontWeight: 'bold'
                      }}
                    >
                      ▶️ Resume
                    </button>
                  )}
                  <button
                    onClick={stopTraining}
                    className="btn danger"
                    style={{
                      flex: 1,
                      padding: '12px',
                      borderRadius: '8px',
                      border: 'none',
                      background: 'linear-gradient(145deg, #dc3545, #c82333)',
                      color: 'white',
                      fontWeight: 'bold'
                    }}
                  >
                    🛑 Stop
                  </button>
                </>
              )}
            </div>
          </div>

          <div className="control-group">
            <label style={{ color: '#FFD700', fontWeight: 'bold', marginBottom: '8px', display: 'block' }}>
              Model Management
            </label>
            <div style={{ display: 'flex', gap: '10px' }}>
              <button
                onClick={loadModel}
                disabled={!agent || isTraining || isPlaying}
                className="btn load"
                style={{
                  flex: 1,
                  padding: '12px',
                  borderRadius: '8px',
                  border: 'none',
                  background: 'linear-gradient(145deg, #fd7e14, #e8590c)',
                  color: 'white',
                  fontWeight: 'bold'
                }}
              >
                📁 Load Model
              </button>
              <button
                onClick={downloadModel}
                disabled={!agent || isTraining || isPlaying}
                className="btn download"
                style={{
                  flex: 1,
                  padding: '12px',
                  borderRadius: '8px',
                  border: 'none',
                  background: 'linear-gradient(145deg, #17a2b8, #138496)',
                  color: 'white',
                  fontWeight: 'bold'
                }}
              >
                🔽 Download Model
              </button>
            </div>
          </div>

          <div className="control-group">
            <label style={{ color: '#FFD700', fontWeight: 'bold', marginBottom: '8px', display: 'block' }}>
              Reset Training
            </label>
            <button
              onClick={resetTraining}
              disabled={isTraining || isPlaying}
              className="btn reset"
              style={{
                width: '100%',
                padding: '12px',
                borderRadius: '8px',
                border: 'none',
                background: 'linear-gradient(145deg, #6c757d, #5a6268)',
                color: 'white',
                fontWeight: 'bold'
              }}
            >
              🔄 Reset All Stats
            </button>
            <div style={{ fontSize: '12px', color: '#D2B48C', marginTop: '5px', textAlign: 'center' }}>
              Clear all progress and start fresh
            </div>
          </div>
        </div>

        {/* Training Speed Control - Full Width */}
        <div className="speed-control" style={{
          background: 'rgba(139, 69, 19, 0.3)',
          padding: '20px',
          borderRadius: '12px',
          border: '2px solid #8B4513',
          marginBottom: '20px'
        }}>
          <div style={{ 
            display: 'flex', 
            alignItems: 'center', 
            gap: '20px',
            maxWidth: '600px',
            margin: '0 auto'
          }}>
            <label style={{ 
              color: '#FFD700', 
              fontWeight: 'bold', 
              minWidth: '140px',
              fontSize: '16px'
            }}>
              ⚡ Training Speed:
            </label>
            <div style={{ flex: 1, display: 'flex', alignItems: 'center', gap: '15px' }}>
              <span style={{ color: '#D2B48C', fontSize: '14px', minWidth: '30px' }}>
                10%
              </span>
              <input
                type="range"
                min="10"
                max="100"
                value={trainingSpeed}
                onChange={(e) => setTrainingSpeed(parseInt(e.target.value))}
                disabled={isTraining}
                style={{
                  flex: 1,
                  height: '8px',
                  borderRadius: '4px',
                  background: 'rgba(139, 69, 19, 0.5)',
                  outline: 'none',
                  accentColor: '#FFD700',
                  cursor: isTraining ? 'not-allowed' : 'pointer'
                }}
              />
              <span style={{ color: '#D2B48C', fontSize: '14px', minWidth: '40px' }}>
                100%
              </span>
              <div style={{ 
                background: 'rgba(255, 215, 0, 0.2)', 
                padding: '6px 12px', 
                borderRadius: '6px',
                border: '1px solid #FFD700',
                minWidth: '60px',
                textAlign: 'center'
              }}>
                <span style={{ color: '#FFD700', fontWeight: 'bold', fontSize: '16px' }}>
                  {trainingSpeed}%
                </span>
              </div>
            </div>
          </div>
          <div style={{ 
            fontSize: '12px', 
            color: '#D2B48C', 
            marginTop: '10px', 
            textAlign: 'center',
            maxWidth: '600px',
            margin: '10px auto 0'
          }}>
            Adjust how fast the AI trains (higher = faster training, lower = more detailed learning)
          </div>
        </div>

        {/* Training Progress */}
        {(isTraining || currentEpisode > 0) && (
          <div className="training-progress" style={{
            background: 'rgba(139, 69, 19, 0.3)',
            padding: '15px',
            borderRadius: '8px',
            border: '1px solid #8B4513',
            marginBottom: '20px'
          }}>
            <div className="progress-header" style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              marginBottom: '10px'
            }}>
              <h4 style={{ color: '#FFD700', margin: 0 }}>Training Progress</h4>
            </div>
            
            <div className="progress-bar" style={{
              width: '100%',
              height: '20px',
              background: 'rgba(139, 69, 19, 0.5)',
              borderRadius: '10px',
              overflow: 'hidden',
              marginBottom: '10px'
            }}>
              <div 
                className="progress-fill"
                style={{ 
                  width: `${(currentEpisode / 500) * 100}%`,
                  height: '100%',
                  background: 'linear-gradient(90deg, #FFD700, #FFA500)',
                  transition: 'width 0.3s ease'
                }}
              />
            </div>
            <div className="progress-text" style={{ color: '#D2B48C', fontSize: '14px', textAlign: 'center' }}>
              Episode {currentEpisode} / 500 ({((currentEpisode / 500) * 100).toFixed(1)}%)
            </div>
          </div>
        )}

        {/* Status Indicators */}
        {(isTraining || isPlaying) && (
          <div className="status-indicators" style={{
            display: 'flex',
            gap: '15px',
            marginBottom: '20px'
          }}>
            {isTraining && (
              <div className="status-card" style={{
                background: 'rgba(40, 167, 69, 0.2)',
                padding: '12px',
                borderRadius: '8px',
                border: '1px solid #28a745',
                flex: 1
              }}>
                <div style={{ color: '#28a745', fontWeight: 'bold', marginBottom: '5px' }}>
                  🚀 Training Active
                </div>
                <div style={{ color: '#D2B48C', fontSize: '12px' }}>
                  {algorithmConfigs[selectedAlgorithm].name} is learning...
                </div>
              </div>
            )}
            
            {isPlaying && (
              <div className="status-card" style={{
                background: 'rgba(23, 162, 184, 0.2)',
                padding: '12px',
                borderRadius: '8px',
                border: '1px solid #17a2b8',
                flex: 1
              }}>
                <div style={{ color: '#17a2b8', fontWeight: 'bold', marginBottom: '5px' }}>
                  🎮 AI Playing
                </div>
                <div style={{ color: '#D2B48C', fontSize: '12px' }}>
                  Watch the AI make moves in real-time
                </div>
              </div>
            )}
          </div>
        )}

        {/* Game Visualization */}
        <DndProvider backend={HTML5Backend}>
          <div className="game-visualization" style={{
            display: 'grid',
            gridTemplateColumns: '2fr 1fr',
            gap: '20px',
            marginBottom: '20px'
          }}>
            {/* Game Board */}
            <div className="game-area" style={{
              background: 'rgba(139, 69, 19, 0.3)',
              padding: '20px',
              borderRadius: '12px',
              border: '2px solid #8B4513'
            }}>
              <h4 style={{ color: '#FFD700', marginBottom: '15px', textAlign: 'center' }}>
                🎯 Game Board
              </h4>
              
              <GameBoard
                grid={grid}
                onBlockPlace={() => {}} // Disabled for AI training
                availableBlocks={availableBlocks}
                isPaused={false}
                difficulty={difficulty}
              />
            </div>

            {/* Game Info */}
            <div className="game-info" style={{
              display: 'flex',
              flexDirection: 'column',
              gap: '15px'
            }}>
              {/* Score Display */}
              <div style={{
                background: 'rgba(139, 69, 19, 0.3)',
                padding: '15px',
                borderRadius: '8px',
                border: '1px solid #8B4513'
              }}>
                <ScoreDisplay 
                  score={score} 
                  bestScore={bestScore} 
                  linesCleared={Math.floor(score / 100)}
                  difficulty={difficulty}
                />
              </div>

              {/* Available Blocks */}
              <div style={{
                background: 'rgba(139, 69, 19, 0.3)',
                padding: '15px',
                borderRadius: '8px',
                border: '1px solid #8B4513'
              }}>
                <h5 style={{ color: '#FFD700', marginBottom: '10px' }}>Available Blocks</h5>
                <BlockTray blocks={availableBlocks} disabled={true} />
              </div>

              {/* Episode Info */}
              <div style={{
                background: 'rgba(139, 69, 19, 0.3)',
                padding: '15px',
                borderRadius: '8px',
                border: '1px solid #8B4513'
              }}>
                <h5 style={{ color: '#FFD700', marginBottom: '10px' }}>Episode Info</h5>
                <div className="episode-stats">
                  <div className="info-item" style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '5px' }}>
                    <span style={{ color: '#D2B48C' }}>Episode:</span>
                    <span style={{ color: '#FFD700', fontWeight: 'bold' }}>{currentEpisode}</span>
                  </div>
                  <div className="info-item" style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '5px' }}>
                    <span style={{ color: '#D2B48C' }}>Episode Score:</span>
                    <span style={{ color: '#FFD700', fontWeight: 'bold' }}>{episodeScore}</span>
                  </div>
                  <div className="info-item" style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '5px' }}>
                    <span style={{ color: '#D2B48C' }}>Total Score:</span>
                    <span style={{ color: '#FFD700', fontWeight: 'bold' }}>{score}</span>
                  </div>
                  <div className="info-item" style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '5px' }}>
                    <span style={{ color: '#D2B48C' }}>Steps:</span>
                    <span style={{ color: '#FFD700', fontWeight: 'bold' }}>{episodeSteps}</span>
                  </div>
                  <div className="info-item" style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span style={{ color: '#D2B48C' }}>Algorithm:</span>
                    <span style={{ color: '#FFD700', fontWeight: 'bold' }}>{algorithmConfigs[selectedAlgorithm].name}</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </DndProvider>

        {/* AI Visualization */}
        <div className="ai-visualization-section" style={{
          background: 'rgba(139, 69, 19, 0.3)',
          padding: '20px',
          borderRadius: '12px',
          border: '2px solid #8B4513'
        }}>
          <div className="visualization-header" style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            marginBottom: '15px'
          }}>
            <h4 style={{ color: '#FFD700', margin: 0 }}>📊 AI Performance Analytics</h4>
            <button 
              className="toggle-btn"
              onClick={() => setVisualTraining(!visualTraining)}
              style={{
                padding: '8px 12px',
                borderRadius: '6px',
                border: 'none',
                background: visualTraining ? 'linear-gradient(145deg, #28a745, #20c997)' : 'rgba(139, 69, 19, 0.5)',
                color: 'white',
                fontWeight: 'bold',
                fontSize: '12px'
              }}
            >
              {visualTraining ? '👁️ Hide' : '👁️ Show'} Visualization
            </button>
          </div>
          
          {visualTraining && (
            <AIVisualization 
              trainingStats={trainingStats}
              isTraining={isTraining}
              compact={false}
            />
          )}
        </div>

        {/* Test Results */}
        {testResults && (
          <div className="test-results" style={{
            background: 'rgba(139, 69, 19, 0.3)',
            padding: '20px',
            borderRadius: '12px',
            border: '2px solid #8B4513',
            marginTop: '20px'
          }}>
            <h4 style={{ color: '#FFD700', marginBottom: '15px' }}>🧪 Test Results</h4>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '10px' }}>
              {Object.entries(testResults).map(([test, passed]) => (
                <div key={test} style={{
                  padding: '10px',
                  borderRadius: '6px',
                  background: passed ? 'rgba(40, 167, 69, 0.2)' : 'rgba(220, 53, 69, 0.2)',
                  border: `1px solid ${passed ? '#28a745' : '#dc3545'}`
                }}>
                  <div style={{ color: passed ? '#28a745' : '#dc3545', fontWeight: 'bold' }}>
                    {passed ? '✅' : '❌'} {test}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Tips Modal */}
      {showTips && <TipsModal />}
    </div>
  );
}

export default AILearningView; 