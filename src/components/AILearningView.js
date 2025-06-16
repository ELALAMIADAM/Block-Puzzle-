import React, { useState, useEffect, useRef, useCallback } from 'react';
import { DndProvider } from 'react-dnd';
import { HTML5Backend } from 'react-dnd-html5-backend';
import { DQNAgent } from '../ai/DQNAgent';
import { DQNEnvironment } from '../ai/DQNEnvironment';
import { EliteDQNAgent } from '../ai/EliteDQNAgent';
import { EliteEnvironment } from '../ai/EliteEnvironment';
import { AlgorithmSelector } from '../ai/AdvancedAIAgents';
import { runAITests } from '../ai/AITestRunner';
import AIVisualization from './AIVisualization';
import GameBoard from './GameBoard';
import BlockTray from './BlockTray';
import ScoreDisplay from './ScoreDisplay';
import AdvancedAITrainingPanel from './AdvancedAITrainingPanel';
import { generateRandomBlocks, checkGameOver } from '../utils/gameLogic';

function AILearningView({ onNavigate }) {
  // Algorithm Selection
  const [selectedAlgorithm, setSelectedAlgorithm] = useState('dqn');
  
  // Training State
  const [isTraining, setIsTraining] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [visualTraining, setVisualTraining] = useState(true);
  const [trainingSpeed, setTrainingSpeed] = useState(50);
  const [maxEpisodes, setMaxEpisodes] = useState(500);
  
  // Game State
  const [grid, setGrid] = useState(Array(9).fill(null).map(() => Array(9).fill(false)));
  const [availableBlocks, setAvailableBlocks] = useState([]);
  const [score, setScore] = useState(0);
  const [bestScore, setBestScore] = useState(0);
  const [difficulty, setDifficulty] = useState('normal');
  const [gameOver, setGameOver] = useState(false);
  const [linesCleared, setLinesCleared] = useState(0);
  const [totalLinesCleared, setTotalLinesCleared] = useState(0);
  
  // Training Progress
  const [currentEpisode, setCurrentEpisode] = useState(0);
  const [episodeScore, setEpisodeScore] = useState(0);
  const [episodeSteps, setEpisodeSteps] = useState(0);
  const [trainingStats, setTrainingStats] = useState({});
  
  // AI Components
  const [agent, setAgent] = useState(null);
  const [environment, setEnvironment] = useState(null);
  
  // UI State
  const [showTips, setShowTips] = useState(false);
  const [testResults, setTestResults] = useState({});
  
  // Refs for training control
  const trainingRef = useRef(false);
  const episodeRef = useRef(0);
  const stepRef = useRef(0);

  // Algorithm configurations - ALL USE SAME ENVIRONMENT FOR FAIR COMPARISON
  const algorithmConfigs = {
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
  };

  // Initialize agent and environment when algorithm changes
  const initializeAgent = useCallback(() => {
    console.log(`🤖 Initializing ${algorithmConfigs[selectedAlgorithm].name}...`);
    
    const config = algorithmConfigs[selectedAlgorithm];
    const env = new config.environmentClass();
    
    let newAgent;
    if (config.agentClass) {
      // Use specific agent class (DQN variants)
      newAgent = new config.agentClass(env.getStateSize(), env.getMaxActionSpace(), config.options);
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
    setLinesCleared(0);
    setGameOver(false);
    
    console.log(`✅ ${config.name} initialized successfully!`);
  }, [selectedAlgorithm]);

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
      if (episodeRef.current >= maxEpisodes) {
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
    setScore(0);
    setLinesCleared(0);
    setTotalLinesCleared(0);
    setTrainingStats({});
    episodeRef.current = 0;
    stepRef.current = 0;
    
    console.log(`🔄 ${algorithmConfigs[selectedAlgorithm].name} training reset`);
  };

  const startAIPlay = async () => {
    if (!agent || !environment) return;
    
    setIsPlaying(true);
    console.log(`🎮 Starting ${algorithmConfigs[selectedAlgorithm].name} gameplay...`);
    
    // Reset environment to current game state
    environment.setState(grid, availableBlocks, score, difficulty);
    
    while (isPlaying && !gameOver) {
      await makeAIMove();
      await new Promise(resolve => setTimeout(resolve, 1000)); // 1 second between moves
    }
    
    setIsPlaying(false);
  };

  const stopAIPlay = () => {
    setIsPlaying(false);
    console.log(`⏹️ ${algorithmConfigs[selectedAlgorithm].name} gameplay stopped`);
  };

  const makeAIMove = async () => {
    if (!agent || !environment || !isPlaying) return;
    
    try {
      const validActions = environment.getValidActions();
      
      if (validActions.length === 0) {
        console.log('🎮 No valid actions available, stopping AI play');
        stopAIPlay();
        return;
      }
      
      // Get state using DQNEnvironment method
      let state = environment.getState(); // DQNEnvironment always has getState() method
      
      let action;
      
      // Algorithm-specific action selection for gameplay
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
        case 'elite-dqn':
          action = await agent.act(state, validActions, environment);
          break;
        case 'dqn':
        default:
          action = await agent.act(state, validActions, environment);
          break;
      }
      
      if (action === null || action === undefined) {
        console.warn(`⚠️ ${selectedAlgorithm} returned null action during play`);
        action = validActions[Math.floor(Math.random() * validActions.length)];
      }
      
      const stepResult = environment.step(action);
      
      // CRITICAL: Use REAL game score, not AI rewards
      const realGameScore = environment.score;
      
      // Update visual state with REAL game performance
      setGrid(environment.grid.map(row => [...row]));
      setAvailableBlocks(environment.availableBlocks.map(block => block.map(row => [...row])));
      setScore(realGameScore); // Use real game score
      setLinesCleared(environment.lineClearsThisEpisode || 0); // Track actual lines cleared
      setBestScore(Math.max(bestScore, realGameScore)); // Use real game score
      
      // Check if game is over
      if (stepResult.done || environment.gameOver) {
        console.log(`🎮 Game Over! REAL Final Score: ${realGameScore} (Algorithm: ${selectedAlgorithm})`);
        stopAIPlay();
        
        // Update agent stats with REAL game score
        if (agent.endEpisode && typeof agent.endEpisode === 'function') {
          switch (selectedAlgorithm) {
            case 'policy-gradient':
              await agent.endEpisode(realGameScore); // Use real game score
              break;
            case 'mcts':
            case 'heuristic':
              agent.endEpisode(realGameScore); // Use real game score
              break;
            case 'elite-dqn':
            case 'dqn':
            default:
              agent.endEpisode(stepResult.reward, realGameScore, environment); // Reward for AI, real score for comparison
              break;
          }
        }
        
        // Update training stats
        if (agent.getStats && typeof agent.getStats === 'function') {
          const stats = agent.getStats();
          stats.lastGameScore = realGameScore; // Ensure real score is tracked
          setTrainingStats(stats);
        }
      }
      
      // Add new blocks when needed - use DQNEnvironment's curriculum blocks
      if (environment.availableBlocks.length === 0 && !stepResult.done) {
        environment.availableBlocks = environment.generateCurriculumBlocks(); // DQNEnvironment method
        setAvailableBlocks(environment.availableBlocks.map(block => block.map(row => [...row])));
        
        if (environment.checkGameOver()) {
          console.log('🎮 No more moves possible, stopping AI play');
          stopAIPlay();
        }
      }
      
      // Clean up state tensor
      if (state && state.dispose && typeof state.dispose === 'function') {
        state.dispose();
      }
      
    } catch (error) {
      console.error(`❌ AI move error for ${selectedAlgorithm}:`, error);
      stopAIPlay();
    }
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
          case 'elite-dqn':
            action = await agent.act(state, validActions, environment);
            break;
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
        
        // Update state for both visual and non-visual training
        if (visualTraining) {
          setGrid(environment.grid.map(row => [...row]));
          setAvailableBlocks(environment.availableBlocks.map(block => block.map(row => [...row])));
          setEpisodeSteps(stepCount + 1);
          stepRef.current = stepCount + 1;
          
          // Visual delay for training observation and UI responsiveness
          if (stepCount % 3 === 0) {
            await new Promise(resolve => setTimeout(resolve, Math.max(10, 100 - trainingSpeed)));
          }
        } else {
          // Without visual training, much faster execution with minimal yielding
          if (stepCount % 20 === 0) {
            await new Promise(resolve => setTimeout(resolve, 1)); // Very minimal delay
          }
        }
        
        // Always update score and lines cleared for both modes
        setScore(actualGameScore); // Update main score display
        setLinesCleared(environment.lineClearsThisEpisode || 0); // Update lines cleared
        setEpisodeScore(actualGameScore); // Show real game score
        
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
        
        // Add new blocks when needed - use DQNEnvironment's curriculum blocks
        if (environment.availableBlocks.length === 0 && !done) {
          environment.availableBlocks = environment.generateCurriculumBlocks(); // DQNEnvironment method
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
        
        // Update total lines cleared
        setTotalLinesCleared(prev => prev + (environment.lineClearsThisEpisode || 0));
        
        // Log progress every 10 episodes - show REAL performance
        if (episodeRef.current % 10 === 0) {
          console.log(`📊 Episode ${episodeRef.current}/${maxEpisodes}: REAL Score=${finalGameScore}, AI Reward=${totalReward.toFixed(1)}, Best=${Math.max(bestScore, finalGameScore)}, Lines=${environment.lineClearsThisEpisode || 0}, Algorithm=${selectedAlgorithm}`);
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

  const saveModel = async () => {
    if (!agent) {
      alert('No agent available to save!');
      return;
    }
    
    try {
      const algorithmName = algorithmConfigs[selectedAlgorithm].name;
      const modelName = `${selectedAlgorithm}-model-${Date.now()}`;
      
      console.log(`💾 Saving ${algorithmName} model...`);
      
      let success = false;
      
      switch (selectedAlgorithm) {
        case 'mcts':
        case 'heuristic':
          // For non-neural algorithms, save stats and configuration
          const agentData = {
            algorithm: selectedAlgorithm,
            stats: agent.getStats(),
            timestamp: new Date().toISOString(),
            version: '1.0'
          };
          
          localStorage.setItem(modelName, JSON.stringify(agentData));
          success = true;
          console.log(`✅ ${algorithmName} configuration saved to localStorage`);
          break;
          
        case 'policy-gradient':
          // Save policy gradient model if it has neural networks
          if (agent.saveModel && typeof agent.saveModel === 'function') {
            success = await agent.saveModel(modelName);
          } else {
            // Fallback to stats saving
            const pgData = {
              algorithm: selectedAlgorithm,
              stats: agent.getStats(),
              timestamp: new Date().toISOString(),
              version: '1.0'
            };
            localStorage.setItem(modelName, JSON.stringify(pgData));
            success = true;
          }
          break;
          
        case 'elite-dqn':
        case 'dqn':
        default:
          // Save neural network models
          if (agent.saveModel && typeof agent.saveModel === 'function') {
            success = await agent.saveModel(modelName);
          } else {
            throw new Error('Agent does not support model saving');
          }
          break;
      }
      
      if (success) {
        alert(`✅ ${algorithmName} model saved successfully!\n\nModel: ${modelName}\nAlgorithm: ${algorithmName}\nBest Score: ${agent.getStats().bestScore || 0}`);
      } else {
        throw new Error('Model saving failed');
      }
      
    } catch (error) {
      console.error('❌ Model save error:', error);
      alert(`❌ Failed to save ${algorithmConfigs[selectedAlgorithm].name} model: ${error.message}`);
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
            const pgKeys = Object.keys(localStorage).filter(key => 
              key.includes('policy_gradient') && key.includes('model')
            );
            
            if (pgKeys.length > 0) {
              const latestPgKey = pgKeys.sort().pop();
              success = await agent.loadModel(latestPgKey);
            } else {
              throw new Error('No saved Policy Gradient models found');
            }
          } else {
            throw new Error('Policy Gradient agent does not support model loading');
          }
          break;
          
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

  const runTests = async () => {
    console.log('🧪 Running comprehensive AI algorithm tests...');
    
    if (!agent || !environment) {
      alert('No trained agent available to test!');
      return;
    }
    
    try {
      // Run the proper algorithm test
      const testResults = await testAlgorithm();
      
      // Also run the system integration tests
      const systemResults = await runAITests();
      const allSystemTestsPassed = Object.values(systemResults).every(result => result);
      
      const combinedMessage = `🧪 COMPREHENSIVE TEST RESULTS:

🎯 ALGORITHM PERFORMANCE TEST:
${testResults.algorithmName}
• Average Score: ${testResults.averageScore.toFixed(1)}
• Best Score: ${testResults.bestScore}
• Games Completed: ${testResults.gamesCompleted}/10

🔧 SYSTEM INTEGRATION TESTS:
${allSystemTestsPassed ? '✅ All system tests passed!' : '❌ Some system tests failed - check console'}

📊 OVERALL ASSESSMENT:
${testResults.averageScore >= 200 && allSystemTestsPassed ? 
  '🏆 EXCELLENT - Algorithm is well-trained and system is stable' :
  testResults.averageScore >= 100 && allSystemTestsPassed ?
  '🥈 GOOD - Algorithm shows promise, system is stable' :
  allSystemTestsPassed ?
  '📈 FAIR - System works but algorithm needs more training' :
  '⚠️ ISSUES DETECTED - Check console for details'
}`;

      console.log(combinedMessage);
      alert(combinedMessage);
      
    } catch (error) {
      console.error('❌ Test execution failed:', error);
      alert(`❌ Test execution failed: ${error.message}`);
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

  // NEW: Proper testing function for fair algorithm comparison
  const testAlgorithm = async () => {
    if (!agent || !environment) {
      alert('No trained agent available to test!');
      return;
    }

    console.log(`🧪 Testing ${algorithmConfigs[selectedAlgorithm].name}...`);
    
    const testResults = {
      algorithm: selectedAlgorithm,
      algorithmName: algorithmConfigs[selectedAlgorithm].name,
      scores: [],
      averageScore: 0,
      bestScore: 0,
      worstScore: Infinity,
      gamesCompleted: 0,
      totalMoves: 0
    };

    const numTestGames = 10; // Test with 10 games for fair comparison
    
    for (let gameNum = 0; gameNum < numTestGames; gameNum++) {
      console.log(`🎯 Test Game ${gameNum + 1}/${numTestGames} for ${algorithmConfigs[selectedAlgorithm].name}`);
      
      // Reset environment for each test game
      environment.reset();
      
      let gameScore = 0;
      let moves = 0;
      let done = false;
      const maxMoves = 100; // Prevent infinite games
      
      // Get initial state
      let state = environment.getState(); // DQNEnvironment always has getState() method
      
      while (!done && moves < maxMoves) {
        const validActions = environment.getValidActions();
        
        if (validActions.length === 0) {
          done = true;
          break;
        }
        
        let action;
        
        // Use trained agent to select action (NO EXPLORATION - pure exploitation)
        try {
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
            case 'elite-dqn':
              // Force exploitation mode for testing
              const oldEpsilon = agent.epsilon;
              agent.epsilon = 0; // No exploration during testing
              action = await agent.act(state, validActions, environment);
              agent.epsilon = oldEpsilon; // Restore original epsilon
              break;
            case 'dqn':
            default:
              // Force exploitation mode for testing
              const oldEps = agent.epsilon;
              agent.epsilon = 0; // No exploration during testing
              action = await agent.act(state, validActions, environment);
              agent.epsilon = oldEps; // Restore original epsilon
              break;
          }
        } catch (error) {
          console.warn(`⚠️ Error during testing, using random action:`, error);
          action = validActions[Math.floor(Math.random() * validActions.length)];
        }
        
        if (action === null || action === undefined) {
          action = validActions[Math.floor(Math.random() * validActions.length)];
        }
        
        const stepResult = environment.step(action);
        gameScore = environment.score; // Use REAL game score
        state = stepResult.state;
        done = stepResult.done;
        moves++;
        
        // Add new blocks when needed - use DQNEnvironment's curriculum blocks
        if (environment.availableBlocks.length === 0 && !done) {
          environment.availableBlocks = environment.generateCurriculumBlocks(); // DQNEnvironment method
          done = environment.checkGameOver();
        }
        
        // Clean up state tensor
        if (state && state.dispose && typeof state.dispose === 'function') {
          state.dispose();
        }
      }
      
      // Record REAL game results
      testResults.scores.push(gameScore);
      testResults.totalMoves += moves;
      testResults.gamesCompleted++;
      testResults.bestScore = Math.max(testResults.bestScore, gameScore);
      testResults.worstScore = Math.min(testResults.worstScore, gameScore);
      
      console.log(`✅ Test Game ${gameNum + 1} completed: Score=${gameScore}, Moves=${moves}`);
    }
    
    // Calculate final statistics
    testResults.averageScore = testResults.scores.reduce((a, b) => a + b, 0) / testResults.scores.length;
    testResults.averageMoves = testResults.totalMoves / testResults.gamesCompleted;
    
    // Display results
    const resultMessage = `🧪 TEST RESULTS for ${testResults.algorithmName}:

📊 Performance Summary:
• Games Played: ${testResults.gamesCompleted}
• Average Score: ${testResults.averageScore.toFixed(1)}
• Best Score: ${testResults.bestScore}
• Worst Score: ${testResults.worstScore}
• Average Moves: ${testResults.averageMoves.toFixed(1)}

📈 Score Distribution:
${testResults.scores.map((score, i) => `Game ${i + 1}: ${score}`).join('\n')}

🎯 Performance Rating:
${testResults.averageScore >= 500 ? '🏆 Excellent' : 
  testResults.averageScore >= 200 ? '🥈 Good' : 
  testResults.averageScore >= 100 ? '🥉 Fair' : '📈 Needs Training'}`;

    console.log(resultMessage);
    alert(resultMessage);
    
    // Store test results for comparison
    setTestResults(prev => ({
      ...prev,
      [selectedAlgorithm]: testResults
    }));
    
    return testResults;
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
              <li><strong>Pause/Resume:</strong> Temporarily halt training and continue later</li>
              <li><strong>Stop:</strong> End training session completely</li>
              <li><strong>Reset:</strong> Clear AI memory and start fresh training</li>
              <li><strong>Speed Control:</strong> Adjust how fast the AI plays during training</li>
            </ul>
          </div>
          
          <div className="tip-section">
            <h4>🧪 Testing & Models</h4>
            <ul>
              <li><strong>Test Model:</strong> See how well your trained AI performs</li>
              <li><strong>Download Model:</strong> Save your trained AI for later use</li>
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
            
            <button
              onClick={runTests}
              disabled={isTraining || isPlaying}
              style={{
                padding: '12px 20px',
                borderRadius: '8px',
                border: 'none',
                background: 'linear-gradient(145deg, #28a745, #20c997)',
                color: 'white',
                fontWeight: 'bold',
                cursor: 'pointer'
              }}
            >
              🧪 Run Tests
            </button>
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
          gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
          gap: '15px',
          marginBottom: '20px'
        }}>
          <div className="control-group">
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
            <div style={{ display: 'flex', gap: '10px' }}>
              {!isPlaying ? (
                <button
                  onClick={startAIPlay}
                  disabled={isTraining || !agent}
                  className="btn info"
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
                  🎮 Start AI Play
                </button>
              ) : (
                <button
                  onClick={stopAIPlay}
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
                  ⏹️ Stop AI Play
                </button>
              )}
            </div>
          </div>

          <div className="control-group">
            <div style={{ display: 'flex', gap: '10px', marginBottom: '10px' }}>
              <button
                onClick={saveModel}
                disabled={!agent || isTraining || isPlaying}
                className="btn save"
                style={{
                  flex: 1,
                  padding: '12px',
                  borderRadius: '8px',
                  border: 'none',
                  background: 'linear-gradient(145deg, #6f42c1, #5a32a3)',
                  color: 'white',
                  fontWeight: 'bold'
                }}
              >
                💾 Save
              </button>
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
                📁 Load
              </button>
            </div>
            <div style={{ display: 'flex', gap: '10px' }}>
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
                🔽 Download
              </button>
              <button
                onClick={runTests}
                disabled={isTraining || isPlaying}
                className="btn test"
                style={{
                  flex: 1,
                  padding: '12px',
                  borderRadius: '8px',
                  border: 'none',
                  background: 'linear-gradient(145deg, #28a745, #1e7e34)',
                  color: 'white',
                  fontWeight: 'bold'
                }}
              >
                🧪 Test
              </button>
            </div>
          </div>

          <div className="control-group">
            <div style={{ display: 'flex', gap: '10px' }}>
              <button
                onClick={testAlgorithm}
                disabled={!agent || isTraining || isPlaying}
                className="btn test-performance"
                style={{
                  flex: 1,
                  padding: '12px',
                  borderRadius: '8px',
                  border: 'none',
                  background: 'linear-gradient(145deg, #e74c3c, #c0392b)',
                  color: 'white',
                  fontWeight: 'bold'
                }}
              >
                🎯 Test Performance
              </button>
            </div>
          </div>

          <div className="control-group">
            <label style={{ color: '#FFD700', fontWeight: 'bold', marginBottom: '8px', display: 'block' }}>
              Max Episodes: {maxEpisodes}
            </label>
            <input
              type="number"
              min="10"
              max="10000"
              value={maxEpisodes}
              onChange={(e) => setMaxEpisodes(parseInt(e.target.value))}
              disabled={isTraining}
              style={{
                width: '100%',
                padding: '8px',
                borderRadius: '6px',
                border: '2px solid #8B4513',
                background: 'rgba(255, 215, 0, 0.1)',
                color: '#FFD700',
                fontSize: '14px',
                fontWeight: 'bold',
                marginBottom: '8px'
              }}
            />
            <div style={{ display: 'flex', gap: '5px', flexWrap: 'wrap' }}>
              {[100, 500, 1000, 2000, 5000].map(episodes => (
                <button
                  key={episodes}
                  onClick={() => setMaxEpisodes(episodes)}
                  disabled={isTraining}
                  style={{
                    padding: '4px 8px',
                    borderRadius: '4px',
                    border: 'none',
                    background: maxEpisodes === episodes 
                      ? 'linear-gradient(145deg, #FFD700, #FFA500)' 
                      : 'rgba(139, 69, 19, 0.5)',
                    color: maxEpisodes === episodes ? '#2C1810' : '#D2B48C',
                    fontSize: '10px',
                    fontWeight: 'bold',
                    cursor: 'pointer'
                  }}
                >
                  {episodes}
                </button>
              ))}
            </div>
          </div>

          <div className="control-group">
            <label style={{ color: '#FFD700', fontWeight: 'bold', marginBottom: '8px', display: 'block' }}>
              Training Speed: {trainingSpeed}%
            </label>
            <input
              type="range"
              min="10"
              max="100"
              value={trainingSpeed}
              onChange={(e) => setTrainingSpeed(parseInt(e.target.value))}
              disabled={isTraining}
              style={{
                width: '100%',
                accentColor: '#FFD700'
              }}
            />
          </div>

          <div className="control-group">
            <label style={{ color: '#FFD700', fontWeight: 'bold', marginBottom: '8px', display: 'block' }}>
              Training Options
            </label>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <label style={{ color: '#D2B48C', fontSize: '14px', display: 'flex', alignItems: 'center', gap: '5px' }}>
                <input
                  type="checkbox"
                  checked={visualTraining}
                  onChange={(e) => setVisualTraining(e.target.checked)}
                  disabled={isTraining}
                  style={{ accentColor: '#FFD700' }}
                />
                Show Game Board
              </label>
            </div>
          </div>
        </div>

        {/* Training Progress */}
        

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

        {/* Game Visualization - Only show when visualTraining is enabled */}
        {visualTraining && (
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
                    linesCleared={linesCleared}
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
                      <span style={{ color: '#D2B48C' }}>Score:</span>
                      <span style={{ color: '#FFD700', fontWeight: 'bold' }}>{score}</span>
                    </div>
                    <div className="info-item" style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '5px' }}>
                      <span style={{ color: '#D2B48C' }}>Lines Cleared:</span>
                      <span style={{ color: '#FFD700', fontWeight: 'bold' }}>{linesCleared}</span>
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
        )}

        {/* Quick Stats when visualization is hidden */}
        {!visualTraining && (isTraining || isPlaying) && (
          <div style={{
            background: 'rgba(139, 69, 19, 0.3)',
            padding: '20px',
            borderRadius: '12px',
            border: '2px solid #8B4513',
            marginBottom: '20px'
          }}>
            <h4 style={{ color: '#FFD700', marginBottom: '15px', textAlign: 'center' }}>
              🚀 Training in Progress (Fast Mode) - {currentEpisode}/{maxEpisodes}
            </h4>
            
            <div style={{ 
              display: 'grid', 
              gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', 
              gap: '15px',
              textAlign: 'center'
            }}>
              <div style={{
                background: 'rgba(255, 215, 0, 0.1)',
                padding: '15px',
                borderRadius: '8px',
                border: '1px solid #FFD700'
              }}>
                <div style={{ color: '#FFD700', fontWeight: 'bold', fontSize: '18px' }}>
                  {currentEpisode}/{maxEpisodes}
                </div>
                <div style={{ color: '#D2B48C', fontSize: '12px' }}>
                  Episode Progress ({((currentEpisode / maxEpisodes) * 100).toFixed(1)}%)
                </div>
              </div>
              
              <div style={{
                background: 'rgba(255, 215, 0, 0.1)',
                padding: '15px',
                borderRadius: '8px',
                border: '1px solid #FFD700'
              }}>
                <div style={{ color: '#28a745', fontWeight: 'bold', fontSize: '18px' }}>
                  {score}
                </div>
                <div style={{ color: '#D2B48C', fontSize: '12px' }}>
                  Current Score
                </div>
              </div>
              
                             <div style={{
                 background: 'rgba(255, 215, 0, 0.1)',
                 padding: '15px',
                 borderRadius: '8px',
                 border: '1px solid #FFD700'
               }}>
                 <div style={{ color: '#17a2b8', fontWeight: 'bold', fontSize: '18px' }}>
                   {linesCleared}
                 </div>
                 <div style={{ color: '#D2B48C', fontSize: '12px' }}>
                   Lines This Episode
                 </div>
               </div>
               
               <div style={{
                 background: 'rgba(255, 215, 0, 0.1)',
                 padding: '15px',
                 borderRadius: '8px',
                 border: '1px solid #FFD700'
               }}>
                 <div style={{ color: '#6f42c1', fontWeight: 'bold', fontSize: '18px' }}>
                   {totalLinesCleared}
                 </div>
                 <div style={{ color: '#D2B48C', fontSize: '12px' }}>
                   Total Lines Cleared
                 </div>
               </div>
              
              <div style={{
                background: 'rgba(255, 215, 0, 0.1)',
                padding: '15px',
                borderRadius: '8px',
                border: '1px solid #FFD700'
              }}>
                <div style={{ color: '#dc3545', fontWeight: 'bold', fontSize: '18px' }}>
                  {bestScore}
                </div>
                <div style={{ color: '#D2B48C', fontSize: '12px' }}>
                  Best Score
                </div>
              </div>
            </div>
            
            <div style={{ 
              marginTop: '15px', 
              textAlign: 'center', 
              color: '#D2B48C', 
              fontSize: '14px' 
            }}>
              💡 Enable "Show Game Board" to see live visualization (slower training)
            </div>
          </div>
        )}

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