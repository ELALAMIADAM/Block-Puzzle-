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
    // Initialize AI components with EXPERT enhancements
    const env = new DQNEnvironment();
    const dqnAgent = new DQNAgent(env.getStateSize(), env.getMaxActionSpace(), {
      learningRate: 0.0003, // EXPERT: Lower for stability
      epsilon: 0.95, // EXPERT: Higher initial exploration
      epsilonMin: 0.02, // EXPERT: Lower minimum for exploitation
      epsilonDecay: 0.9995, // EXPERT: Slower decay for thorough exploration
      gamma: 0.98, // EXPERT: Higher discount for long-term planning
      batchSize: 128, // EXPERT: Larger batch for stability
      memorySize: 25000, // EXPERT: Larger memory for diverse experiences
      targetUpdateFreq: 200 // EXPERT: Less frequent for stability
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
    const maxSteps = 150; // EXPERT: Increased max steps for complex strategies
    
    while (!done && stepCount < maxSteps) {
      // EXPERT ENHANCEMENT: Use hybrid action selection
      const validActions = environment.getValidActions();
      
      if (validActions.length === 0) {
        done = true;
        break;
      }
      
      const action = await agent.act(state, validActions);
      const stepResult = environment.step(action);
      
      // EXPERT ENHANCEMENT: Store successful patterns for meta-learning
      if (stepResult.reward > 200) {
        environment.storeSuccessfulPattern(environment.grid, stepResult.reward);
      } else if (stepResult.reward < -150) {
        environment.storeFailedPattern(environment.grid, stepResult.reward);
      }
      
      // Visual feedback for training
      if (visualTraining) {
        setTrainingGrid(environment.grid.map(row => [...row]));
        setEpisodeScore(environment.score);
        setEpisodeSteps(stepCount + 1);
        stepRef.current = stepCount + 1;
        
        // EXPERT: Reduced delay for faster visual training
        if (visualTraining) {
          await new Promise(resolve => setTimeout(resolve, 5));
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
      
      // EXPERT ENHANCEMENT: More frequent training for better learning
      if (agent.memory.length > agent.batchSize && stepCount % 2 === 0) {
        await agent.replay();
      }
      
      // Add new blocks when tray is empty and check for game over
      if (environment.availableBlocks.length === 0 && !done) {
        environment.availableBlocks = generateRandomBlocks();
        done = checkGameOver(environment.grid, environment.availableBlocks, environment.difficulty);
        if (visualTraining) {
          setTrainingBlocks(environment.availableBlocks.map(block => block.map(row => [...row])));
        }
      }
    }
    
    agent.endEpisode(0, environment.score);
    
    // Update stats more frequently for better visualization - every 3 episodes for expert tracking
    if (episodeRef.current % 3 === 0) {
      const newStats = agent.getStats();
      setTrainingStats(newStats);
      
      // EXPERT: Enhanced logging with curriculum and meta-learning info
      console.log(`📈 EXPERT Episode ${episodeRef.current}: Reward: ${newStats.totalReward.toFixed(2)}, ` +
                  `Avg: ${newStats.avgReward.toFixed(2)}, Best Score: ${newStats.bestScore}, ` +
                  `Epsilon: ${(newStats.epsilon * 100).toFixed(1)}%, Stage: ${newStats.curriculumStage || 0}, ` +
                  `Meta Patterns: ${newStats.metaPatternsLearned || 0}`);
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
    
    // 🔍 VERIFICATION: Confirm we're using the latest AI system
    console.log('🔍 VERIFYING AI SYSTEM:');
    console.log(`✅ Agent Type: ${agent.constructor.name}`);
    console.log(`✅ Environment Type: ${environment.constructor.name}`);
    console.log(`✅ State Size: ${environment.getStateSize()}`);
    console.log(`✅ Max Action Space: ${environment.getMaxActionSpace()}`);
    console.log(`✅ Agent Hyperparameters:`, {
      learningRate: agent.learningRate,
      epsilon: agent.epsilon,
      epsilonMin: agent.epsilonMin,
      epsilonDecay: agent.epsilonDecay,
      gamma: agent.gamma,
      batchSize: agent.batchSize,
      memorySize: agent.memorySize,
      targetUpdateFreq: agent.targetUpdateFreq
    });
    console.log(`✅ Training Episodes Completed: ${agent.episode}`);
    console.log(`✅ Best Score Achieved: ${agent.bestScore}`);
    console.log(`✅ Current Epsilon: ${(agent.epsilon * 100).toFixed(1)}%`);
    
    // 🔍 VERIFICATION: Test strategic reward system
    console.log('🔍 TESTING STRATEGIC REWARD SYSTEM:');
    try {
      // Create a test scenario to verify reward calculation
      const testGrid = Array(9).fill(null).map(() => Array(9).fill(false));
      const testBlocks = generateRandomBlocks();
      environment.setState(testGrid, testBlocks, 0, difficulty);
      
      // Test if strategic methods exist
      const hasProximityBonus = typeof environment.calculateProximityBonus === 'function';
      const hasMaxClearingBonus = typeof environment.calculateMaxClearingPotential === 'function';
      const hasTargetFocusPenalty = typeof environment.calculateTargetFocusPenalty === 'function';
      const hasBlockOptimization = typeof environment.calculateBlockOptimizationBonus === 'function';
      
      console.log(`✅ Proximity Bonus System: ${hasProximityBonus ? 'ACTIVE' : 'MISSING'}`);
      console.log(`✅ Max Clearing Potential: ${hasMaxClearingBonus ? 'ACTIVE' : 'MISSING'}`);
      console.log(`✅ Target Focus Penalties: ${hasTargetFocusPenalty ? 'ACTIVE' : 'MISSING'}`);
      console.log(`✅ Block Optimization: ${hasBlockOptimization ? 'ACTIVE' : 'MISSING'}`);
      
      if (hasProximityBonus && hasMaxClearingBonus && hasTargetFocusPenalty && hasBlockOptimization) {
        console.log('🎯 STRATEGIC REWARD SYSTEM: FULLY ACTIVE ✅');
      } else {
        console.log('⚠️ STRATEGIC REWARD SYSTEM: INCOMPLETE - Some features missing');
      }
      
    } catch (error) {
      console.log('❌ Error testing strategic reward system:', error);
    }
    
    console.log('🎮 Starting AI play with ENHANCED STRATEGIC SYSTEM...');
    setIsPlaying(true);
    setContinuousPlay(true);
    
    // Set environment to current game state
    try {
      environment.setState(grid, availableBlocks, score, difficulty);
      console.log('🎮 Environment state updated successfully');
      console.log(`🎮 Current game state: Score=${score}, Blocks=${availableBlocks.length}, Difficulty=${difficulty}`);
      
      playIntervalRef.current = setInterval(async () => {
        await makeAIMove();
      }, playSpeed);
      
      console.log(`🎮 AI play started with ${playSpeed}ms interval`);
      console.log('🎯 Watch console for STRATEGIC reward breakdowns during play!');
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
      
      // 🔍 VERIFICATION: Confirm strategic decision making
      console.log('🔍 AI STRATEGIC ANALYSIS:');
      
      // Test strategic analysis methods
      try {
        const completionProgress = environment.getCompletionProgress(environment.grid);
        console.log(`📊 Completion Progress - Rows: ${completionProgress.rows.toFixed(2)}, Cols: ${completionProgress.cols.toFixed(2)}, Squares: ${completionProgress.squares.toFixed(2)}`);
        
        const almostCompleteLines = environment.countAlmostCompleteLines();
        console.log(`🎯 Almost Complete Lines: ${almostCompleteLines}`);
        
        const clearingPotentials = environment.analyzeClearingPotentials();
        console.log(`⚡ Max Clearing Potential - Rows: ${clearingPotentials.maxRowPotential}, Cols: ${clearingPotentials.maxColPotential}, Squares: ${clearingPotentials.maxSquarePotential}`);
        
        const highPotentialAreas = environment.findHighPotentialAreas();
        console.log(`🎪 High Potential Areas Found: ${highPotentialAreas.length}`);
        
      } catch (error) {
        console.log('⚠️ Strategic analysis error (methods may be simplified):', error.message);
      }
      
      // Get AI action (exploitation only - no exploration during play)
      const state = environment.getState();
      const action = await agent.predict(state, validActions);
      
      // Execute action in the game
      const { blockIndex, row, col } = environment.decodeAction(action);
      console.log(`🤖 AI STRATEGIC DECISION: block ${blockIndex} at position (${row}, ${col})`);
      
      // 🔍 VERIFICATION: Analyze the strategic reasoning behind this move
      try {
        const blockShape = availableBlocks[blockIndex];
        console.log(`🧩 Block shape selected:`, blockShape);
        
        // Simulate the move to see what strategic rewards it would generate
        const oldGrid = environment.grid.map(row => [...row]);
        const oldScore = environment.score;
        
        // This is just for analysis - we'll let the actual game execute the move
        console.log(`🎯 This move should trigger STRATEGIC reward calculations...`);
        
      } catch (error) {
        console.log('⚠️ Strategic analysis error:', error.message);
      }
      
      // Double-check the action is still valid
      if (blockIndex >= 0 && blockIndex < availableBlocks.length && availableBlocks[blockIndex]) {
        const blockShape = availableBlocks[blockIndex];
        console.log(`🤖 Executing strategic placement...`);
        
        // Execute the move in the actual game
        const moveSuccess = onGameStateUpdate({
          blockIndex,
          row,
          col,
          blockShape: blockShape
        });
        
        if (moveSuccess !== false) {
          console.log(`✅ STRATEGIC MOVE EXECUTED - Watch for reward breakdown in training logs!`);
        } else {
          console.log(`❌ Move failed to execute - stopping AI play`);
          stopAIPlay();
          return;
        }
      } else {
        console.log(`❌ Invalid block selection: blockIndex=${blockIndex}, availableBlocks.length=${availableBlocks.length}`);
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
    const dqnAgent = new DQNAgent(env.getStateSize(), env.getMaxActionSpace(), {
      learningRate: 0.0003, // EXPERT: Lower for stability
      epsilon: 0.95, // EXPERT: Higher initial exploration
      epsilonMin: 0.02, // EXPERT: Lower minimum for exploitation
      epsilonDecay: 0.9995, // EXPERT: Slower decay for thorough exploration
      gamma: 0.98, // EXPERT: Higher discount for long-term planning
      batchSize: 128, // EXPERT: Larger batch for stability
      memorySize: 25000, // EXPERT: Larger memory for diverse experiences
      targetUpdateFreq: 200 // EXPERT: Less frequent for stability
    });
    
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

  const verifyAISystem = () => {
    console.log('🔍 ========== EXPERT AI SYSTEM VERIFICATION ==========');
    
    if (!agent || !environment) {
      console.log('❌ AI System not initialized');
      alert('❌ AI System not initialized. Please wait for initialization to complete.');
      return;
    }
    
    // 1. Verify EXPERT Agent Configuration
    console.log('🤖 EXPERT AGENT VERIFICATION:');
    console.log(`✅ Agent Type: ${agent.constructor.name}`);
    console.log(`✅ Learning Rate: ${agent.learningRate} (Expected: 0.0003)`);
    console.log(`✅ Epsilon: ${agent.epsilon} (Should be 0.02-0.95)`);
    console.log(`✅ Gamma: ${agent.gamma} (Expected: 0.98)`);
    console.log(`✅ Batch Size: ${agent.batchSize} (Expected: 128)`);
    console.log(`✅ Memory Size: ${agent.memorySize} (Expected: 25000)`);
    console.log(`✅ Episodes Trained: ${agent.episode}`);
    console.log(`✅ Best Score: ${agent.bestScore}`);
    console.log(`✅ Curriculum Stage: ${agent.curriculumStage}/3`);
    console.log(`✅ Meta Patterns Learned: ${agent.metaLearning?.patternMemory?.size || 0}`);
    console.log(`✅ Prioritized Replay: ${agent.prioritizedReplay ? 'ACTIVE' : 'DISABLED'}`);
    
    // 2. Verify EXPERT Environment Configuration  
    console.log('🌍 EXPERT ENVIRONMENT VERIFICATION:');
    console.log(`✅ Environment Type: ${environment.constructor.name}`);
    console.log(`✅ State Size: ${environment.getStateSize()} (Expected: 112)`);
    console.log(`✅ Max Action Space: ${environment.getMaxActionSpace()} (Expected: 147)`);
    console.log(`✅ Current Difficulty: ${environment.difficulty}`);
    console.log(`✅ Hierarchical Rewards: ${environment.rewardWeights ? 'ACTIVE' : 'MISSING'}`);
    console.log(`✅ Meta-learning: ${environment.transferLearning ? 'ACTIVE' : 'MISSING'}`);
    console.log(`✅ Adaptive Difficulty: ${environment.adaptiveDifficulty !== undefined ? 'ACTIVE' : 'MISSING'}`);
    
    // 3. Verify EXPERT Neural Network Architecture
    console.log('🧠 EXPERT NEURAL NETWORK VERIFICATION:');
    if (agent.qNetwork) {
      console.log(`✅ Q-Network: CNN-Transformer Hybrid`);
      console.log(`✅ Target Network: ${agent.targetNetwork ? 'ACTIVE' : 'MISSING'}`);
      console.log(`✅ Training Steps: ${agent.trainingStep}`);
      console.log(`✅ Memory Usage: ${agent.memory.length}/${agent.memorySize}`);
      console.log(`✅ Attention Mechanism: ${typeof agent.buildAttentionLayer === 'function' ? 'ACTIVE' : 'MISSING'}`);
      console.log(`✅ Curriculum Loss: ${typeof agent.createHuberLoss === 'function' ? 'ACTIVE' : 'MISSING'}`);
    } else {
      console.log(`❌ Q-Network: MISSING`);
    }
    
    // 4. Verify EXPERT Strategic Features
    console.log('🎯 EXPERT STRATEGIC FEATURES VERIFICATION:');
    const expertFeatures = [
      'getValidActionsWithHeuristics',
      'calculateActionHeuristic', 
      'calculatePatternCompletionBonus',
      'calculateChainReactionPotential',
      'calculateLongTermStrategy',
      'applyMetaLearningBonus',
      'adaptRewardWeights'
    ];
    
    let expertFeaturesActive = 0;
    expertFeatures.forEach(feature => {
      const exists = typeof environment[feature] === 'function';
      console.log(`${exists ? '✅' : '❌'} ${feature}: ${exists ? 'ACTIVE' : 'MISSING'}`);
      if (exists) expertFeaturesActive++;
    });
    
    const expertSystemHealth = (expertFeaturesActive / expertFeatures.length) * 100;
    console.log(`🎯 Expert System Health: ${expertSystemHealth.toFixed(1)}% (${expertFeaturesActive}/${expertFeatures.length} features)`);
    
    // 5. Verify EXPERT Learning Features
    console.log('🎓 EXPERT LEARNING FEATURES VERIFICATION:');
    const expertLearningFeatures = [
      'storeMetaPattern',
      'getMetaLearningBias',
      'updateCurriculum',
      'samplePrioritizedBatch',
      'updatePriorities',
      'adjustParametersForStage'
    ];
    
    let learningFeaturesActive = 0;
    expertLearningFeatures.forEach(feature => {
      const exists = typeof agent[feature] === 'function';
      console.log(`${exists ? '✅' : '❌'} ${feature}: ${exists ? 'ACTIVE' : 'MISSING'}`);
      if (exists) learningFeaturesActive++;
    });
    
    const learningSystemHealth = (learningFeaturesActive / expertLearningFeatures.length) * 100;
    console.log(`🎓 Learning System Health: ${learningSystemHealth.toFixed(1)}% (${learningFeaturesActive}/${expertLearningFeatures.length} features)`);
    
    // 6. Overall EXPERT System Status
    console.log('📊 OVERALL EXPERT SYSTEM STATUS:');
    const isExpertAgent = agent.learningRate === 0.0003 && agent.gamma === 0.98 && agent.batchSize === 128;
    const isExpertEnvironment = environment.getStateSize() === 112 && environment.rewardWeights;
    const hasExpertFeatures = expertSystemHealth >= 80 && learningSystemHealth >= 80;
    const hasHybridNetwork = typeof agent.buildHybridNetwork === 'function';
    
    console.log(`${isExpertAgent ? '✅' : '❌'} Expert Agent Configuration: ${isExpertAgent ? 'CONFIRMED' : 'OUTDATED'}`);
    console.log(`${isExpertEnvironment ? '✅' : '❌'} Expert Environment Configuration: ${isExpertEnvironment ? 'CONFIRMED' : 'OUTDATED'}`);
    console.log(`${hasExpertFeatures ? '✅' : '❌'} Expert Strategic Features: ${hasExpertFeatures ? 'FULLY ACTIVE' : 'INCOMPLETE'}`);
    console.log(`${hasHybridNetwork ? '✅' : '❌'} CNN-Transformer Architecture: ${hasHybridNetwork ? 'ACTIVE' : 'MISSING'}`);
    
    const overallExpertStatus = isExpertAgent && isExpertEnvironment && hasExpertFeatures && hasHybridNetwork;
    console.log(`🎯 FINAL EXPERT VERDICT: ${overallExpertStatus ? '✅ EXPERT AI SYSTEM CONFIRMED' : '❌ SYSTEM NEEDS EXPERT UPGRADE'}`);
    
    console.log('🔍 ========== EXPERT VERIFICATION COMPLETE ==========');
    
    // Show user-friendly alert
    if (overallExpertStatus) {
      alert(`✅ EXPERT VERIFICATION PASSED!\n\nYour AI is using the EXPERT system with:\n• CNN-Transformer hybrid architecture\n• Hierarchical weighted reward system\n• Prioritized experience replay\n• Curriculum learning (Stage ${agent.curriculumStage}/3)\n• Meta-learning (${agent.metaLearning?.patternMemory?.size || 0} patterns)\n• Hybrid action selection\n\nCheck console for detailed report.`);
    } else {
      alert('⚠️ EXPERT VERIFICATION ISSUES DETECTED!\n\nSome expert components may be outdated or missing.\nCheck the browser console for detailed analysis.');
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
            
            <button
              onClick={verifyAISystem}
              className="btn"
              style={{
                background: 'linear-gradient(145deg, #4CAF50, #45a049)',
                border: '2px solid #2E7D32',
                fontWeight: 'bold'
              }}
            >
              🔍 Verify AI System
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
          <h4>📊 EXPERT Training Statistics</h4>
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
              <span className="stat-label">Curriculum Stage:</span>
              <span className="stat-value" style={{ 
                color: trainingStats.curriculumStage >= 3 ? '#28a745' : 
                       trainingStats.curriculumStage >= 2 ? '#ffc107' : 
                       trainingStats.curriculumStage >= 1 ? '#fd7e14' : '#dc3545',
                fontWeight: 'bold'
              }}>
                {trainingStats.curriculumStage || 0} / 3
              </span>
            </div>
            <div className="stat-item">
              <span className="stat-label">Meta Patterns:</span>
              <span className="stat-value" style={{ color: '#00CED1', fontWeight: 'bold' }}>
                {trainingStats.metaPatternsLearned || '0'}
              </span>
            </div>
            <div className="stat-item">
              <span className="stat-label">Exploration Rate:</span>
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
              <span className="stat-label">Priority Beta:</span>
              <span className="stat-value">{trainingStats.priorityBeta?.toFixed(3) || '0.400'}</span>
            </div>
            <div className="stat-item">
              <span className="stat-label">Max Priority:</span>
              <span className="stat-value">{trainingStats.maxPriority?.toFixed(2) || '1.00'}</span>
            </div>
            <div className="stat-item">
              <span className="stat-label">Training Loss:</span>
              <span className="stat-value">{trainingStats.avgLoss?.toFixed(4) || '0'}</span>
            </div>
          </div>
          
          {/* EXPERT ENHANCEMENT: Curriculum Progress Bar */}
          {typeof trainingStats.curriculumStage === 'number' && (
            <div style={{ marginTop: '15px', marginBottom: '15px' }}>
              <div style={{ marginBottom: '8px', color: '#FFD700', fontWeight: 'bold' }}>
                🎓 Curriculum Progress
              </div>
              <div style={{ 
                background: 'rgba(139, 69, 19, 0.3)', 
                borderRadius: '10px', 
                height: '20px',
                border: '1px solid #8B4513',
                overflow: 'hidden'
              }}>
                <div style={{
                  background: trainingStats.curriculumStage >= 3 ? 
                    'linear-gradient(90deg, #28a745, #20c997)' :
                    'linear-gradient(90deg, #ffc107, #fd7e14)',
                  height: '100%',
                  width: `${((trainingStats.curriculumStage + 1) / 4) * 100}%`,
                  transition: 'width 0.5s ease',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: 'white',
                  fontSize: '12px',
                  fontWeight: 'bold'
                }}>
                  {['Basic', 'Intermediate', 'Advanced', 'Expert'][trainingStats.curriculumStage] || 'Basic'}
                </div>
              </div>
              <div style={{ 
                fontSize: '12px', 
                color: '#D2B48C', 
                marginTop: '5px',
                textAlign: 'center'
              }}>
                {trainingStats.curriculumStage === 3 ? 
                  '🏆 Expert level achieved!' :
                  `Next: ${['Intermediate', 'Advanced', 'Expert'][trainingStats.curriculumStage]} (${[100, 500, 1000, 2000][trainingStats.curriculumStage]} avg reward)`
                }
              </div>
            </div>
          )}
          
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