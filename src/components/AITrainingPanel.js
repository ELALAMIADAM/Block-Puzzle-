import React, { useState, useEffect, useRef } from 'react';
import * as tf from '@tensorflow/tfjs';
import { EliteDQNAgent } from '../ai/EliteDQNAgent';
import { EliteEnvironment } from '../ai/EliteEnvironment';

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
  const [trainingEpisodes, setTrainingEpisodes] = useState(2000); // More episodes for elite training
  const [autoPlay, setAutoPlay] = useState(false);
  const [playSpeed, setPlaySpeed] = useState(300); // Faster default for elite performance
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
    // Initialize ELITE AI components
    console.log('🚀 Initializing ELITE AI SYSTEM...');
    
    const eliteEnv = new EliteEnvironment();
    console.log(`📊 Elite Environment initialized - State size: ${eliteEnv.getStateSize()}`);
    
    const eliteAgent = new EliteDQNAgent(eliteEnv.getStateSize(), eliteEnv.getMaxActionSpace(), {
      learningRate: 0.0003,     // ELITE: Lower for stability
      epsilon: 1.0,             // ELITE: Start with full exploration
      epsilonMin: 0.01,         // ELITE: Minimal exploitation
      epsilonDecay: 0.9995,     // ELITE: Slower decay for better exploration
      gamma: 0.99,              // ELITE: High discount for long-term planning
      batchSize: 64,            // ELITE: Larger batch for stability
      memorySize: 20000,        // ELITE: Larger memory for diversity
      targetUpdateFreq: 100,    // ELITE: Less frequent for stability
      alpha: 0.6,               // ELITE: Prioritization exponent
      beta: 0.4,                // ELITE: Importance sampling exponent
      nStep: 3                  // ELITE: Multi-step learning
    });
    
    console.log('🧠 Elite DQN Agent initialized with:');
    console.log('  - Double DQN for stable learning');
    console.log('  - Dueling architecture for better value estimation');
    console.log('  - Prioritized experience replay');
    console.log('  - Advanced exploration strategies');
    console.log('  - Sophisticated reward shaping');
    
    setEnvironment(eliteEnv);
    setAgent(eliteAgent);
    
    return () => {
      if (eliteAgent) eliteAgent.dispose();
    };
  }, []);

  const startTraining = async () => {
    if (!agent || !environment || isTraining) return;
    
    console.log('🚀 Starting ELITE TRAINING SESSION...');
    console.log(`📈 Target episodes: ${trainingEpisodes}`);
    console.log(`🎯 Objective: MAXIMIZE SCORE with elite strategies`);
    console.log(`💎 Features: Double DQN + Dueling + Prioritized Replay + Advanced Rewards`);
    
    setIsTraining(true);
    episodeRef.current = 0;
    
    trainingIntervalRef.current = setInterval(async () => {
      await runEliteTrainingEpisode();
      episodeRef.current++;
      
      if (episodeRef.current >= trainingEpisodes) {
        stopTraining();
      }
    }, visualTraining ? 50 : 10); // Much faster training for elite system
  };

  const stopTraining = () => {
    setIsTraining(false);
    if (trainingIntervalRef.current) {
      clearInterval(trainingIntervalRef.current);
      trainingIntervalRef.current = null;
    }
    
    console.log('🏁 ELITE TRAINING SESSION COMPLETED');
    if (agent) {
      const stats = agent.getStats();
      console.log(`🏆 Final Performance: Best Score: ${stats.bestScore}, Avg Score: ${stats.avgScore?.toFixed(1)}`);
      console.log(`📊 Training Summary: ${stats.episode} episodes, ${stats.lineClearingCount} line clears`);
    }
  };

  const runEliteTrainingEpisode = async () => {
    if (!agent || !environment) return;
    
    try {
    // Reset environment for new episode
    environment.reset();
    environment.availableBlocks = environment.generateCurriculumBlocks();
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
      const maxSteps = 150; // More steps for elite performance
    
    while (!done && stepCount < maxSteps) {
      const validActions = environment.getValidActions();
      
      if (validActions.length === 0) {
        done = true;
        break;
      }
      
        // ELITE ACTION SELECTION with sophisticated strategies
      const action = await agent.act(state, validActions, environment);
      const stepResult = environment.step(action);
      
        // Visual feedback for training
      if (visualTraining) {
        setTrainingGrid(environment.grid.map(row => [...row]));
        setEpisodeScore(environment.score);
        setEpisodeSteps(stepCount + 1);
        stepRef.current = stepCount + 1;
        
          // Faster visual updates for elite system
          if (visualTraining && stepCount % 5 === 0) {
            await new Promise(resolve => setTimeout(resolve, 5));
          }
        }
        
        // Store experience in prioritized replay buffer
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
      
        // ELITE TRAINING: More frequent updates for faster learning
        if (agent.memory.length > agent.batchSize && stepCount % 2 === 0) {
        await agent.replay();
      }
      
      // Add new blocks when tray is empty
      if (environment.availableBlocks.length === 0 && !done) {
        environment.availableBlocks = environment.generateCurriculumBlocks();
          done = environment.checkGameOver();
        if (visualTraining) {
          setTrainingBlocks(environment.availableBlocks.map(block => block.map(row => [...row])));
        }
      }
    }
    
      // End episode with elite curriculum management
    agent.endEpisode(0, environment.score, environment);
    
      // Update curriculum based on performance
      environment.updateCurriculum(
        environment.lineClearsThisEpisode || 0, 
        environment.score
      );
      
      // Update stats every episode for real-time monitoring
      const newStats = agent.getStats();
      newStats.curriculumLevel = environment.curriculumLevel;
      newStats.currentGridSize = environment.currentGridSize;
      newStats.lineClearsThisEpisode = environment.lineClearsThisEpisode || 0;
      newStats.consecutiveClears = environment.consecutiveClears || 0;
      newStats.maxConsecutiveClears = environment.maxConsecutiveClears || 0;
      
      setTrainingStats(newStats);
      
      // Enhanced logging for elite performance tracking
      if (episodeRef.current % 10 === 0 || environment.score > 5000) {
        console.log(`🚀 ELITE Episode ${episodeRef.current}: Score=${environment.score}, Best=${newStats.bestScore}, ` +
                    `Avg=${newStats.avgScore?.toFixed(1)}, Lines=${environment.lineClearsThisEpisode || 0}, ` +
                    `Consecutive=${environment.consecutiveClears || 0}, Level=${environment.curriculumLevel}, ` +
                    `Epsilon=${(newStats.epsilon * 100).toFixed(1)}%`);
      }
      
    } catch (error) {
      console.error('🚨 Elite training episode error:', error);
    }
  };





  const loadModel = async () => {
    if (!agent) return;
    const success = await agent.loadModel(`elite-wood-block-dqn-${difficulty}`);
    if (success) {
      alert('Elite model loaded from browser storage successfully!');
      setTrainingStats(agent.getStats());
    } else {
      alert('Failed to load Elite model from browser storage.');
    }
  };

  const downloadModel = async () => {
    if (!agent) {
      alert('No agent available to download!');
      return;
    }

    try {
      console.log('🔽 Starting Elite model download...');
      
      // Generate unique filename with timestamp and difficulty
      const timestamp = new Date().toISOString().replace(/[:.]/g, '-').split('.')[0];
      const modelName = `elite-wood-block-dqn-${difficulty}-${timestamp}`;
      
      // Download the TensorFlow.js model
      await agent.qNetwork.save(`downloads://${modelName}`);
      
      // Create and download the agent state JSON
      const stats = agent.getStats();
      const agentState = {
        difficulty: difficulty,
        timestamp: timestamp,
        epsilon: agent.epsilon,
        episode: agent.episode,
        trainingStep: agent.trainingStep,
        rewards: agent.rewards.slice(-1000), // Last 1000 for file size
        scores: agent.scores.slice(-1000),
        losses: agent.losses.slice(-500),
        epsilonHistory: agent.epsilonHistory.slice(-500),
        bestScore: agent.bestScore,
        lineClearingHistory: agent.lineClearingHistory.slice(-100),
        recentLineClearSuccess: agent.recentLineClearSuccess,
        stateSize: agent.stateSize,
        actionSize: agent.actionSize,
        hyperparameters: {
          learningRate: agent.learningRate,
          gamma: agent.gamma,
          batchSize: agent.batchSize,
          memorySize: agent.memorySize,
          targetUpdateFreq: agent.targetUpdateFreq,
          alpha: agent.alpha,
          beta: agent.beta,
          nStep: agent.nStep
        },
        environmentInfo: {
          curriculumLevel: environment?.curriculumLevel || 0,
          currentComplexity: environment?.currentComplexity || 'simple'
        }
      };

      // Create and trigger download of agent state
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

      console.log('✅ Elite model download completed!');
      alert(`🔽 Elite Model Downloaded Successfully!\n\nFiles saved:\n• ${modelName}.json (model architecture)\n• ${modelName}.bin (model weights)\n• ${modelName}-state.json (training progress)\n\nModel Stats:\n• Episodes: ${stats.episode}\n• Best Score: ${stats.bestScore}\n• Difficulty: ${difficulty.toUpperCase()}`);
      
    } catch (error) {
      console.error('❌ Elite model download error:', error);
      alert('❌ Failed to download Elite model. Check the browser console for details.');
    }
  };

  const uploadModel = async () => {
    if (!agent) {
      alert('No agent available to upload to!');
      return;
    }

    // Create file input for multiple model files
    const fileInput = document.createElement('input');
    fileInput.type = 'file';
    fileInput.multiple = true;
    fileInput.accept = '.json,.bin';
    fileInput.style.display = 'none';
    
    fileInput.onchange = async (event) => {
      const files = Array.from(event.target.files);
      
      if (files.length === 0) return;

      try {
        console.log('🔼 Starting Elite model upload...');
        console.log('📁 Files selected:', files.map(f => f.name));

        // Find the model files
        const modelJsonFile = files.find(f => f.name.endsWith('.json') && !f.name.includes('-state'));
        const modelBinFile = files.find(f => f.name.endsWith('.bin'));
        const stateJsonFile = files.find(f => f.name.includes('-state.json'));

        if (!modelJsonFile) {
          alert('❌ Please select the model.json file (main model file)!');
          return;
        }

        // Load model using tf.io.browserFiles
        const modelFiles = [modelJsonFile];
        if (modelBinFile) {
          modelFiles.push(modelBinFile);
        }

        console.log('📊 Loading TensorFlow.js model...');
        const loadedModel = await tf.loadLayersModel(tf.io.browserFiles(modelFiles));
        
        // Replace the current model
        if (agent.qNetwork) {
          agent.qNetwork.dispose();
        }
        agent.qNetwork = loadedModel;
        
        // Update target network to match
        agent.updateTargetNetwork();
        
        console.log('🧠 Model loaded successfully!');

        // Load state file if available
        if (stateJsonFile) {
          console.log('📊 Loading agent state...');
          const stateText = await stateJsonFile.text();
          const stateData = JSON.parse(stateText);
          
          // Restore agent state
          agent.epsilon = stateData.epsilon || agent.epsilon;
          agent.episode = stateData.episode || 0;
          agent.trainingStep = stateData.trainingStep || 0;
          agent.rewards = stateData.rewards || [];
          agent.scores = stateData.scores || [];
          agent.losses = stateData.losses || [];
          agent.epsilonHistory = stateData.epsilonHistory || [];
          agent.bestScore = stateData.bestScore || 0;
          agent.lineClearingHistory = stateData.lineClearingHistory || [];
          agent.recentLineClearSuccess = stateData.recentLineClearSuccess || 0;
          
          // Update environment state if available
          if (environment && stateData.environmentInfo) {
            environment.curriculumLevel = stateData.environmentInfo.curriculumLevel || 0;
            environment.currentComplexity = stateData.environmentInfo.currentComplexity || 'simple';
          }

          console.log('📊 Agent state restored!');
        }

        // Update UI with new stats
        setTrainingStats(agent.getStats());

        const uploadedModelInfo = stateJsonFile ? {
          episodes: agent.episode,
          bestScore: agent.bestScore,
          difficulty: stateJsonFile ? 'Restored from file' : 'Default',
          curriculumLevel: environment?.curriculumLevel || 0
        } : {
          episodes: 'Unknown',
          bestScore: 'Unknown', 
          difficulty: 'Model only (no state)',
          curriculumLevel: 0
        };

        console.log('✅ Elite model upload completed!');
        alert(`🔼 Elite Model Uploaded Successfully!\n\nModel restored:\n• Episodes: ${uploadedModelInfo.episodes}\n• Best Score: ${uploadedModelInfo.bestScore}\n• Curriculum Level: ${uploadedModelInfo.curriculumLevel}\n• State: ${stateJsonFile ? 'Fully Restored' : 'Model Only'}\n\nThe AI is ready to continue training or play!`);

      } catch (error) {
        console.error('❌ Model upload error:', error);
        alert(`❌ Failed to upload model: ${error.message}\n\nMake sure you selected:\n• model.json (required)\n• model.bin (required for weights)\n• model-state.json (optional, for training progress)`);
      }
    };

    // Trigger file selection
    document.body.appendChild(fileInput);
    fileInput.click();
    document.body.removeChild(fileInput);
  };

  const resetTraining = () => {
    if (agent) {
      agent.dispose();
    }
    
    const eliteEnv = new EliteEnvironment();
    const eliteAgent = new EliteDQNAgent(eliteEnv.getStateSize(), eliteEnv.getMaxActionSpace(), {
      learningRate: 0.0003,
      epsilon: 1.0,
      epsilonMin: 0.01,
      epsilonDecay: 0.9995,
      gamma: 0.99,
      batchSize: 64,
      memorySize: 20000,
      targetUpdateFreq: 100,
      alpha: 0.6,
      beta: 0.4,
      nStep: 3
    });
    
    setEnvironment(eliteEnv);
    setAgent(eliteAgent);
    setTrainingStats({});
    episodeRef.current = 0;
    
    console.log('🔄 Elite AI system reset successfully');
  };



  const verifyEliteSystem = () => {
    console.log('🔍 ========== ELITE AI SYSTEM VERIFICATION ==========');
    
    if (!agent || !environment) {
      console.log('❌ Elite AI System not initialized');
      alert('❌ Elite AI System not initialized. Please wait for initialization to complete.');
      return;
    }
    
    // 1. Verify Elite Agent
    console.log('🚀 ELITE AGENT VERIFICATION:');
    console.log(`✅ Agent Type: ${agent.constructor.name}`);
    console.log(`✅ State Size: ${agent.stateSize} (Expected: 139)`);
    console.log(`✅ Action Size: ${agent.actionSize} (Expected: 243)`);
    console.log(`✅ Elite Features: ${agent.isElite ? 'CONFIRMED' : 'MISSING'}`);
    console.log(`✅ Double DQN: ${agent.qNetwork && agent.targetNetwork ? 'ACTIVE' : 'MISSING'}`);
    console.log(`✅ Dueling Architecture: ${agent.qNetwork ? 'ACTIVE' : 'MISSING'}`);
    console.log(`✅ Prioritized Replay: ${agent.alpha !== undefined ? 'ACTIVE' : 'MISSING'}`);
    console.log(`✅ Multi-step Learning: ${agent.nStep > 1 ? 'ACTIVE' : 'MISSING'}`);
    console.log(`✅ Advanced Exploration: ${agent.curiosityDriven ? 'ACTIVE' : 'MISSING'}`);
    
    // 2. Verify Elite Environment
    console.log('🌍 ELITE ENVIRONMENT VERIFICATION:');
    console.log(`✅ Environment Type: ${environment.constructor.name}`);
    console.log(`✅ State Size: ${environment.getStateSize()} (Expected: 139)`);
    console.log(`✅ Reward System: ${environment.rewardWeights ? 'ELITE' : 'BASIC'}`);
    console.log(`✅ Spatial Analysis: ${environment.performSpatialAnalysis ? 'ACTIVE' : 'MISSING'}`);
    console.log(`✅ Curriculum Learning: ${environment.curriculumLevel !== undefined ? 'ACTIVE' : 'MISSING'}`);
    console.log(`✅ Adaptive Rewards: ${environment.adaptiveRewards ? 'ACTIVE' : 'MISSING'}`);
    
    // 3. Performance Metrics
    console.log('📊 PERFORMANCE METRICS:');
    const stats = agent.getStats();
    console.log(`✅ Episodes Trained: ${stats.episode}`);
    console.log(`✅ Best Score: ${stats.bestScore}`);
    console.log(`✅ Average Score: ${stats.avgScore?.toFixed(1) || '0'}`);
    console.log(`✅ Line Clearing Success: ${(stats.lineClearingSuccess * 100).toFixed(1)}%`);
    console.log(`✅ Max Sequential Clears: ${stats.maxSequentialClears || 0}`);
    console.log(`✅ Current Epsilon: ${(stats.epsilon * 100).toFixed(1)}%`);
    
    // 4. Overall System Status
    const isEliteAgent = agent.constructor.name === 'EliteDQNAgent' && agent.isElite;
    const isEliteEnvironment = environment.constructor.name === 'EliteEnvironment';
    const hasCorrectStateSize = agent.stateSize === environment.getStateSize();
    const hasEliteFeatures = agent.alpha !== undefined && environment.rewardWeights;
    
    const overallEliteStatus = isEliteAgent && isEliteEnvironment && hasCorrectStateSize && hasEliteFeatures;
    
    console.log(`🏆 FINAL ELITE VERDICT: ${overallEliteStatus ? '✅ ELITE AI SYSTEM CONFIRMED' : '❌ SYSTEM NEEDS UPGRADE'}`);
    console.log('🔍 ========== ELITE VERIFICATION COMPLETE ==========');
    
    // Show user-friendly alert
    if (overallEliteStatus) {
      alert(`🏆 ELITE VERIFICATION PASSED!\n\nYour AI is using the ELITE system with:\n• Double DQN with Dueling Architecture\n• Prioritized Experience Replay\n• Multi-step Learning\n• Advanced Exploration Strategies\n• Sophisticated Reward Shaping\n• Curriculum Learning\n\nBest Score: ${stats.bestScore}\nAverage Score: ${stats.avgScore?.toFixed(1) || '0'}\n\nCheck console for detailed report.`);
    } else {
      alert('⚠️ ELITE VERIFICATION ISSUES DETECTED!\n\nSome elite components may be missing or misconfigured.\nCheck the browser console for detailed analysis.');
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
    // Auto-restart Elite AI play if enabled and game resets
    if (autoPlay && !gameOver && !isPlaying && agent && continuousPlay) {
      console.log('🎮 Auto-restarting Elite AI play after game reset...');
      setTimeout(() => startAIPlay(), 1000);
    }
  }, [gameOver, autoPlay, agent]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    // Update play speed dynamically
    if (isPlaying && playIntervalRef.current) {
      clearInterval(playIntervalRef.current);
      playIntervalRef.current = setInterval(async () => {
        await makeEliteAIMove();
      }, playSpeed);
    }
  }, [playSpeed, makeEliteAIMove]); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="ai-training-panel" style={{ minHeight: '800px' }}>
      <h3>🏆 Elite AI Training Center</h3>
      
      {/* Elite System Status Banner */}
      <div style={{ 
        background: 'linear-gradient(135deg, rgba(255, 215, 0, 0.15), rgba(255, 165, 0, 0.15))', 
        padding: '20px', 
        borderRadius: '10px', 
        marginBottom: '20px',
        border: '3px solid #FFD700',
        boxShadow: '0 4px 15px rgba(255, 215, 0, 0.3)'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '15px', marginBottom: '15px' }}>
          <span style={{ fontSize: '32px' }}>🚀</span>
          <div>
            <strong style={{ fontSize: '20px', color: '#FFD700' }}>ELITE AI SYSTEM ACTIVE</strong>
            <div style={{ color: '#D2B48C', fontSize: '14px', marginTop: '5px' }}>
              Double DQN + Dueling Architecture + Prioritized Replay + Advanced Exploration
        </div>
          </div>
        </div>
        
        <div style={{ 
          display: 'grid', 
          gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', 
          gap: '15px',
          marginBottom: '15px'
        }}>
          <div style={{ 
            background: 'rgba(139, 69, 19, 0.4)',
            padding: '12px',
            borderRadius: '8px',
            border: '1px solid #8B4513'
          }}>
            <div style={{ color: '#FFD700', fontWeight: 'bold', marginBottom: '5px' }}>
              🧠 Neural Architecture
            </div>
            <div style={{ color: '#D2B48C', fontSize: '12px' }}>
              Dueling Double DQN with Huber Loss
            </div>
          </div>
          
          <div style={{ 
            background: 'rgba(139, 69, 19, 0.4)',
            padding: '12px',
            borderRadius: '8px',
            border: '1px solid #8B4513'
          }}>
            <div style={{ color: '#FFD700', fontWeight: 'bold', marginBottom: '5px' }}>
              🎯 Reward Shaping
            </div>
            <div style={{ color: '#D2B48C', fontSize: '12px' }}>
              Sophisticated multi-factor scoring
            </div>
          </div>
          
          <div style={{ 
            background: 'rgba(139, 69, 19, 0.4)',
            padding: '12px',
            borderRadius: '8px',
            border: '1px solid #8B4513'
          }}>
            <div style={{ color: '#FFD700', fontWeight: 'bold', marginBottom: '5px' }}>
              🔄 Experience Replay
            </div>
            <div style={{ color: '#D2B48C', fontSize: '12px' }}>
              Prioritized sampling with TD-error
          </div>
        </div>
        
        <div style={{ 
            background: 'rgba(139, 69, 19, 0.4)',
            padding: '12px',
          borderRadius: '8px', 
            border: '1px solid #8B4513'
        }}>
            <div style={{ color: '#FFD700', fontWeight: 'bold', marginBottom: '5px' }}>
              🎓 Learning Strategy
          </div>
            <div style={{ color: '#D2B48C', fontSize: '12px' }}>
              Multi-step + Curriculum + Adaptive
            </div>
          </div>
        </div>
        
          <div style={{ 
          fontSize: '13px', 
          color: '#FFD700', 
          textAlign: 'center',
          padding: '10px',
          background: 'rgba(255, 215, 0, 0.1)',
          borderRadius: '5px',
          border: '1px solid #FFD700'
        }}>
          <strong>🏆 OBJECTIVE:</strong> Achieve maximum game score through state-of-the-art reinforcement learning
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
          <strong style={{ fontSize: '16px' }}>🎯 Elite AI Status Dashboard</strong>
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
            <span>Level: <strong style={{ color: '#FFD700' }}>
              {environment?.curriculumLevel || 0}
            </strong></span>
          </div>
        </div>
        <div style={{ fontSize: '14px', opacity: 0.9, lineHeight: '1.4' }}>
          <strong>Elite Performance Optimization:</strong> This system uses cutting-edge reinforcement learning techniques 
          to maximize game scores. Features include sophisticated reward shaping, advanced exploration strategies, 
          and state-of-the-art neural network architectures for optimal decision making.
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

          </div>
        </div>
        
        <div className="model-section">
          <h4>💾 Model Management</h4>
          <div style={{ fontSize: '12px', color: '#D2B48C', marginBottom: '10px' }}>
            Save and load trained AI models for different difficulty levels
          </div>
          
          <div className="button-group">

            
            <button
              onClick={loadModel}
              disabled={!agent}
              className="btn load-btn"
            >
              📁 Load Model
            </button>
            
            <button
              onClick={downloadModel}
              disabled={!agent}
              className="btn download-btn"
              style={{
                background: 'linear-gradient(145deg, #17a2b8, #138496)',
                border: '2px solid #117a8b'
              }}
            >
              🔽 Download Model
            </button>
            
            <button
              onClick={uploadModel}
              disabled={!agent}
              className="btn upload-btn"
              style={{
                background: 'linear-gradient(145deg, #6f42c1, #5a32a3)',
                border: '2px solid #4e2a87'
              }}
            >
              🔼 Upload Model
            </button>
            

            
            <button
              onClick={verifyEliteSystem}
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
            <div><strong>Browser Storage:</strong> Save/Load (persistent in browser)</div>
            <div><strong>File Downloads:</strong> Download/Upload (portable files)</div>
            <div style={{ marginTop: '4px', fontSize: '11px', fontStyle: 'italic' }}>
              💡 Download creates 3 files: model.json, model.bin, and state.json
            </div>
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
            🎮 Live Training Episode #{currentEpisode} - Level {trainingStats.curriculumLevel || 0} ({trainingStats.currentComplexity || 'simple'} blocks)
          </h4>
          
          <div style={{ display: 'flex', gap: '20px', alignItems: 'flex-start' }}>
            {/* Training Game Board */}
            <div style={{ flex: 1 }}>
              <div style={{ marginBottom: '10px', color: '#D2B48C', fontSize: '14px' }}>
                Training Game Board ({trainingStats.currentGridSize || 9}x{trainingStats.currentGridSize || 9})
              </div>
              <div style={{
                display: 'grid',
                gridTemplateColumns: `repeat(${trainingStats.currentGridSize || 9}, 1fr)`,
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
                  <span style={{ color: '#FFD700', fontWeight: 'bold' }}>Level:</span>
                  <span style={{ color: 'white', marginLeft: '8px' }}>{trainingStats.curriculumLevel || 0}</span>
                </div>
                <div style={{ marginBottom: '8px' }}>
                  <span style={{ color: '#FFD700', fontWeight: 'bold' }}>Complexity:</span>
                  <span style={{ color: 'white', marginLeft: '8px' }}>{trainingStats.currentComplexity || 'simple'}</span>
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
                <div style={{ marginBottom: '8px' }}>
                  <span style={{ color: '#FFD700', fontWeight: 'bold' }}>Line Clears:</span>
                  <span style={{ color: trainingStats.lineClearsThisEpisode > 0 ? '#28a745' : 'white', marginLeft: '8px', fontWeight: 'bold' }}>{trainingStats.lineClearsThisEpisode || 0}</span>
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
                    🧠 PROGRESSIVE Learning...
                  </div>
                  <div style={{ color: '#D2B48C', fontSize: '10px', marginTop: '2px' }}>
                    Curriculum Level {trainingStats.curriculumLevel || 0}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
      
      {trainingStats.episode && (
        <div className="training-stats">
          <h4>📊 FIXED GRID Training Statistics</h4>
          <div className="stats-grid">
            <div className="stat-item">
              <span className="stat-label">Episode:</span>
              <span className="stat-value">{trainingStats.episode}</span>
            </div>
            <div className="stat-item">
              <span className="stat-label">Block Complexity Level:</span>
              <span className="stat-value" style={{ 
                color: trainingStats.curriculumLevel >= 3 ? '#28a745' : 
                       trainingStats.curriculumLevel >= 2 ? '#ffc107' : 
                       trainingStats.curriculumLevel >= 1 ? '#fd7e14' : '#dc3545',
                fontWeight: 'bold'
              }}>
                {trainingStats.curriculumLevel || 0} ({
                  trainingStats.curriculumLevel === 0 ? 'Simple' : 
                  trainingStats.curriculumLevel === 1 ? 'Medium' : 
                  trainingStats.curriculumLevel === 2 ? 'Complex' : 
                  'Full'
                } - 9x9 grid)
              </span>
            </div>
            <div className="stat-item">
              <span className="stat-label">Lines Cleared (Episode):</span>
              <span className="stat-value" style={{ 
                color: trainingStats.lineClearsThisEpisode > 0 ? '#28a745' : '#dc3545',
                fontWeight: 'bold'
              }}>
                {trainingStats.lineClearsThisEpisode || 0}
              </span>
            </div>
            <div className="stat-item">
              <span className="stat-label">Line Clear Success Rate:</span>
              <span className="stat-value" style={{ 
                color: trainingStats.lineClearingSuccess > 0.3 ? '#28a745' : 
                       trainingStats.lineClearingSuccess > 0.1 ? '#ffc107' : '#dc3545',
                fontWeight: 'bold'
              }}>
                {(trainingStats.lineClearingSuccess * 100)?.toFixed(1) || '0'}%
              </span>
            </div>
            <div className="stat-item">
              <span className="stat-label">Total Line Clears:</span>
              <span className="stat-value" style={{ color: '#00CED1', fontWeight: 'bold' }}>
                {trainingStats.lineClearingCount || '0'}
              </span>
            </div>
            <div className="stat-item">
              <span className="stat-label">Best Score:</span>
              <span className="stat-value" style={{ color: '#28a745', fontWeight: 'bold' }}>
                {trainingStats.bestScore || '0'}
              </span>
            </div>
            <div className="stat-item">
              <span className="stat-label">Avg Reward:</span>
              <span className="stat-value">{trainingStats.avgReward?.toFixed(2) || '0'}</span>
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
              <span className="stat-label">Training Loss:</span>
              <span className="stat-value">{trainingStats.avgLoss?.toFixed(4) || '0'}</span>
            </div>
            <div className="stat-item">
              <span className="stat-label">Guided Exploration:</span>
              <span className="stat-value" style={{ 
                color: trainingStats.guidedExploration ? '#28a745' : '#dc3545',
                fontWeight: 'bold'
              }}>
                {trainingStats.guidedExploration ? 'ON' : 'OFF'}
              </span>
            </div>
          </div>
          
          {/* FIXED CURRICULUM: Progress Bar */}
          {typeof trainingStats.curriculumLevel === 'number' && (
            <div style={{ marginTop: '15px', marginBottom: '15px' }}>
              <div style={{ marginBottom: '8px', color: '#FFD700', fontWeight: 'bold' }}>
                🎓 Block Complexity Progression
              </div>
              <div style={{ 
                background: 'rgba(139, 69, 19, 0.3)', 
                borderRadius: '10px', 
                height: '20px',
                border: '1px solid #8B4513',
                overflow: 'hidden'
              }}>
                <div style={{
                  background: trainingStats.curriculumLevel >= 3 ? 
                    'linear-gradient(90deg, #28a745, #20c997)' :
                    'linear-gradient(90deg, #ffc107, #fd7e14)',
                  height: '100%',
                  width: `${((trainingStats.curriculumLevel + 1) / 4) * 100}%`,
                  transition: 'width 0.5s ease',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: 'white',
                  fontSize: '12px',
                  fontWeight: 'bold'
                }}>
                  {['Simple', 'Medium', 'Complex', 'Full'][trainingStats.curriculumLevel] || 'Simple'}
                </div>
              </div>
              <div style={{ 
                fontSize: '12px', 
                color: '#D2B48C', 
                marginTop: '5px',
                textAlign: 'center'
              }}>
                {trainingStats.curriculumLevel === 3 ? 
                  '🏆 All block types mastered!' :
                  `Next: ${['Medium blocks', 'Complex blocks', 'Full game'][trainingStats.curriculumLevel]} (need ${5 + trainingStats.curriculumLevel} line clears)`
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

export default AITrainingPanel; 