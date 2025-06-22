import * as tf from '@tensorflow/tfjs';

/**
 * VISUAL CNN-DQN AGENT - ADVANCED CONVOLUTIONAL SPATIAL INTELLIGENCE
 * 
 * Revolutionary Features:
 * - Convolutional Neural Network for spatial pattern recognition
 * - Multi-channel visual input processing (4 channels)
 * - Advanced feature extraction through conv layers
 * - Spatial relationship learning
 * - Visual pattern memory system
 * - Sophisticated exploration strategies for visual patterns
 */
export class ConvDQNAgent {
  constructor(visualStateSize, actionSize, options = {}) {
    this.visualStateSize = visualStateSize; // [CHANNELS, HEIGHT, WIDTH]
    this.actionSize = actionSize;
    
    // CNN-OPTIMIZED HYPERPARAMETERS
    this.learningRate = options.learningRate || 0.0005; // Lower LR for CNN stability
    this.epsilon = options.epsilon || 0.9; // High initial exploration for visual learning
    this.epsilonMin = options.epsilonMin || 0.05; 
    this.epsilonDecay = options.epsilonDecay || 0.996; // Slower decay for CNN learning
    this.gamma = options.gamma || 0.99; // Standard discount factor
    this.batchSize = options.batchSize || 32;
    this.memorySize = options.memorySize || 5000; // Standard memory size
    this.targetUpdateFreq = options.targetUpdateFreq || 100; // Less frequent updates for stability
    
    // VISUAL PATTERN LEARNING - Simplified
    this.visualExploration = true;
    this.patternRecognitionRate = 0.3; // 30% pattern-guided exploration
    this.lineClearingPriority = 15.0; // High priority for line clearing experiences
    
    // Pattern memory system (simplified)
    this.visualPatternHistory = [];
    this.recentLineClearSuccess = 0;
    
    // Experience replay with visual prioritization
    this.memory = [];
    this.memoryCounter = 0;
    
    // Training metrics
    this.trainingStep = 0;
    this.totalReward = 0;
    this.episode = 0;
    this.isTraining = false;
    
    // Build CNN networks
    this.qNetwork = this.buildConvolutionalNetwork();
    this.targetNetwork = this.buildConvolutionalNetwork();
    this.updateTargetNetwork();
    
    // Training and performance tracking
    this.losses = [];
    this.rewards = [];
    this.scores = [];
    this.epsilonHistory = [];
    this.bestScore = 0;
    
    // Mark as CNN agent
    this.isCNN = true;
    this.isVisual = true;
    
    console.log('🎨 VISUAL CNN-DQN AGENT INITIALIZED');
    console.log(`🖼️  Input Shape: [${Array.isArray(this.visualStateSize) ? this.visualStateSize.join(', ') : this.visualStateSize}]`);
    console.log(`🧠 Architecture: CNN for 12×12 spatial pattern recognition`);
    console.log(`🎯 Focus: Line clearing with visual intelligence`);
  }

  /**
   * CNN ARCHITECTURE - Optimized for 12x12x4 input
   */
  buildConvolutionalNetwork() {
    const model = tf.sequential();
    
    // CNN layers for spatial pattern recognition
    model.add(tf.layers.conv2d({
      inputShape: [12, 12, 4], // [height, width, channels] for TF.js
      filters: 32,
      kernelSize: [3, 3],
      strides: [1, 1],
      padding: 'same',
      activation: 'relu',
      kernelInitializer: 'heNormal',
      name: 'conv1'
    }));
    
    model.add(tf.layers.conv2d({
      filters: 64,
      kernelSize: [3, 3],
      strides: [1, 1],
      padding: 'same',
      activation: 'relu',
      kernelInitializer: 'heNormal',
      name: 'conv2'
    }));
    
    model.add(tf.layers.maxPooling2d({
      poolSize: [2, 2],
      strides: [2, 2],
      name: 'pool1'
    }));
    
    model.add(tf.layers.conv2d({
      filters: 128,
      kernelSize: [3, 3],
      strides: [1, 1],
      padding: 'same',
      activation: 'relu',
      kernelInitializer: 'heNormal',
      name: 'conv3'
    }));
    
    // Global average pooling for spatial intelligence
    model.add(tf.layers.globalAveragePooling2d({ name: 'gap' }));
    
    // Dense decision layers
    model.add(tf.layers.dense({
      units: 256,
      activation: 'relu',
      kernelInitializer: 'heNormal',
      name: 'dense1'
    }));
    
    model.add(tf.layers.dropout({ rate: 0.2, name: 'dropout1' }));
    
    model.add(tf.layers.dense({
      units: 128,
      activation: 'relu',
      kernelInitializer: 'heNormal',
      name: 'dense2'
    }));
    
    model.add(tf.layers.dropout({ rate: 0.2, name: 'dropout2' }));
    
    // Output layer
    model.add(tf.layers.dense({
      units: this.actionSize,
      activation: 'linear',
      kernelInitializer: 'heNormal',
      name: 'q_values'
    }));
    
    // Compile with optimizer
    const optimizer = tf.train.adam(this.learningRate);
    
    model.compile({
      optimizer: optimizer,
      loss: 'meanSquaredError',
      metrics: ['mse']
    });
    
    console.log('🧠 CNN ARCHITECTURE BUILT:');
    model.summary();
    
    return model;
  }

  /**
   * ACTION SELECTION - Like DQN but with visual pattern guidance
   */
  async act(state, validActions = null, environment = null) {
    // Adaptive epsilon based on curriculum level and success
    let adaptiveEpsilon = this.epsilon;
    if (environment && environment.curriculumLevel !== undefined) {
      adaptiveEpsilon = this.epsilon * (1 - environment.curriculumLevel * 0.2);
    }
    
    if (Math.random() <= adaptiveEpsilon) {
      // PATTERN-GUIDED EXPLORATION (like DQN's guided exploration)
      if (this.visualExploration && environment && Math.random() < this.patternRecognitionRate) {
        return this.selectPatternGuidedAction(validActions, environment);
      } else {
        // Random exploration
        if (validActions && validActions.length > 0) {
          return validActions[Math.floor(Math.random() * validActions.length)];
        }
        return Math.floor(Math.random() * this.actionSize);
      }
    }
    
    // CNN-based exploitation
    return await this.predictWithCNN(state, validActions);
  }

  /**
   * PATTERN-GUIDED EXPLORATION - Like DQN's guided exploration but for visual patterns
   */
  selectPatternGuidedAction(validActions, environment) {
    if (!validActions || validActions.length === 0) {
      return 0;
    }
    
    let bestAction = validActions[0];
    let bestScore = -Infinity;
    
    // Evaluate actions based on line completion potential (like DQN)
    for (const action of validActions.slice(0, Math.min(10, validActions.length))) {
      const { blockIndex, row, col } = environment.decodeAction(action);
      
      if (blockIndex < environment.availableBlocks.length) {
        const block = environment.availableBlocks[blockIndex];
        const lineScore = this.evaluateLineCompletionPotential(block, row, col, environment);
        
        if (lineScore > bestScore) {
          bestScore = lineScore;
          bestAction = action;
        }
      }
    }
    
    console.log(`🎨 PATTERN-GUIDED: Selected action with line score ${bestScore.toFixed(2)}`);
    return bestAction;
  }

  /**
   * EVALUATE LINE COMPLETION POTENTIAL - Like DQN but adapted for 12x12
   */
  evaluateLineCompletionPotential(block, row, col, environment) {
    let score = 0;
    const grid = environment.grid;
    const size = environment.currentGridSize;
    
    // Simulate placing the block
    const testGrid = grid.map(r => [...r]);
    for (let r = 0; r < block.length; r++) {
      for (let c = 0; c < block[r].length; c++) {
        if (block[r][c] && row + r < size && col + c < size) {
          testGrid[row + r][col + c] = true;
        }
      }
    }
    
    // Count how many lines this would complete
    let completedLines = 0;
    
    // Check rows
    for (let r = 0; r < size; r++) {
      if (testGrid[r].every(cell => cell)) {
        completedLines++;
      }
    }
    
    // Check columns
    for (let c = 0; c < size; c++) {
      let complete = true;
      for (let r = 0; r < size; r++) {
        if (!testGrid[r][c]) {
          complete = false;
          break;
        }
      }
      if (complete) completedLines++;
    }
    
    // Massive bonus for completing lines (like DQN)
    score += completedLines * 1000;
    
    // Small bonus for making lines closer to completion
    for (let r = 0; r < size; r++) {
      const filled = testGrid[r].filter(cell => cell).length;
      if (filled >= size - 1) score += 10; // Almost complete
    }
    
    for (let c = 0; c < size; c++) {
      let filled = 0;
      for (let r = 0; r < size; r++) {
        if (testGrid[r][c]) filled++;
      }
      if (filled >= size - 1) score += 10; // Almost complete
    }
    
    return score;
  }

  /**
   * CNN PREDICTION
   */
  async predictWithCNN(state, validActions = null) {
    const qValues = this.qNetwork.predict(state);
    
    let actionIndex;
    
    if (validActions && validActions.length > 0) {
      const qArray = await qValues.data();
      let bestValue = -Infinity;
      let bestAction = validActions[0];
      
      // Map environment action IDs to network indices
      for (let i = 0; i < validActions.length; i++) {
        const action = validActions[i];
        const networkActionIndex = i % this.actionSize;
        const value = qArray[networkActionIndex];
        
        if (value > bestValue) {
          bestValue = value;
          bestAction = action;
        }
      }
      
      actionIndex = bestAction;
    } else {
      actionIndex = (await qValues.argMax(1).data())[0];
    }
    
    qValues.dispose();
    
    return actionIndex;
  }

  /**
   * PRIORITIZED MEMORY - Like DQN but for line clearing
   */
  remember(state, action, reward, nextState, done) {
    const experience = {
      state: state.clone(),
      action,
      reward,
      nextState: nextState.clone(),
      done
    };
    
    // Store with prioritization (like DQN)
    if (this.memory.length < this.memorySize) {
      this.memory.push(experience);
    } else {
      // Overwrite oldest non-line-clearing experience first
      let replaceIndex = this.memoryCounter % this.memorySize;
      
      // If this is a line clearing experience, replace a non-line-clearing one
      if (reward > 5000) {
        for (let i = 0; i < this.memory.length; i++) {
          if (this.memory[i].reward < 5000) {
            replaceIndex = i;
            break;
          }
        }
      }
      
      // Clean up old tensors
      if (this.memory[replaceIndex]) {
        this.memory[replaceIndex].state.dispose();
        this.memory[replaceIndex].nextState.dispose();
      }
      
      this.memory[replaceIndex] = experience;
    }
    
    this.memoryCounter++;
    
    // Track line clearing specifically (like DQN)
    if (reward > 5000) {
      this.visualPatternHistory.push({
        episode: this.episode,
        reward: reward,
        timestamp: Date.now()
      });
      
      console.log(`🎨 LINE CLEARING EXPERIENCE STORED! Reward: ${reward.toFixed(2)}, Episode: ${this.episode}`);
    }
    
    // Update recent success rate
    const recentHistory = this.visualPatternHistory.slice(-10);
    this.recentLineClearSuccess = recentHistory.length / 10;
    
    this.totalReward += reward;
  }

  /**
   * TRAINING - Like DQN but with CNN
   */
  async replay() {
    if (this.memory.length < this.batchSize) {
      return;
    }

    if (this.isTraining) {
      return;
    }
    
    this.isTraining = true;

    try {
      // Sample batch with line clearing bias (like DQN)
      const batch = this.sampleLineClearingBatch();
      
      const states = tf.stack(batch.map(exp => exp.state));
      const nextStates = tf.stack(batch.map(exp => exp.nextState));
      
      const currentQValues = this.qNetwork.predict(states);
      const nextQValues = this.targetNetwork.predict(nextStates);
      const maxNextQValues = nextQValues.max(1);
      
      const targets = await currentQValues.array();
      const nextQArray = await maxNextQValues.array();
      
      for (let i = 0; i < batch.length; i++) {
        const exp = batch[i];
        let target = exp.reward;
        
        if (!exp.done) {
          target += this.gamma * nextQArray[i];
        }
        
        // Map action to network index
        const actionIndex = exp.action % this.actionSize;
        targets[i][actionIndex] = target;
      }
      
      const targetTensor = tf.tensor2d(targets);
      
      // Train the network
      const history = await this.qNetwork.fit(states, targetTensor, {
        epochs: 1,
        verbose: 0,
        batchSize: this.batchSize
      });
      
      this.losses.push(history.history.loss[0]);
      
      // Adaptive epsilon decay based on line clearing success (like DQN)
      const successBonus = this.recentLineClearSuccess * 0.002;
      if (this.epsilon > this.epsilonMin) {
        this.epsilon *= (this.epsilonDecay + successBonus);
      }
      this.epsilonHistory.push(this.epsilon);
      
      // Update target network
      this.trainingStep++;
      if (this.trainingStep % this.targetUpdateFreq === 0) {
        this.updateTargetNetwork();
      }
      
      // Cleanup tensors
      states.dispose();
      nextStates.dispose();
      currentQValues.dispose();
      nextQValues.dispose();
      maxNextQValues.dispose();
      targetTensor.dispose();
      
    } catch (error) {
      console.error('🚨 CNN Training error:', error);
    } finally {
      this.isTraining = false;
    }
  }

  /**
   * PRIORITIZED SAMPLING - Like DQN
   */
  sampleLineClearingBatch() {
    const batch = [];
    const lineClearingExperiences = this.memory.filter(exp => exp.reward > 5000);
    const otherExperiences = this.memory.filter(exp => exp.reward <= 5000);
    
    // Include more line clearing experiences (like DQN)
    const lineClearingCount = Math.min(
      Math.floor(this.batchSize * 0.7), // 70% line clearing experiences
      lineClearingExperiences.length
    );
    
    // Sample line clearing experiences
    for (let i = 0; i < lineClearingCount; i++) {
      const randomIndex = Math.floor(Math.random() * lineClearingExperiences.length);
      batch.push(lineClearingExperiences[randomIndex]);
    }
    
    // Fill remaining with other experiences
    const remainingCount = this.batchSize - lineClearingCount;
    for (let i = 0; i < remainingCount && otherExperiences.length > 0; i++) {
      const randomIndex = Math.floor(Math.random() * otherExperiences.length);
      batch.push(otherExperiences[randomIndex]);
    }
    
    // If not enough experiences, fill with random samples
    while (batch.length < this.batchSize && this.memory.length > 0) {
      const randomIndex = Math.floor(Math.random() * this.memory.length);
      batch.push(this.memory[randomIndex]);
    }
    
    console.log(`🎨 CNN Training batch: ${lineClearingCount} line clearing + ${remainingCount} other experiences`);
    return batch;
  }

  updateTargetNetwork() {
    const weights = this.qNetwork.getWeights();
    this.targetNetwork.setWeights(weights);
  }

  startEpisode() {
    this.episode++;
    this.totalReward = 0;
  }

  endEpisode(finalReward = 0, finalScore = 0, environment = null) {
    this.totalReward += finalReward;
    this.rewards.push(this.totalReward);
    
    // Track REAL game score separately from AI rewards (like DQN)
    this.scores.push(finalScore);
    
    // Update best score with REAL game score
    if (finalScore > this.bestScore) {
      this.bestScore = finalScore;
      console.log(`🏆 NEW BEST REAL SCORE: ${this.bestScore} (Episode ${this.episode})`);
    }
    
    // Update curriculum if environment is provided (like DQN)
    let curriculumAdvanced = false;
    if (environment && environment.updateCurriculum) {
      curriculumAdvanced = environment.updateCurriculum(
        environment.patternsCompletedThisEpisode || 0, 
        finalScore
      );
      
      if (curriculumAdvanced) {
        console.log(`🎓 CURRICULUM ADVANCED! Now at level ${environment.curriculumLevel}: ${environment.currentComplexity} blocks`);
        
        // Reset some learning parameters for new curriculum level
        this.epsilon = Math.min(this.epsilon * 1.1, 0.8); // Increase exploration
        this.learningRate *= 0.98; // Slightly reduce learning rate
        
        // Recompile network with new learning rate
        const optimizer = tf.train.adam(this.learningRate);
        this.qNetwork.compile({
          optimizer: optimizer,
          loss: 'meanSquaredError',
          metrics: ['mse']
        });
      }
    }
    
    // Enhanced logging with REAL performance metrics (like DQN)
    const patternCount = environment ? environment.patternsCompletedThisEpisode || 0 : 0;
    const curriculumLevel = environment ? environment.curriculumLevel || 0 : 0;
    const gridSize = environment ? environment.currentGridSize || 12 : 12;
    
    console.log(`📊 Episode ${this.episode}: AI_Reward=${this.totalReward.toFixed(2)}, REAL_Score=${finalScore}, Best_Real=${this.bestScore}, Patterns=${patternCount}, Level=${curriculumLevel} (${gridSize}x${gridSize}), Success=${(this.recentLineClearSuccess * 100).toFixed(1)}%, Epsilon=${(this.epsilon * 100).toFixed(1)}%`);
    
    if (curriculumAdvanced) {
      console.log(`🎓 CURRICULUM ADVANCED to level ${curriculumLevel}!`);
    }
  }

  getStats() {
    const recentRewards = this.rewards.slice(-50);
    const avgReward = recentRewards.length > 0 ? 
      recentRewards.reduce((a, b) => a + b, 0) / recentRewards.length : 0;
    
    // Calculate average REAL game score for fair comparison (like DQN)
    const recentScores = this.scores.slice(-50);
    const avgScore = recentScores.length > 0 ? 
      recentScores.reduce((a, b) => a + b, 0) / recentScores.length : 0;
    
    const recentLosses = this.losses.slice(-50);
    const avgLoss = recentLosses.length > 0 ? 
      recentLosses.reduce((a, b) => a + b, 0) / recentLosses.length : 0;

    return {
      episode: this.episode,
      totalReward: this.totalReward,
      avgReward: avgReward,
      bestScore: this.bestScore, // REAL best game score
      avgScore: avgScore, // REAL average game score
      epsilon: this.epsilon,
      avgLoss: avgLoss,
      memorySize: this.memory.length,
      trainingSteps: this.trainingStep,
      rewards: this.rewards,
      scores: this.scores, // REAL game scores for visualization
      losses: this.losses,
      epsilonHistory: this.epsilonHistory,
      lineClearingSuccess: this.recentLineClearSuccess,
      patternRecognitions: this.visualPatternHistory.length,
      // ENSURE REAL PERFORMANCE IS CLEARLY MARKED
      realPerformance: {
        bestScore: this.bestScore,
        avgScore: avgScore,
        scores: this.scores.slice(-100)
      }
    };
  }

  async saveModel(name = 'visual-cnn-dqn') {
    try {
      await this.qNetwork.save(`localstorage://${name}`);
      
      const agentState = {
        epsilon: this.epsilon,
        episode: this.episode,
        trainingStep: this.trainingStep,
        rewards: this.rewards,
        scores: this.scores.slice(-500),
        losses: this.losses.slice(-500),
        epsilonHistory: this.epsilonHistory.slice(-500),
        bestScore: this.bestScore,
        visualPatternHistory: this.visualPatternHistory.slice(-50),
        recentLineClearSuccess: this.recentLineClearSuccess
      };
      
      localStorage.setItem(`${name}-state`, JSON.stringify(agentState));
      console.log(`Visual CNN model saved as ${name} (Patterns: ${this.visualPatternHistory.length})`);
      return true;
    } catch (error) {
      console.error('Error saving visual CNN model:', error);
      return false;
    }
  }

  async loadModel(name = 'visual-cnn-dqn') {
    try {
      this.qNetwork = await tf.loadLayersModel(`localstorage://${name}`);
      this.targetNetwork = this.buildConvolutionalNetwork();
      this.updateTargetNetwork();
      
      const agentStateStr = localStorage.getItem(`${name}-state`);
      if (agentStateStr) {
        const agentState = JSON.parse(agentStateStr);
        this.epsilon = agentState.epsilon || this.epsilon;
        this.episode = agentState.episode || 0;
        this.trainingStep = agentState.trainingStep || 0;
        this.rewards = agentState.rewards || [];
        this.scores = agentState.scores || [];
        this.losses = agentState.losses || [];
        this.epsilonHistory = agentState.epsilonHistory || [];
        this.bestScore = agentState.bestScore || 0;
        this.visualPatternHistory = agentState.visualPatternHistory || [];
        this.recentLineClearSuccess = agentState.recentLineClearSuccess || 0;
      }
      
      console.log(`Visual CNN model loaded: ${name} (Patterns: ${this.visualPatternHistory.length})`);
      return true;
    } catch (error) {
      console.error('Error loading visual CNN model:', error);
      return false;
    }
  }

  dispose() {
    this.qNetwork.dispose();
    this.targetNetwork.dispose();
    
    this.memory.forEach(exp => {
      if (exp.state && exp.state.dispose) exp.state.dispose();
      if (exp.nextState && exp.nextState.dispose) exp.nextState.dispose();
    });
    this.memory = [];
  }
} 