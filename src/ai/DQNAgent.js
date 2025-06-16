import * as tf from '@tensorflow/tfjs';

export class DQNAgent {
  constructor(stateSize, actionSize, options = {}) {
    this.stateSize = stateSize;
    this.actionSize = actionSize;
    
    // PROGRESSIVE LEARNING HYPERPARAMETERS
    this.learningRate = options.learningRate || 0.001; // Higher for faster initial learning
    this.epsilon = options.epsilon || 0.95; // Start with high exploration
    this.epsilonMin = options.epsilonMin || 0.01; // Low minimum for exploitation
    this.epsilonDecay = options.epsilonDecay || 0.995; // Decay rate
    this.gamma = options.gamma || 0.99; // High discount for planning
    this.batchSize = options.batchSize || 32; // Smaller batch for frequent updates
    this.memorySize = options.memorySize || 5000; // Smaller memory for curriculum
    this.targetUpdateFreq = options.targetUpdateFreq || 50; // Frequent updates
    
    // GUIDED EXPLORATION for line clearing
    this.guidedExploration = true;
    this.guidedExplorationRate = 0.3; // 30% guided vs random exploration
    
    // LINE CLEARING FOCUS
    this.lineClearingPriority = 10.0; // Massive priority for line clearing experiences
    this.lineClearingHistory = [];
    this.recentLineClearSuccess = 0;
    
    // Experience replay memory
    this.memory = [];
    this.memoryCounter = 0;
    
    // Training counters
    this.trainingStep = 0;
    this.totalReward = 0;
    this.episode = 0;
    
    // Training lock
    this.isTraining = false;
    
    // Build networks
    this.qNetwork = this.buildProgressiveNetwork();
    this.targetNetwork = this.buildProgressiveNetwork();
    this.updateTargetNetwork();
    
    // Training metrics
    this.losses = [];
    this.rewards = [];
    this.scores = []; // NEW: Track episode scores for visualization
    this.epsilonHistory = [];
    this.bestScore = 0;
  }

  /**
   * PROGRESSIVE NETWORK: Simpler architecture for curriculum learning
   */
  buildProgressiveNetwork() {
    const model = tf.sequential({
      layers: [
        // Input layer
        tf.layers.dense({
          inputShape: [this.stateSize],
          units: 128, // Smaller for faster learning
          activation: 'relu',
          kernelInitializer: 'heUniform'
        }),
        
        tf.layers.dropout({ rate: 0.2 }),
        
        // Hidden layer focusing on line patterns
        tf.layers.dense({
          units: 128,
          activation: 'relu',
          kernelInitializer: 'heUniform'
        }),
        
        tf.layers.dropout({ rate: 0.2 }),
        
        // Decision layer
        tf.layers.dense({
          units: 64,
          activation: 'relu',
          kernelInitializer: 'heUniform'
        }),
        
        // Output layer
        tf.layers.dense({
          units: this.actionSize,
          activation: 'linear',
          kernelInitializer: 'heUniform'
        })
      ]
    });
    
    const optimizer = tf.train.adam(this.learningRate);
    
    model.compile({
      optimizer: optimizer,
      loss: 'meanSquaredError',
      metrics: ['mse']
    });
    
    return model;
  }

  /**
   * GUIDED EXPLORATION: Bias toward line-completing moves
   */
  async act(state, validActions = null, environment = null) {
    // Use adaptive epsilon based on curriculum level and success
    let adaptiveEpsilon = this.epsilon;
    if (environment && environment.curriculumLevel !== undefined) {
      // Lower epsilon for higher curriculum levels (more exploitation)
      adaptiveEpsilon = this.epsilon * (1 - environment.curriculumLevel * 0.2);
    }
    
    // Epsilon-greedy with guided exploration
    if (Math.random() <= adaptiveEpsilon) {
      // GUIDED EXPLORATION: Bias toward line-completing moves
      if (this.guidedExploration && environment && Math.random() < this.guidedExplorationRate) {
        return this.selectGuidedAction(validActions, environment);
      } else {
        // Random exploration
        if (validActions && validActions.length > 0) {
          return validActions[Math.floor(Math.random() * validActions.length)];
        }
        return Math.floor(Math.random() * this.actionSize);
      }
    }
    
    // Exploitation
    return await this.predict(state, validActions);
  }

  /**
   * GUIDED EXPLORATION: Select actions that help complete lines
   */
  selectGuidedAction(validActions, environment) {
    if (!validActions || validActions.length === 0) {
      return 0;
    }
    
    let bestAction = validActions[0];
    let bestScore = -Infinity;
    
    // Evaluate each action for line completion potential
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
    
    console.log(`🎯 GUIDED EXPLORATION: Selected action with line score ${bestScore.toFixed(2)}`);
    return bestAction;
  }

  /**
   * GUIDED EXPLORATION: Evaluate how much an action helps complete lines
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
    
    // Massive bonus for completing lines
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
   * PRIORITIZED MEMORY: Massive priority for line clearing experiences
   */
  remember(state, action, reward, nextState, done) {
    const experience = {
      state: state.clone(),
      action,
      reward,
      nextState: nextState.clone(),
      done
    };
    
    // Store with prioritization
    if (this.memory.length < this.memorySize) {
      this.memory.push(experience);
    } else {
      // Overwrite oldest non-line-clearing experience first
      let replaceIndex = this.memoryCounter % this.memorySize;
      
      // If this is a line clearing experience, replace a non-line-clearing one
      if (reward > 1000) {
        for (let i = 0; i < this.memory.length; i++) {
          if (this.memory[i].reward < 1000) {
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
    
    // Track line clearing specifically
    if (reward > 1000) {
      this.lineClearingHistory.push({
        episode: this.episode,
        reward: reward,
        timestamp: Date.now()
      });
      
      console.log(`🎯 LINE CLEARING EXPERIENCE STORED! Reward: ${reward.toFixed(2)}, Episode: ${this.episode}`);
    }
    
    // Update recent success rate
    const recentHistory = this.lineClearingHistory.slice(-10);
    this.recentLineClearSuccess = recentHistory.length / 10;
    
    this.totalReward += reward;
  }

  /**
   * PROGRESSIVE TRAINING: Focus on line clearing experiences
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
      // Sample batch with line clearing bias
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
        
        // Map action to network index (use modulo to ensure it fits)
        const actionIndex = exp.action % this.actionSize;
        targets[i][actionIndex] = target;
      }
      
      const targetTensor = tf.tensor2d(targets);
      
      // Ensure target tensor has correct shape
      if (targetTensor.shape[1] !== this.actionSize) {
        console.error(`🚨 Target tensor shape mismatch: expected [${this.batchSize}, ${this.actionSize}], got [${targetTensor.shape[0]}, ${targetTensor.shape[1]}]`);
        targetTensor.dispose();
        states.dispose();
        nextStates.dispose();
        currentQValues.dispose();
        nextQValues.dispose();
        maxNextQValues.dispose();
        return;
      }
      
      // Train the network
      const history = await this.qNetwork.fit(states, targetTensor, {
        epochs: 1,
        verbose: 0,
        batchSize: this.batchSize
      });
      
      this.losses.push(history.history.loss[0]);
      
      // Adaptive epsilon decay based on line clearing success
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
      console.error('🚨 Training error:', error);
    } finally {
      this.isTraining = false;
    }
  }

  /**
   * PRIORITIZED SAMPLING: Bias toward line clearing experiences
   */
  sampleLineClearingBatch() {
    const batch = [];
    const lineClearingExperiences = this.memory.filter(exp => exp.reward > 1000);
    const otherExperiences = this.memory.filter(exp => exp.reward <= 1000);
    
    // Include more line clearing experiences
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
    
    console.log(`🎯 Training batch: ${lineClearingCount} line clearing + ${remainingCount} other experiences`);
    return batch;
  }

  updateTargetNetwork() {
    const weights = this.qNetwork.getWeights();
    this.targetNetwork.setWeights(weights);
  }

  async predict(state, validActions = null) {
    const stateInput = state.expandDims(0);
    const qValues = this.qNetwork.predict(stateInput);
    
    let actionIndex;
    
    if (validActions && validActions.length > 0) {
      const qArray = await qValues.data();
      let bestValue = -Infinity;
      let bestAction = validActions[0];
      
      // Map environment action IDs to network indices
      for (let i = 0; i < validActions.length; i++) {
        const action = validActions[i];
        const networkActionIndex = i % this.actionSize; // Use index in valid actions array
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
    stateInput.dispose();
    
    return actionIndex;
  }

  startEpisode() {
    this.episode++;
    this.totalReward = 0;
  }

  endEpisode(finalReward = 0, finalScore = 0, environment = null) {
    this.totalReward += finalReward;
    this.rewards.push(this.totalReward);
    
    // CRITICAL: Track REAL game score separately from AI rewards
    this.scores.push(finalScore); // Real game score for fair comparison
    
    // Update best score with REAL game score
    if (finalScore > this.bestScore) {
      this.bestScore = finalScore;
      console.log(`🏆 NEW BEST REAL SCORE: ${this.bestScore} (Episode ${this.episode})`);
    }
    
    // Update curriculum if environment is provided
    let curriculumAdvanced = false;
    if (environment && environment.updateCurriculum) {
      curriculumAdvanced = environment.updateCurriculum(
        environment.lineClearsThisEpisode || 0, 
        finalScore // Use real game score for curriculum
      );
      
      if (curriculumAdvanced) {
        console.log(`🎓 CURRICULUM ADVANCED! Now at level ${environment.curriculumLevel}: ${environment.currentGridSize}x${environment.currentGridSize}`);
        
        // Reset some learning parameters for new curriculum level
        this.epsilon = Math.min(this.epsilon * 1.1, 0.7); // Increase exploration for new level
        this.learningRate *= 0.95; // Slightly reduce learning rate
        
        // Recompile network with new learning rate
        const optimizer = tf.train.adam(this.learningRate);
        this.qNetwork.compile({
          optimizer: optimizer,
          loss: 'meanSquaredError',
          metrics: ['mse']
        });
      }
    }
    
    // Enhanced logging with REAL performance metrics
    const lineClearCount = environment ? environment.lineClearsThisEpisode || 0 : 0;
    const curriculumLevel = environment ? environment.curriculumLevel || 0 : 0;
    const gridSize = environment ? environment.currentGridSize || 9 : 9;
    
    console.log(`📊 Episode ${this.episode}: AI_Reward=${this.totalReward.toFixed(2)}, REAL_Score=${finalScore}, Best_Real=${this.bestScore}, Lines=${lineClearCount}, Level=${curriculumLevel} (${gridSize}x${gridSize}), Success=${(this.recentLineClearSuccess * 100).toFixed(1)}%, Epsilon=${(this.epsilon * 100).toFixed(1)}%`);
    
    if (curriculumAdvanced) {
      console.log(`🎓 CURRICULUM ADVANCED to level ${curriculumLevel}!`);
    }
  }

  getStats() {
    const recentRewards = this.rewards.slice(-50);
    const avgReward = recentRewards.length > 0 ? 
      recentRewards.reduce((a, b) => a + b, 0) / recentRewards.length : 0;
    
    // CRITICAL: Calculate average REAL game score for fair comparison
    const recentScores = this.scores.slice(-50);
    const avgScore = recentScores.length > 0 ? 
      recentScores.reduce((a, b) => a + b, 0) / recentScores.length : 0;
    
    const recentLosses = this.losses.slice(-50);
    const avgLoss = recentLosses.length > 0 ? 
      recentLosses.reduce((a, b) => a + b, 0) / recentLosses.length : 0;

    return {
      episode: this.episode,
      totalReward: this.totalReward, // AI internal reward
      avgReward: avgReward, // AI internal reward average
      bestScore: this.bestScore, // REAL best game score
      avgScore: avgScore, // REAL average game score - CRITICAL for fair comparison
      epsilon: this.epsilon,
      avgLoss: avgLoss,
      memorySize: this.memory.length,
      trainingSteps: this.trainingStep,
      rewards: this.rewards, // AI internal rewards
      scores: this.scores, // REAL game scores for visualization
      losses: this.losses,
      epsilonHistory: this.epsilonHistory,
      lineClearingSuccess: this.recentLineClearSuccess,
      lineClearingCount: this.lineClearingHistory.length,
      guidedExploration: this.guidedExploration,
      // ENSURE REAL PERFORMANCE IS CLEARLY MARKED
      realPerformance: {
        bestScore: this.bestScore,
        avgScore: avgScore,
        scores: this.scores.slice(-100) // Last 100 real scores
      }
    };
  }

  async saveModel(name = 'progressive-line-clearing-dqn') {
    try {
      await this.qNetwork.save(`localstorage://${name}`);
      
      const agentState = {
        epsilon: this.epsilon,
        episode: this.episode,
        trainingStep: this.trainingStep,
        rewards: this.rewards,
        scores: this.scores.slice(-500), // NEW: Save score history
        losses: this.losses.slice(-500),
        epsilonHistory: this.epsilonHistory.slice(-500),
        bestScore: this.bestScore,
        lineClearingHistory: this.lineClearingHistory.slice(-50),
        recentLineClearSuccess: this.recentLineClearSuccess
      };
      
      localStorage.setItem(`${name}-state`, JSON.stringify(agentState));
      console.log(`Progressive model saved as ${name} (LineClears: ${this.lineClearingHistory.length})`);
      return true;
    } catch (error) {
      console.error('Error saving progressive model:', error);
      return false;
    }
  }

  async loadModel(name = 'progressive-line-clearing-dqn') {
    try {
      this.qNetwork = await tf.loadLayersModel(`localstorage://${name}`);
      this.targetNetwork = this.buildProgressiveNetwork();
      this.updateTargetNetwork();
      
      const agentStateStr = localStorage.getItem(`${name}-state`);
      if (agentStateStr) {
        const agentState = JSON.parse(agentStateStr);
        this.epsilon = agentState.epsilon || this.epsilon;
        this.episode = agentState.episode || 0;
        this.trainingStep = agentState.trainingStep || 0;
        this.rewards = agentState.rewards || [];
        this.scores = agentState.scores || []; // NEW: Restore score history
        this.losses = agentState.losses || [];
        this.epsilonHistory = agentState.epsilonHistory || [];
        this.bestScore = agentState.bestScore || 0;
        this.lineClearingHistory = agentState.lineClearingHistory || [];
        this.recentLineClearSuccess = agentState.recentLineClearSuccess || 0;
      }
      
      console.log(`Progressive model loaded: ${name} (LineClears: ${this.lineClearingHistory.length})`);
      return true;
    } catch (error) {
      console.error('Error loading progressive model:', error);
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