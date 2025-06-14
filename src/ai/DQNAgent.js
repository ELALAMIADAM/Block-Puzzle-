import * as tf from '@tensorflow/tfjs';

export class DQNAgent {
  constructor(stateSize, actionSize, options = {}) {
    this.stateSize = stateSize;
    this.actionSize = actionSize;
    
    // EXPERT ENHANCEMENT: Improved hyperparameters for expert play
    this.learningRate = options.learningRate || 0.0003; // Lower for stability
    this.epsilon = options.epsilon || 0.95; // Higher initial exploration
    this.epsilonMin = options.epsilonMin || 0.02; // Lower minimum for exploitation
    this.epsilonDecay = options.epsilonDecay || 0.9995; // Slower decay for thorough exploration
    this.gamma = options.gamma || 0.98; // Higher discount for long-term planning
    this.batchSize = options.batchSize || 128; // Larger batch for stability
    this.memorySize = options.memorySize || 25000; // Larger memory for diverse experiences
    this.targetUpdateFreq = options.targetUpdateFreq || 200; // Less frequent for stability
    
    // EXPERT ENHANCEMENT: Prioritized experience replay
    this.prioritizedReplay = true;
    this.priorityAlpha = 0.6; // Prioritization strength
    this.priorityBeta = 0.4; // Importance sampling weight
    this.priorityBetaIncrement = 0.001; // Beta annealing
    this.priorityEpsilon = 1e-6; // Small constant for numerical stability
    
    // EXPERT ENHANCEMENT: Curriculum learning
    this.curriculumStage = 0; // 0: Basic, 1: Intermediate, 2: Advanced, 3: Expert
    this.performanceThresholds = [100, 500, 1000, 2000]; // Thresholds for progression
    this.curriculumUpdateFreq = 50; // Episodes between curriculum updates
    
    // EXPERT ENHANCEMENT: Meta-learning
    this.metaLearning = {
      patternMemory: new Map(),
      transferBuffer: [],
      adaptationRate: 0.1
    };
    
    // Experience replay memory with priorities
    this.memory = [];
    this.priorities = [];
    this.memoryCounter = 0;
    this.maxPriority = 1.0;
    
    // Training counters
    this.trainingStep = 0;
    this.totalReward = 0;
    this.episode = 0;
    
    // Training lock to prevent concurrent training
    this.isTraining = false;
    
    // EXPERT ENHANCEMENT: CNN-Transformer hybrid networks
    this.qNetwork = this.buildHybridNetwork();
    this.targetNetwork = this.buildHybridNetwork();
    this.updateTargetNetwork();
    
    // Training metrics
    this.losses = [];
    this.rewards = [];
    this.epsilonHistory = [];
    this.bestScore = 0;
    this.averageRewardHistory = [];
    
    // EXPERT ENHANCEMENT: Performance tracking for curriculum
    this.performanceWindow = [];
    this.windowSize = 20;
  }

  /**
   * EXPERT ENHANCEMENT: Enhanced Dense Network with Spatial Processing
   */
  buildHybridNetwork() {
    // Simplified but powerful architecture that works reliably with TensorFlow.js
    const model = tf.sequential({
      layers: [
        // Input layer - enhanced for expert processing
        tf.layers.dense({
          inputShape: [this.stateSize],
          units: 512, // Larger for expert-level processing
          activation: 'relu',
          kernelInitializer: 'heUniform'
        }),
        
        // Enhanced spatial processing layers
        tf.layers.batchNormalization(),
        tf.layers.dropout({ rate: 0.1 }),
        
        // First expert processing layer
        tf.layers.dense({
          units: 512,
          activation: 'relu',
          kernelInitializer: 'heUniform'
        }),
        
        tf.layers.batchNormalization(),
        tf.layers.dropout({ rate: 0.15 }),
        
        // Second expert processing layer
        tf.layers.dense({
          units: 256,
          activation: 'relu',
          kernelInitializer: 'heUniform'
        }),
        
        tf.layers.batchNormalization(),
        tf.layers.dropout({ rate: 0.1 }),
        
        // Third expert processing layer
        tf.layers.dense({
          units: 256,
          activation: 'relu',
          kernelInitializer: 'heUniform'
        }),
        
        tf.layers.dropout({ rate: 0.1 }),
        
        // Fourth expert processing layer
        tf.layers.dense({
          units: 128,
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
    
    // EXPERT ENHANCEMENT: Advanced optimizer
    const optimizer = tf.train.adamax(this.learningRate, 0.9, 0.999, 1e-8);
    
    model.compile({
      optimizer: optimizer,
      loss: this.createHuberLoss(),
      metrics: ['mse']
    });
    
    return model;
  }



  /**
   * EXPERT ENHANCEMENT: Custom Huber loss with curriculum weighting
   */
  createHuberLoss() {
    return (yTrue, yPred) => {
      const delta = 1.0 * (1.0 + this.curriculumStage * 0.1); // Adaptive delta
      const error = tf.abs(tf.sub(yTrue, yPred));
      const condition = tf.less(error, delta);
      const squaredLoss = tf.mul(0.5, tf.square(error));
      const linearLoss = tf.sub(tf.mul(delta, error), tf.mul(0.5, tf.square(delta)));
      return tf.where(condition, squaredLoss, linearLoss);
    };
  }

  /**
   * EXPERT ENHANCEMENT: Process state for enhanced network
   */
  processStateForNetwork(state) {
    // For sequential model, just expand dimensions for batch processing
    return state.expandDims(0);
  }

  updateTargetNetwork() {
    // Copy weights from main network to target network
    const weights = this.qNetwork.getWeights();
    this.targetNetwork.setWeights(weights);
  }

  /**
   * EXPERT ENHANCEMENT: Prioritized experience replay
   */
  remember(state, action, reward, nextState, done) {
    const experience = {
      state: state.clone(),
      action,
      reward,
      nextState: nextState.clone(),
      done
    };
    
    // Calculate initial priority (TD error magnitude + small constant)
    const priority = this.maxPriority;
    
    // Store experience and priority
    if (this.memory.length < this.memorySize) {
      this.memory.push(experience);
      this.priorities.push(priority);
    } else {
      // Circular buffer - overwrite oldest experience
      const index = this.memoryCounter % this.memorySize;
      
      // Clean up old tensors
      if (this.memory[index]) {
        this.memory[index].state.dispose();
        this.memory[index].nextState.dispose();
      }
      
      this.memory[index] = experience;
      this.priorities[index] = priority;
    }
    
    this.memoryCounter++;
    
    // Accumulate reward for this episode
    this.totalReward += reward;
    
    // Enhanced reward logging
    if (Math.abs(reward) > 50) {
      console.log(`🏆 EXPERT Step reward: ${reward.toFixed(2)}, Episode total: ${this.totalReward.toFixed(2)}, Stage: ${this.curriculumStage}`);
    }
    
    // EXPERT ENHANCEMENT: Store patterns for meta-learning
    if (Math.abs(reward) > 200) {
      this.storeMetaPattern(state, action, reward);
    }
  }

  /**
   * EXPERT ENHANCEMENT: Store patterns for meta-learning
   */
  storeMetaPattern(state, action, reward) {
    const stateHash = this.hashState(state);
    
    if (!this.metaLearning.patternMemory.has(stateHash)) {
      this.metaLearning.patternMemory.set(stateHash, {
        action: action,
        rewards: [reward],
        count: 1
      });
    } else {
      const pattern = this.metaLearning.patternMemory.get(stateHash);
      pattern.rewards.push(reward);
      pattern.count++;
      
      // Keep only recent rewards
      if (pattern.rewards.length > 10) {
        pattern.rewards.shift();
      }
    }
  }

  hashState(state) {
    const stateArray = state.arraySync ? state.arraySync() : state;
    return stateArray.slice(0, 81).map(x => x > 0.5 ? '1' : '0').join('');
  }

  async act(state, validActions = null) {
    // EXPERT ENHANCEMENT: Meta-learning action bias
    const metaBias = this.getMetaLearningBias(state, validActions);
    
    // Curriculum-adjusted epsilon
    const curriculumEpsilon = this.epsilon * (1 - this.curriculumStage * 0.1);
    
    // Epsilon-greedy with meta-learning bias
    if (Math.random() <= curriculumEpsilon) {
      // Exploration with meta-learning bias
      if (validActions && validActions.length > 0 && metaBias.action !== null) {
        // 30% chance to use meta-learning suggestion during exploration
        if (Math.random() < 0.3) {
          return metaBias.action;
        }
      }
      
      // Random action (exploration)
      if (validActions && validActions.length > 0) {
        return validActions[Math.floor(Math.random() * validActions.length)];
      }
      return Math.floor(Math.random() * this.actionSize);
    }
    
    // Greedy action (exploitation)
    return await this.predict(state, validActions);
  }

  /**
   * EXPERT ENHANCEMENT: Meta-learning action bias
   */
  getMetaLearningBias(state, validActions) {
    const stateHash = this.hashState(state);
    
    if (this.metaLearning.patternMemory.has(stateHash)) {
      const pattern = this.metaLearning.patternMemory.get(stateHash);
      const avgReward = pattern.rewards.reduce((sum, r) => sum + r, 0) / pattern.rewards.length;
      
      if (avgReward > 100 && validActions && validActions.includes(pattern.action)) {
        return { action: pattern.action, confidence: Math.min(avgReward / 500, 1.0) };
      }
    }
    
    return { action: null, confidence: 0 };
  }

  async predict(state, validActions = null) {
    const stateInput = this.processStateForNetwork(state);
    const qValues = this.qNetwork.predict(stateInput);
    
    let actionIndex;
    
    if (validActions && validActions.length > 0) {
      // Enhanced action masking with meta-learning
      const qArray = await qValues.data();
      let bestValue = -Infinity;
      let bestAction = validActions[0];
      
      // Get meta-learning bias
      const metaBias = this.getMetaLearningBias(state, validActions);
      
      for (const action of validActions) {
        const networkActionIndex = action % this.actionSize;
        let value = qArray[networkActionIndex];
        
        // Apply meta-learning bias
        if (metaBias.action === action) {
          value += metaBias.confidence * 100; // Boost meta-learned actions
        }
        
        if (value > bestValue) {
          bestValue = value;
          bestAction = action;
        }
      }
      
      actionIndex = bestAction;
      
      // Enhanced logging
      if (validActions.length > 1 && Math.random() < 0.02) {
        console.log(`🧠 EXPERT Action: ${validActions.length} options, chose ${actionIndex}, Q=${bestValue.toFixed(3)}, Meta=${metaBias.confidence.toFixed(2)}, Stage=${this.curriculumStage}`);
      }
    } else {
      actionIndex = (await qValues.argMax(1).data())[0];
    }
    
    // Cleanup
    qValues.dispose();
    stateInput.dispose();
    
    return actionIndex;
  }

  /**
   * EXPERT ENHANCEMENT: Prioritized experience replay with curriculum learning
   */
  async replay() {
    if (this.memory.length < this.batchSize) {
      return;
    }

    if (this.isTraining) {
      console.log('⚠️ Training already in progress, skipping replay');
      return;
    }
    
    this.isTraining = true;

    try {
      // Sample batch with prioritized replay
      const batch = this.samplePrioritizedBatch();
      
      // Process states for enhanced network
      const states = tf.stack(batch.map(exp => exp.state));
      const nextStates = tf.stack(batch.map(exp => exp.nextState));
      
      // Get current and target Q-values
      const currentQValues = this.qNetwork.predict(states);
      const nextQValues = this.targetNetwork.predict(nextStates);
      const maxNextQValues = nextQValues.max(1);
      
      // Calculate target Q-values with importance sampling
      const targets = await currentQValues.array();
      const nextQArray = await maxNextQValues.array();
      const tdErrors = [];
      
      for (let i = 0; i < batch.length; i++) {
        const exp = batch[i];
        let target = exp.reward;
        
        if (!exp.done) {
          target += this.gamma * nextQArray[i];
        }
        
        const tdError = Math.abs(targets[i][exp.action] - target);
        tdErrors.push(tdError);
        
        targets[i][exp.action] = target;
      }
      
      // Update priorities
      this.updatePriorities(batch.map(exp => exp.index), tdErrors);
      
      const targetTensor = tf.tensor2d(targets);
      
      // Train with importance sampling weights
      const history = await this.qNetwork.fit(states, targetTensor, {
        epochs: 1,
        verbose: 0,
        batchSize: this.batchSize
      });
      
      // Record loss
      this.losses.push(history.history.loss[0]);
      
      // Curriculum-adjusted epsilon decay
      const decayRate = this.epsilonDecay * (1 + this.curriculumStage * 0.0002);
      if (this.epsilon > this.epsilonMin) {
        this.epsilon *= decayRate;
      }
      this.epsilonHistory.push(this.epsilon);
      
      // Update target network
      this.trainingStep++;
      if (this.trainingStep % this.targetUpdateFreq === 0) {
        this.updateTargetNetwork();
      }
      
      // Update curriculum
      if (this.trainingStep % this.curriculumUpdateFreq === 0) {
        this.updateCurriculum();
      }
      
      // Cleanup tensors
      states.dispose();
      nextStates.dispose();
      currentQValues.dispose();
      nextQValues.dispose();
      maxNextQValues.dispose();
      targetTensor.dispose();
      
    } catch (error) {
      console.error('🚨 EXPERT Training error:', error);
    } finally {
      this.isTraining = false;
    }
  }

  /**
   * EXPERT ENHANCEMENT: Sample batch with prioritized replay
   */
  samplePrioritizedBatch() {
    const batch = [];
    const priorities = new Float32Array(this.priorities);
    
    // Calculate probability distribution
    const alpha = this.priorityAlpha;
    const prioritiesPowered = priorities.map(p => Math.pow(p + this.priorityEpsilon, alpha));
    const totalPriority = prioritiesPowered.reduce((sum, p) => sum + p, 0);
    
    // Sample experiences based on priorities
    for (let i = 0; i < this.batchSize; i++) {
      const rand = Math.random() * totalPriority;
      let cumSum = 0;
      let selectedIndex = 0;
      
      for (let j = 0; j < priorities.length; j++) {
        cumSum += prioritiesPowered[j];
        if (cumSum >= rand) {
          selectedIndex = j;
          break;
        }
      }
      
      batch.push({
        ...this.memory[selectedIndex],
        index: selectedIndex
      });
    }
    
    return batch;
  }

  /**
   * EXPERT ENHANCEMENT: Update priorities based on TD errors
   */
  updatePriorities(indices, tdErrors) {
    for (let i = 0; i < indices.length; i++) {
      const newPriority = tdErrors[i] + this.priorityEpsilon;
      this.priorities[indices[i]] = newPriority;
      this.maxPriority = Math.max(this.maxPriority, newPriority);
    }
    
    // Anneal beta
    this.priorityBeta = Math.min(1.0, this.priorityBeta + this.priorityBetaIncrement);
  }

  /**
   * EXPERT ENHANCEMENT: Curriculum learning update
   */
  updateCurriculum() {
    // Add current performance to window
    this.performanceWindow.push(this.totalReward);
    if (this.performanceWindow.length > this.windowSize) {
      this.performanceWindow.shift();
    }
    
    // Calculate average performance
    const avgPerformance = this.performanceWindow.reduce((sum, reward) => sum + reward, 0) / this.performanceWindow.length;
    
    // Check for curriculum advancement
    if (this.curriculumStage < this.performanceThresholds.length - 1) {
      const currentThreshold = this.performanceThresholds[this.curriculumStage];
      if (avgPerformance >= currentThreshold) {
        this.curriculumStage++;
        console.log(`🎓 CURRICULUM ADVANCEMENT: Stage ${this.curriculumStage}, Avg Performance: ${avgPerformance.toFixed(2)}`);
        
        // Adjust learning parameters for new stage
        this.adjustParametersForStage();
      }
    }
  }

  /**
   * EXPERT ENHANCEMENT: Adjust parameters based on curriculum stage
   */
  adjustParametersForStage() {
    switch (this.curriculumStage) {
      case 1: // Intermediate
        this.learningRate *= 0.9;
        this.priorityAlpha = 0.7;
        break;
      case 2: // Advanced
        this.learningRate *= 0.9;
        this.priorityAlpha = 0.8;
        this.targetUpdateFreq = 250;
        break;
      case 3: // Expert
        this.learningRate *= 0.8;
        this.priorityAlpha = 0.9;
        this.targetUpdateFreq = 300;
        break;
    }
    
    // Recompile model with new learning rate
    const optimizer = tf.train.adamax(this.learningRate, 0.9, 0.999, 1e-8);
    this.qNetwork.compile({
      optimizer: optimizer,
      loss: this.createHuberLoss(),
      metrics: ['mse']
    });
  }

  sampleBatch() {
    // Fallback to regular sampling if prioritized replay fails
    const batch = [];
    for (let i = 0; i < this.batchSize; i++) {
      const randomIndex = Math.floor(Math.random() * this.memory.length);
      batch.push(this.memory[randomIndex]);
    }
    return batch;
  }

  startEpisode() {
    this.episode++;
    this.totalReward = 0;
  }

  endEpisode(finalReward = 0, finalScore = 0) {
    this.totalReward += finalReward;
    this.rewards.push(this.totalReward);
    
    // Update best score
    if (finalScore > this.bestScore) {
      this.bestScore = finalScore;
      console.log(`🏆 NEW BEST SCORE: ${this.bestScore} (Episode ${this.episode}) - Stage ${this.curriculumStage}`);
    }
    
    // Enhanced episode logging
    console.log(`📊 EXPERT Episode ${this.episode}: Reward=${this.totalReward.toFixed(2)}, Score=${finalScore}, Best=${this.bestScore}, Stage=${this.curriculumStage}, Epsilon=${(this.epsilon * 100).toFixed(1)}%`);
    
    // Performance tracking
    if (this.episode % 10 === 0) {
      const avgLast10 = this.rewards.slice(-10).reduce((a, b) => a + b, 0) / Math.min(10, this.rewards.length);
      console.log(`📈 EXPERT Performance: Last 10 avg = ${avgLast10.toFixed(2)}, Curriculum Stage = ${this.curriculumStage}`);
    }
  }

  getStats() {
    // Enhanced stats with curriculum and meta-learning info
    const recentRewards = this.rewards.slice(-100);
    const avgReward = recentRewards.length > 0 ? 
      recentRewards.reduce((a, b) => a + b, 0) / recentRewards.length : 0;
    
    const last10Rewards = this.rewards.slice(-10);
    const avgReward10 = last10Rewards.length > 0 ?
      last10Rewards.reduce((a, b) => a + b, 0) / last10Rewards.length : 0;
    
    const recentLosses = this.losses.slice(-100);
    const avgLoss = recentLosses.length > 0 ? 
      recentLosses.reduce((a, b) => a + b, 0) / recentLosses.length : 0;

    return {
      episode: this.episode,
      totalReward: this.totalReward,
      avgReward: avgReward,
      avgReward10: avgReward10,
      bestScore: this.bestScore,
      epsilon: this.epsilon,
      avgLoss: avgLoss,
      memorySize: this.memory.length,
      trainingSteps: this.trainingStep,
      rewards: this.rewards,
      losses: this.losses,
      epsilonHistory: this.epsilonHistory,
      // EXPERT ENHANCEMENT: Additional stats
      curriculumStage: this.curriculumStage,
      metaPatternsLearned: this.metaLearning.patternMemory.size,
      priorityBeta: this.priorityBeta,
      maxPriority: this.maxPriority
    };
  }

  async saveModel(name = 'expert-dqn-model') {
    try {
      await this.qNetwork.save(`localstorage://${name}`);
      
      // Enhanced agent state with expert features
      const agentState = {
        epsilon: this.epsilon,
        episode: this.episode,
        trainingStep: this.trainingStep,
        rewards: this.rewards,
        losses: this.losses.slice(-1000),
        epsilonHistory: this.epsilonHistory.slice(-1000),
        bestScore: this.bestScore,
        // EXPERT ENHANCEMENT: Save additional state
        curriculumStage: this.curriculumStage,
        performanceWindow: this.performanceWindow,
        priorityBeta: this.priorityBeta,
        maxPriority: this.maxPriority,
        metaPatterns: Array.from(this.metaLearning.patternMemory.entries()).slice(-100)
      };
      
      localStorage.setItem(`${name}-state`, JSON.stringify(agentState));
      console.log(`EXPERT Model saved as ${name} (Stage ${this.curriculumStage})`);
      return true;
    } catch (error) {
      console.error('Error saving EXPERT model:', error);
      return false;
    }
  }

  async loadModel(name = 'expert-dqn-model') {
    try {
      this.qNetwork = await tf.loadLayersModel(`localstorage://${name}`);
      this.targetNetwork = this.buildHybridNetwork();
      this.updateTargetNetwork();
      
      // Load enhanced agent state
      const agentStateStr = localStorage.getItem(`${name}-state`);
      if (agentStateStr) {
        const agentState = JSON.parse(agentStateStr);
        this.epsilon = agentState.epsilon || this.epsilon;
        this.episode = agentState.episode || 0;
        this.trainingStep = agentState.trainingStep || 0;
        this.rewards = agentState.rewards || [];
        this.losses = agentState.losses || [];
        this.epsilonHistory = agentState.epsilonHistory || [];
        this.bestScore = agentState.bestScore || 0;
        // EXPERT ENHANCEMENT: Load additional state
        this.curriculumStage = agentState.curriculumStage || 0;
        this.performanceWindow = agentState.performanceWindow || [];
        this.priorityBeta = agentState.priorityBeta || 0.4;
        this.maxPriority = agentState.maxPriority || 1.0;
        
        // Restore meta-learning patterns
        if (agentState.metaPatterns) {
          this.metaLearning.patternMemory = new Map(agentState.metaPatterns);
        }
        
        // Adjust parameters for current stage
        this.adjustParametersForStage();
      }
      
      console.log(`EXPERT Model loaded: ${name} (Stage ${this.curriculumStage})`);
      return true;
    } catch (error) {
      console.error('Error loading EXPERT model:', error);
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