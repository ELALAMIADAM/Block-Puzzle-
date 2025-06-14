import * as tf from '@tensorflow/tfjs';

export class DQNAgent {
  constructor(stateSize, actionSize, options = {}) {
    this.stateSize = stateSize;
    this.actionSize = actionSize;
    
    // IMPROVED Hyperparameters optimized for dynamic action space
    this.learningRate = options.learningRate || 0.0005; // Reduced for more stable learning
    this.epsilon = options.epsilon || 0.9; // Start with less exploration
    this.epsilonMin = options.epsilonMin || 0.05; // Higher minimum for continued exploration
    this.epsilonDecay = options.epsilonDecay || 0.998; // Slower decay for better exploration
    this.gamma = options.gamma || 0.97; // Slightly higher discount for longer-term planning
    this.batchSize = options.batchSize || 64; // Larger batch for more stable gradients
    this.memorySize = options.memorySize || 15000; // Larger memory for better sampling
    this.targetUpdateFreq = options.targetUpdateFreq || 150; // Less frequent updates for stability
    
    // Experience replay memory
    this.memory = [];
    this.memoryCounter = 0;
    
    // Training counters
    this.trainingStep = 0;
    this.totalReward = 0;
    this.episode = 0;
    
    // Training lock to prevent concurrent training
    this.isTraining = false;
    
    // Neural networks
    this.qNetwork = this.buildNetwork();
    this.targetNetwork = this.buildNetwork();
    this.updateTargetNetwork();
    
    // Training metrics
    this.losses = [];
    this.rewards = [];
    this.epsilonHistory = [];
    this.bestScore = 0;
  }

  buildNetwork() {
    const model = tf.sequential({
      layers: [
        // Input layer - optimized for state size 112
        tf.layers.dense({
          inputShape: [this.stateSize],
          units: 256, // Reduced from 512 for faster training
          activation: 'relu',
          kernelInitializer: 'heUniform' // Better for ReLU activation
        }),
        
        // Hidden layers with batch normalization and dropout
        tf.layers.batchNormalization(),
        tf.layers.dropout({ rate: 0.15 }), // Reduced dropout for better learning
        tf.layers.dense({
          units: 256,
          activation: 'relu',
          kernelInitializer: 'heUniform'
        }),
        
        tf.layers.batchNormalization(),
        tf.layers.dropout({ rate: 0.15 }),
        tf.layers.dense({
          units: 128, // Slightly larger for better representation
          activation: 'relu',
          kernelInitializer: 'heUniform'
        }),
        
        tf.layers.dropout({ rate: 0.1 }),
        tf.layers.dense({
          units: 64, // Additional layer for more complex patterns
          activation: 'relu',
          kernelInitializer: 'heUniform'
        }),
        
        // Output layer - now much smaller due to optimized action space
        tf.layers.dense({
          units: this.actionSize, // Now ~50-150 instead of 243
          activation: 'linear',
          kernelInitializer: 'heUniform'
        })
      ]
    });

    // Improved optimizer with better hyperparameters
    model.compile({
      optimizer: tf.train.adam(this.learningRate, 0.9, 0.999, 1e-8), // Explicit beta values
      loss: tf.losses.huberLoss, // More stable than MSE for RL
      metrics: ['mse']
    });

    return model;
  }

  updateTargetNetwork() {
    // Copy weights from main network to target network
    const weights = this.qNetwork.getWeights();
    this.targetNetwork.setWeights(weights);
  }

  remember(state, action, reward, nextState, done) {
    const experience = {
      state: state.clone(),
      action,
      reward,
      nextState: nextState.clone(),
      done
    };
    
    // Accumulate reward for this episode
    this.totalReward += reward;
    
    // Debug logging for step rewards
    if (Math.abs(reward) > 0.1) {
      console.log(`🏆 Step reward: ${reward.toFixed(2)}, Episode total so far: ${this.totalReward.toFixed(2)}`);
    }
    
    if (this.memory.length < this.memorySize) {
      this.memory.push(experience);
    } else {
      // Circular buffer - overwrite oldest experience
      this.memory[this.memoryCounter % this.memorySize] = experience;
    }
    
    this.memoryCounter++;
  }

  async act(state, validActions = null) {
    // Epsilon-greedy action selection
    if (Math.random() <= this.epsilon) {
      // Random action (exploration)
      if (validActions && validActions.length > 0) {
        return validActions[Math.floor(Math.random() * validActions.length)];
      }
      return Math.floor(Math.random() * this.actionSize);
    }
    
    // Greedy action (exploitation)
    return await this.predict(state, validActions);
  }

  async predict(state, validActions = null) {
    const stateTensor = state.expandDims(0); // Add batch dimension
    const qValues = this.qNetwork.predict(stateTensor);
    
    let actionIndex;
    
    if (validActions && validActions.length > 0) {
      // IMPROVED: More efficient action masking with better performance
      const qArray = await qValues.data();
      let bestValue = -Infinity;
      let bestAction = validActions[0];
      
      // Only evaluate valid actions (much more efficient with smaller action space)
      for (const action of validActions) {
        // Handle action IDs that might be larger than network output size
        const networkActionIndex = action % this.actionSize;
        if (qArray[networkActionIndex] > bestValue) {
          bestValue = qArray[networkActionIndex];
          bestAction = action; // Return original action ID
        }
      }
      
      actionIndex = bestAction;
      
      // Debug: Log action selection for training analysis
      if (validActions.length > 1 && Math.random() < 0.01) { // 1% sampling
        console.log(`🎯 Action selection: ${validActions.length} valid actions, chose ${actionIndex} (Q-value: ${bestValue.toFixed(3)})`);
      }
    } else {
      // Fallback: No action masking - choose highest Q-value
      actionIndex = (await qValues.argMax(1).data())[0];
    }
    
    qValues.dispose();
    stateTensor.dispose();
    
    return actionIndex;
  }

  async replay() {
    if (this.memory.length < this.batchSize) {
      return;
    }

    // Prevent concurrent training calls
    if (this.isTraining) {
      console.log('⚠️ Training already in progress, skipping this replay call');
      return;
    }
    
    this.isTraining = true;

    try {
      // Sample random batch from memory
      const batch = this.sampleBatch();
    
    // Prepare training data
    const states = tf.stack(batch.map(exp => exp.state));
    const nextStates = tf.stack(batch.map(exp => exp.nextState));
    
    // Get current Q-values
    const currentQValues = this.qNetwork.predict(states);
    
    // Get target Q-values from target network
    const nextQValues = this.targetNetwork.predict(nextStates);
    const maxNextQValues = nextQValues.max(1);
    
    // Calculate target Q-values
    const targets = await currentQValues.array();
    const nextQArray = await maxNextQValues.array();
    
    for (let i = 0; i < batch.length; i++) {
      const exp = batch[i];
      let target = exp.reward;
      
      if (!exp.done) {
        target += this.gamma * nextQArray[i];
      }
      
      targets[i][exp.action] = target;
    }
    
    const targetTensor = tf.tensor2d(targets);
    
    // Train the network
    const history = await this.qNetwork.fit(states, targetTensor, {
      epochs: 1,
      verbose: 0,
      batchSize: this.batchSize
    });
    
    // Record loss
    this.losses.push(history.history.loss[0]);
    
    // Decay epsilon
    if (this.epsilon > this.epsilonMin) {
      this.epsilon *= this.epsilonDecay;
    }
    this.epsilonHistory.push(this.epsilon);
    
    // Update target network periodically
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
      // Always release training lock
      this.isTraining = false;
    }
  }

  sampleBatch() {
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
      console.log(`🏆 NEW BEST SCORE: ${this.bestScore} (Episode ${this.episode})`);
    }
    
    // Log episode statistics with more detail
    console.log(`📊 Episode ${this.episode} Complete: Total Reward = ${this.totalReward.toFixed(2)}, Score = ${finalScore}, Best Score = ${this.bestScore}, Epsilon = ${(this.epsilon * 100).toFixed(1)}%`);
    
    // Log additional stats every 10 episodes
    if (this.episode % 10 === 0) {
      const avgLast10 = this.rewards.slice(-10).reduce((a, b) => a + b, 0) / Math.min(10, this.rewards.length);
      console.log(`📈 Last 10 episodes average reward: ${avgLast10.toFixed(2)}`);
    }
  }

  getStats() {
    // Calculate average reward from last 100 episodes (or all if less than 100)
    const recentRewards = this.rewards.slice(-100);
    const avgReward = recentRewards.length > 0 ? 
      recentRewards.reduce((a, b) => a + b, 0) / recentRewards.length : 0;
    
    // Calculate average reward from last 10 episodes for quick feedback
    const last10Rewards = this.rewards.slice(-10);
    const avgReward10 = last10Rewards.length > 0 ?
      last10Rewards.reduce((a, b) => a + b, 0) / last10Rewards.length : 0;
    
    // Calculate average loss from recent training steps
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
      rewards: this.rewards, // Include full reward history for visualization
      losses: this.losses,   // Include full loss history for visualization
      epsilonHistory: this.epsilonHistory
    };
  }

  async saveModel(name = 'dqn-model') {
    try {
      await this.qNetwork.save(`localstorage://${name}`);
      
      // Save additional agent state
      const agentState = {
        epsilon: this.epsilon,
        episode: this.episode,
        trainingStep: this.trainingStep,
        rewards: this.rewards,
        losses: this.losses.slice(-1000), // Keep last 1000 losses
        epsilonHistory: this.epsilonHistory.slice(-1000),
        bestScore: this.bestScore
      };
      
      localStorage.setItem(`${name}-state`, JSON.stringify(agentState));
      console.log(`Model saved as ${name}`);
      return true;
    } catch (error) {
      console.error('Error saving model:', error);
      return false;
    }
  }

  async loadModel(name = 'dqn-model') {
    try {
      // Load the neural network
      this.qNetwork = await tf.loadLayersModel(`localstorage://${name}`);
      this.targetNetwork = this.buildNetwork();
      this.updateTargetNetwork();
      
      // Load additional agent state
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
      }
      
      console.log(`Model loaded: ${name}`);
      return true;
    } catch (error) {
      console.error('Error loading model:', error);
      return false;
    }
  }

  dispose() {
    // Clean up tensors and models
    this.qNetwork.dispose();
    this.targetNetwork.dispose();
    
    // Clean up memory
    this.memory.forEach(exp => {
      exp.state.dispose();
      exp.nextState.dispose();
    });
    this.memory = [];
  }

  // Prioritized experience replay (optional enhancement)
  rememberPrioritized(state, action, reward, nextState, done, priority = 1.0) {
    const experience = {
      state: state.clone(),
      action,
      reward,
      nextState: nextState.clone(),
      done,
      priority
    };
    
    // Insert based on priority (simplified implementation)
    if (this.memory.length < this.memorySize) {
      this.memory.push(experience);
      this.memory.sort((a, b) => b.priority - a.priority);
    } else {
      // Replace lowest priority if new experience has higher priority
      const minPriorityIndex = this.memory.length - 1;
      if (experience.priority > this.memory[minPriorityIndex].priority) {
        this.memory[minPriorityIndex].state.dispose();
        this.memory[minPriorityIndex].nextState.dispose();
        this.memory[minPriorityIndex] = experience;
        this.memory.sort((a, b) => b.priority - a.priority);
      }
    }
  }
} 