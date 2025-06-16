import * as tf from '@tensorflow/tfjs';

/**
 * ADVANCED AI AGENTS - Enhanced algorithms for wood block puzzle
 * Features: MCTS, Policy Gradient, Hybrid Heuristic
 */

/**
 * MONTE CARLO TREE SEARCH AGENT
 * Uses tree search with rollouts for strategic decision making
 */
export class MCTSAgent {
  constructor(stateSize, actionSize, options = {}) {
    this.stateSize = stateSize;
    this.actionSize = actionSize;
    this.maxSimulations = options.simulations || options.maxSimulations || 50; // Use simulations parameter
    this.explorationConstant = options.explorationConstant || Math.sqrt(2);
    this.maxDepth = options.maxDepth || 10;
    this.rolloutPolicy = options.rolloutPolicy || 'heuristic';
    
    // Performance tracking
    this.episode = 0;
    this.scores = [];
    this.bestScore = 0;
    this.avgScore = 0;
    this.totalDecisions = 0;
    this.totalDecisionTime = 0;
    this.totalSimulations = 0;
    this.maxSearchDepth = 0;
    this.performanceHistory = [];
    
    console.log(`🌳 MCTS Agent initialized: ${this.maxSimulations} simulations, depth ${this.maxDepth}`);
  }

  async selectAction(environment) {
    const startTime = performance.now();
    
    if (!environment || typeof environment.getValidActions !== 'function') {
      console.error('❌ MCTS: Invalid environment provided');
      return null;
    }
    
    const validActions = environment.getValidActions();
    if (!validActions || validActions.length === 0) {
      return null;
    }
    
    if (validActions.length === 1) {
      const decisionTime = performance.now() - startTime;
      this.totalDecisionTime += decisionTime;
      this.totalDecisions++;
      return validActions[0];
    }
    
    // Create root node for current state
    const rootState = environment.cloneState();
    const rootNode = new MCTSNode(null, null, rootState);
    
    let simulationsRun = 0;
    const maxSimulations = this.maxSimulations; // Use configured simulations
    
    // Run MCTS simulations with periodic yields
    for (let i = 0; i < maxSimulations; i++) {
      const envClone = environment.clone();
      await this.runSimulation(rootNode, envClone);
      simulationsRun++;
      
      // Yield to event loop every 10 simulations to prevent UI freezing
      if (i % 10 === 0) {
        await new Promise(resolve => setTimeout(resolve, 0));
      }
      
      // Early termination if we find a clearly winning move
      if (rootNode.children.length > 0) {
        const bestChild = rootNode.children.reduce((best, child) => 
          child.visits > 5 && child.value / child.visits > 200 ? child : best
        );
        if (bestChild && bestChild.value / bestChild.visits > 500) {
          console.log(`🌳 MCTS early termination: found good move after ${i + 1} simulations`);
          break;
        }
      }
    }
    
    // Select best action based on visit count (exploitation)
    let bestAction = validActions[0];
    let bestVisits = 0;
    
    for (const child of rootNode.children) {
      if (child.visits > bestVisits) {
        bestVisits = child.visits;
        bestAction = child.action;
      }
    }
    
    // Update statistics
    const decisionTime = performance.now() - startTime;
    this.totalDecisionTime += decisionTime;
    this.totalDecisions++;
    this.totalSimulations += simulationsRun;
    this.maxSearchDepth = Math.max(this.maxSearchDepth, this.getMaxDepth(rootNode));
    
    console.log(`🌳 MCTS decision: Action ${bestAction}, ${simulationsRun} simulations, ${decisionTime.toFixed(1)}ms`);
    
    return bestAction;
  }

  async runSimulation(node, environment) {
    let currentNode = node;
    let depth = 0;
    
    // Selection: traverse tree using UCB1
    while (!currentNode.isLeaf() && depth < this.maxDepth) {
      currentNode = this.selectChild(currentNode);
      
      // Apply action to environment
      if (currentNode.action !== null) {
        const stepResult = environment.step(currentNode.action);
        if (stepResult.done) break;
      }
      
      depth++;
    }
    
    // Expansion: add new child if not terminal
    if (!environment.gameOver && depth < this.maxDepth) {
      const validActions = environment.getValidActions();
      if (validActions.length > 0) {
        currentNode = this.expandNode(currentNode, environment);
      }
    }
    
    // Simulation: rollout using heuristic policy
    const reward = await this.rollout(currentNode, environment);
    
    // Backpropagation: update values up the tree
    while (currentNode !== null) {
      currentNode.visits++;
      currentNode.value += reward;
      currentNode = currentNode.parent;
    }
  }

  selectChild(node) {
    let bestChild = null;
    let bestUCB = -Infinity;
    
    for (const child of node.children) {
      const exploitation = child.value / child.visits;
      const exploration = this.explorationConstant * Math.sqrt(Math.log(node.visits) / child.visits);
      const ucb = exploitation + exploration;
      
      if (ucb > bestUCB) {
        bestUCB = ucb;
        bestChild = child;
      }
    }
    
    return bestChild;
  }

  expandNode(node, environment) {
    const validActions = environment.getValidActions();
    
    // Add all valid actions as children
    for (const action of validActions) {
      if (!node.children.some(child => child.action === action)) {
        const newState = environment.cloneState();
        const childNode = new MCTSNode(node, action, newState);
        node.children.push(childNode);
      }
    }
    
    // Return random child for simulation
    if (node.children.length > 0) {
      return node.children[Math.floor(Math.random() * node.children.length)];
    }
    
    return node;
  }

  async rollout(node, environment) {
    let totalReward = 0;
    let steps = 0;
    const maxRolloutSteps = 10; // Reduce rollout steps to prevent freezing
    
    while (!environment.gameOver && steps < maxRolloutSteps) {
      const validActions = environment.getValidActions();
      if (validActions.length === 0) break;
      
      let action;
      if (this.rolloutPolicy === 'heuristic' && validActions.length <= 5) {
        // Only use heuristic for small action spaces to avoid computation
        action = this.selectHeuristicAction(validActions, environment);
      } else {
        // Use random policy for larger action spaces
        action = validActions[Math.floor(Math.random() * validActions.length)];
      }
      
      const stepResult = environment.step(action);
      totalReward += stepResult.reward;
      
      if (stepResult.done) break;
      
      // Add new blocks if needed (simplified)
      if (environment.availableBlocks.length === 0) {
        if (environment.generateCurriculumBlocks) {
          environment.availableBlocks = environment.generateCurriculumBlocks();
        } else {
          // Fallback to simple block generation
          environment.availableBlocks = [[[true]]]; // Simple 1x1 block
        }
      }
      
      steps++;
      
      // Yield occasionally during long rollouts
      if (steps % 5 === 0) {
        await new Promise(resolve => setTimeout(resolve, 0));
      }
    }
    
    return totalReward;
  }

  selectHeuristicAction(validActions, environment) {
    let bestAction = validActions[0];
    let bestScore = -Infinity;
    
    // Limit evaluation to prevent computation overload
    const actionsToEvaluate = Math.min(5, validActions.length);
    
    for (let i = 0; i < actionsToEvaluate; i++) {
      const action = validActions[i];
      
      // Simple heuristic: prefer actions that might complete lines
      let score = Math.random() * 10; // Base random score
      
      // Decode action to get placement info
      const { blockIndex, row, col } = environment.decodeAction ? 
        environment.decodeAction(action) : { blockIndex: 0, row: 0, col: 0 };
      
      // Simple line completion check
      if (blockIndex < environment.availableBlocks.length) {
        const block = environment.availableBlocks[blockIndex];
        if (block && environment.grid) {
          // Check if placement might complete a row
          if (row < environment.grid.length) {
            const rowFilled = environment.grid[row].filter(cell => cell).length;
            if (rowFilled >= environment.grid.length - 3) {
              score += 50; // Bonus for near-complete rows
            }
          }
        }
      }
      
      if (score > bestScore) {
        bestScore = score;
        bestAction = action;
      }
    }
    
    return bestAction;
  }

  evaluateLineCompletion(block, row, col, environment) {
    if (!block || !environment || !environment.grid) return 0;
    
    let score = 0;
    const grid = environment.grid;
    const size = grid.length;
    
    // Count potential line completions
    for (let r = 0; r < size; r++) {
      const filled = grid[r].filter(cell => cell).length;
      if (filled >= size - 2) score += 100; // Near complete row
    }
    
    for (let c = 0; c < size; c++) {
      let filled = 0;
      for (let r = 0; r < size; r++) {
        if (grid[r][c]) filled++;
      }
      if (filled >= size - 2) score += 100; // Near complete column
    }
    
    return score;
  }

  countNodes(node) {
    if (!node) return 0;
    let count = 1;
    for (const child of node.children) {
      count += this.countNodes(child);
    }
    return count;
  }

  getMaxDepth(node, depth = 0) {
    if (!node || node.children.length === 0) return depth;
    let maxDepth = depth;
    for (const child of node.children) {
      maxDepth = Math.max(maxDepth, this.getMaxDepth(child, depth + 1));
    }
    return maxDepth;
  }

  getStats() {
    const recentDecisionTimes = this.decisionTimes.slice(-50);
    const avgDecisionTime = recentDecisionTimes.length > 0 ? 
      recentDecisionTimes.reduce((a, b) => a + b, 0) / recentDecisionTimes.length : 0;
    
    const recentScores = this.scores.slice(-50);
    const avgScore = recentScores.length > 0 ? 
      recentScores.reduce((a, b) => a + b, 0) / recentScores.length : 0;

    return {
      algorithm: 'MCTS',
      episode: this.episode,
      bestScore: this.bestScore,
      avgScore: avgScore,
      scores: this.scores,
      rewards: this.rewards,
      avgDecisionTime: avgDecisionTime,
      avgSimulations: this.avgSimulations,
      nodeCount: this.nodeCount,
      searchDepth: this.searchDepth,
      memorySize: this.nodeCount,
      trainingSteps: 0, // MCTS doesn't train
      supportsTraining: false,
      supportsVisualization: true
    };
  }

  startEpisode() {
    this.episode++;
    this.decisionTimes = [];
  }

  endEpisode(finalScore) {
    this.episode++;
    
    // CRITICAL: Track REAL game score for fair comparison
    this.scores.push(finalScore);
    
    // Update best score with REAL game score
    if (finalScore > this.bestScore) {
      this.bestScore = finalScore;
      console.log(`🌳 MCTS NEW BEST REAL SCORE: ${this.bestScore} (Episode ${this.episode})`);
    }
    
    // Calculate average REAL game score
    const recentScores = this.scores.slice(-50);
    this.avgScore = recentScores.length > 0 ? 
      recentScores.reduce((a, b) => a + b, 0) / recentScores.length : 0;
    
    // Performance tracking
    this.performanceHistory.push({
      episode: this.episode,
      score: finalScore, // REAL game score
      avgDecisionTime: this.totalDecisionTime / Math.max(this.totalDecisions, 1),
      avgSimulations: this.totalSimulations / Math.max(this.totalDecisions, 1),
      timestamp: Date.now()
    });
    
    // Keep only recent history
    if (this.performanceHistory.length > 100) {
      this.performanceHistory = this.performanceHistory.slice(-50);
    }
    
    console.log(`🌳 MCTS Episode ${this.episode}: REAL_Score=${finalScore}, Best_Real=${this.bestScore}, Avg_Real=${this.avgScore.toFixed(1)}, Decisions=${this.totalDecisions}, AvgTime=${(this.totalDecisionTime / Math.max(this.totalDecisions, 1)).toFixed(1)}ms`);
    
    // Reset episode counters
    this.totalDecisions = 0;
    this.totalDecisionTime = 0;
    this.totalSimulations = 0;
  }

  dispose() {
    // MCTS doesn't have neural networks to dispose
  }
}

class MCTSNode {
  constructor(parent, action, state) {
    this.parent = parent;
    this.action = action;
    this.state = state;
    this.children = [];
    this.visits = 0;
    this.value = 0;
  }

  isLeaf() {
    return this.children.length === 0;
  }
}

/**
 * POLICY GRADIENT AGENT (REINFORCE)
 * Learns a policy network directly through gradient ascent
 */
export class PolicyGradientAgent {
  constructor(stateSize, actionSize, options = {}) {
    this.stateSize = stateSize;
    this.actionSize = actionSize;
    this.learningRate = options.learningRate || 0.001;
    this.gamma = options.gamma || 0.99;
    this.entropyCoeff = options.entropyCoeff || 0.01;
    
    // Training state
    this.episode = 0;
    this.scores = [];
    this.bestScore = 0;
    this.avgScore = 0;
    this.episodeStates = [];
    this.episodeActions = [];
    this.episodeRewards = [];
    this.losses = [];
    this.totalActions = 0;
    this.totalDecisions = 0;
    this.totalDecisionTime = 0;
    this.performanceHistory = [];
    
    // Build policy network
    this.policyNetwork = this.buildPolicyNetwork();
    
    console.log('📈 Policy Gradient Agent initialized with enhanced tracking');
  }

  buildPolicyNetwork() {
    const model = tf.sequential({
      layers: [
        tf.layers.dense({
          inputShape: [this.stateSize],
          units: 128,
          activation: 'relu',
          kernelInitializer: 'heUniform'
        }),
        
        tf.layers.dropout({ rate: 0.2 }),
        
        tf.layers.dense({
          units: 128,
          activation: 'relu',
          kernelInitializer: 'heUniform'
        }),
        
        tf.layers.dropout({ rate: 0.2 }),
        
        tf.layers.dense({
          units: 64,
          activation: 'relu',
          kernelInitializer: 'heUniform'
        }),
        
        // Output layer with softmax for action probabilities
        tf.layers.dense({
          units: this.actionSize,
          activation: 'softmax',
          kernelInitializer: 'heUniform'
        })
      ]
    });
    
    model.compile({
      optimizer: tf.train.adam(this.learningRate),
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy']
    });
    
    return model;
  }

  async selectAction(state, validActions = null) {
    if (!state || !state.expandDims) {
      console.error('❌ Policy Gradient: Invalid state tensor');
      return null;
    }
    
    if (!validActions || validActions.length === 0) {
      return null;
    }
    
    try {
      const stateInput = state.expandDims(0);
      const actionProbs = this.policyNetwork.predict(stateInput);
      const probArray = await actionProbs.data();
      
      // Create masked probabilities for valid actions only
      const validProbs = new Array(this.actionSize).fill(0);
      let totalValidProb = 0;
      
      for (const action of validActions) {
        const actionIndex = action % this.actionSize;
        validProbs[actionIndex] = probArray[actionIndex];
        totalValidProb += probArray[actionIndex];
      }
      
      // Normalize probabilities
      if (totalValidProb > 0) {
        for (let i = 0; i < validProbs.length; i++) {
          validProbs[i] /= totalValidProb;
        }
      } else {
        // Uniform distribution if no valid probabilities
        const uniformProb = 1.0 / validActions.length;
        for (const action of validActions) {
          const actionIndex = action % this.actionSize;
          validProbs[actionIndex] = uniformProb;
        }
      }
      
      // Sample action from probability distribution
      const selectedActionIndex = this.sampleFromDistribution(validProbs);
      const selectedAction = validActions.find(action => (action % this.actionSize) === selectedActionIndex) || validActions[0];
      
      // Store for training
      this.episodeStates.push(state.clone());
      this.episodeActions.push(selectedActionIndex);
      
      // Cleanup
      stateInput.dispose();
      actionProbs.dispose();
      
      return selectedAction;
      
    } catch (error) {
      console.error('❌ Policy Gradient action selection error:', error);
      return validActions[0]; // Fallback to first valid action
    }
  }

  sampleFromDistribution(probabilities) {
    const random = Math.random();
    let cumulative = 0;
    
    for (let i = 0; i < probabilities.length; i++) {
      cumulative += probabilities[i];
      if (random <= cumulative) {
        return i;
      }
    }
    
    // Fallback to last index
    return probabilities.length - 1;
  }

  remember(reward) {
    this.episodeRewards.push(reward);
  }

  async train() {
    if (this.episodeRewards.length === 0 || this.episodeStates.length === 0) {
      return;
    }
    
    try {
      // Calculate discounted rewards
      const discountedRewards = this.calculateDiscountedRewards();
      
      // Normalize rewards
      const mean = discountedRewards.reduce((sum, r) => sum + r, 0) / discountedRewards.length;
      const std = Math.sqrt(discountedRewards.map(r => Math.pow(r - mean, 2)).reduce((sum, sq) => sum + sq, 0) / discountedRewards.length);
      const normalizedRewards = discountedRewards.map(r => std > 0 ? (r - mean) / std : r);
      
      // Train the network
      await this.trainStep(this.episodeStates, this.episodeActions, normalizedRewards);
      
      console.log(`📈 Policy Gradient trained on ${this.episodeRewards.length} steps, avg reward: ${mean.toFixed(2)}`);
      
    } catch (error) {
      console.error('❌ Policy Gradient training error:', error);
    }
    
    // Clear episode data
    this.episodeStates.forEach(state => state.dispose());
    this.episodeStates = [];
    this.episodeActions = [];
    this.episodeRewards = [];
  }

  calculateDiscountedRewards() {
    const discounted = [];
    let runningAdd = 0;
    
    // Calculate discounted rewards backwards
    for (let i = this.episodeRewards.length - 1; i >= 0; i--) {
      runningAdd = runningAdd * this.gamma + this.episodeRewards[i];
      discounted.unshift(runningAdd);
    }
    
    return discounted;
  }

  async trainStep(states, actions, rewards) {
    const f = () => {
      const statesTensor = tf.stack(states);
      const actionsTensor = tf.tensor1d(actions, 'int32');
      const rewardsTensor = tf.tensor1d(rewards);
      
      // Get action probabilities
      const actionProbs = this.policyNetwork.apply(statesTensor);
      
      // Calculate log probabilities for selected actions
      const oneHotActions = tf.oneHot(actionsTensor, this.actionSize);
      const logProbs = tf.log(tf.sum(tf.mul(actionProbs, oneHotActions), 1));
      
      // Policy gradient loss: -log(prob) * reward
      const loss = tf.neg(tf.mean(tf.mul(logProbs, rewardsTensor)));
      
      // Add entropy bonus for exploration
      const entropy = tf.neg(tf.sum(tf.mul(actionProbs, tf.log(tf.add(actionProbs, 1e-10))), 1));
      const entropyBonus = tf.mul(tf.scalar(this.entropyCoeff), tf.mean(entropy));
      
      const totalLoss = tf.sub(loss, entropyBonus);
      
      // Cleanup intermediate tensors
      statesTensor.dispose();
      actionsTensor.dispose();
      rewardsTensor.dispose();
      actionProbs.dispose();
      oneHotActions.dispose();
      logProbs.dispose();
      loss.dispose();
      entropy.dispose();
      entropyBonus.dispose();
      
      return totalLoss;
    };
    
    const { value: lossValue } = await this.policyNetwork.optimizer.minimize(f, true);
    this.losses.push(await lossValue.data());
    lossValue.dispose();
  }

  startEpisode() {
    this.episode++;
    this.episodeStates = [];
    this.episodeActions = [];
    this.episodeRewards = [];
  }

  async endEpisode(finalScore) {
    this.episode++;
    
    // CRITICAL: Track REAL game score for fair comparison
    this.scores.push(finalScore);
    
    // Update best score with REAL game score
    if (finalScore > this.bestScore) {
      this.bestScore = finalScore;
      console.log(`🌳 MCTS NEW BEST REAL SCORE: ${this.bestScore} (Episode ${this.episode})`);
    }
    
    // Calculate average REAL game score
    const recentScores = this.scores.slice(-50);
    this.avgScore = recentScores.length > 0 ? 
      recentScores.reduce((a, b) => a + b, 0) / recentScores.length : 0;
    
    // Performance tracking
    this.performanceHistory.push({
      episode: this.episode,
      score: finalScore, // REAL game score
      avgDecisionTime: this.totalDecisionTime / Math.max(this.totalDecisions, 1),
      avgSimulations: this.totalSimulations / Math.max(this.totalDecisions, 1),
      timestamp: Date.now()
    });
    
    // Keep only recent history
    if (this.performanceHistory.length > 100) {
      this.performanceHistory = this.performanceHistory.slice(-50);
    }
    
    console.log(`📈 Policy Gradient Episode ${this.episode}: REAL_Score=${finalScore}, Best_Real=${this.bestScore}, Avg_Real=${this.avgScore.toFixed(1)}, Decisions=${this.totalDecisions}, AvgTime=${(this.totalDecisionTime / Math.max(this.totalDecisions, 1)).toFixed(1)}ms`);
    
    // Reset episode counters
    this.totalDecisions = 0;
    this.totalDecisionTime = 0;
    this.totalSimulations = 0;
  }

  getStats() {
    // Calculate average REAL game score
    const recentScores = this.scores.slice(-50);
    const avgScore = recentScores.length > 0 ? 
      recentScores.reduce((a, b) => a + b, 0) / recentScores.length : 0;

    return {
      episode: this.episode,
      bestScore: this.bestScore, // REAL best game score
      avgScore: avgScore, // REAL average game score - CRITICAL for fair comparison
      scores: this.scores, // REAL game scores for visualization
      totalDecisions: this.totalDecisions,
      avgDecisionTime: this.totalDecisions > 0 ? this.totalDecisionTime / this.totalDecisions : 0,
      avgEvaluations: this.totalDecisions > 0 ? this.totalEvaluations / this.totalDecisions : 0,
      lookaheadNodes: this.lookaheadNodes,
      performanceHistory: this.performanceHistory.slice(-20),
      // ENSURE REAL PERFORMANCE IS CLEARLY MARKED
      realPerformance: {
        bestScore: this.bestScore,
        avgScore: avgScore,
        scores: this.scores.slice(-100) // Last 100 real scores
      },
      // Algorithm-specific capabilities
      capabilities: {
        supportsTraining: true,
        supportsVisualization: true,
        requiresExploration: false,
        algorithmType: 'policy_gradient'
      }
    };
  }

  dispose() {
    if (this.policyNetwork) {
      this.policyNetwork.dispose();
    }
    
    this.episodeStates.forEach(state => {
      if (state && state.dispose) state.dispose();
    });
  }
}

/**
 * HYBRID HEURISTIC AGENT
 * Uses hand-crafted rules with lookahead search
 */
export class HybridHeuristicAgent {
  constructor(stateSize, actionSize, options = {}) {
    this.stateSize = stateSize;
    this.actionSize = actionSize;
    this.lookaheadDepth = options.lookaheadDepth || 2;
    this.maxEvaluations = options.maxEvaluations || 100;
    
    // Performance tracking
    this.episode = 0;
    this.scores = [];
    this.bestScore = 0;
    this.avgScore = 0;
    this.totalDecisions = 0;
    this.totalDecisionTime = 0;
    this.totalEvaluations = 0;
    this.lookaheadNodes = 0;
    this.performanceHistory = [];
    
    console.log('🧠 Heuristic Agent initialized with enhanced performance tracking');
  }

  async selectAction(environment) {
    const startTime = performance.now();
    
    if (!environment || typeof environment.getValidActions !== 'function') {
      console.error('❌ Heuristic: Invalid environment provided');
      return null;
    }
    
    const validActions = environment.getValidActions();
    if (!validActions || validActions.length === 0) {
      return null;
    }
    
    if (validActions.length === 1) {
      this.decisionTimes.push(performance.now() - startTime);
      return validActions[0];
    }
    
    let bestAction = validActions[0];
    let bestScore = -Infinity;
    let evaluations = 0;
    
    // Evaluate each action
    for (const action of validActions) {
      const score = this.evaluateAction(action, environment);
      evaluations++;
      
      if (score > bestScore) {
        bestScore = score;
        bestAction = action;
      }
    }
    
    // Update statistics
    const decisionTime = performance.now() - startTime;
    this.decisionTimes.push(decisionTime);
    this.evaluationCount = evaluations;
    
    console.log(`🧠 Heuristic decision: Action ${bestAction}, score ${bestScore.toFixed(1)}, ${evaluations} evaluations, ${decisionTime.toFixed(1)}ms`);
    
    return bestAction;
  }

  evaluateAction(action, environment) {
    const { blockIndex, row, col } = environment.decodeAction(action);
    
    if (blockIndex >= environment.availableBlocks.length) {
      return -Infinity;
    }
    
    const block = environment.availableBlocks[blockIndex];
    let totalScore = 0;
    
    // 1. Immediate line completion reward
    const lineScore = this.countCompletedLines(block, row, col, environment);
    totalScore += lineScore * this.lineCompletionWeight;
    
    // 2. Spatial positioning quality
    const spatialScore = this.evaluateSpatialPositioning(block, row, col, environment);
    totalScore += spatialScore * this.spatialEfficiencyWeight;
    
    // 3. Future opportunity creation
    const futureScore = this.evaluateFutureOpportunities(block, row, col, environment);
    totalScore += futureScore * this.futureOpportunityWeight;
    
    // 4. Lookahead evaluation
    if (this.lookaheadDepth > 0) {
      const lookaheadScore = this.lookaheadEvaluation(action, environment, this.lookaheadDepth);
      totalScore += lookaheadScore * 0.3; // Weight lookahead less than immediate rewards
    }
    
    return totalScore;
  }

  countCompletedLines(block, row, col, environment) {
    if (!block || !environment.grid) return 0;
    
    const grid = environment.grid.map(r => [...r]); // Clone grid
    const gridSize = grid.length;
    
    // Simulate placing the block
    for (let r = 0; r < block.length; r++) {
      for (let c = 0; c < block[r].length; c++) {
        if (block[r][c] && row + r < gridSize && col + c < gridSize) {
          grid[row + r][col + c] = true;
        }
      }
    }
    
    let completedLines = 0;
    
    // Check rows
    for (let r = 0; r < gridSize; r++) {
      if (grid[r].every(cell => cell)) {
        completedLines++;
      }
    }
    
    // Check columns
    for (let c = 0; c < gridSize; c++) {
      let complete = true;
      for (let r = 0; r < gridSize; r++) {
        if (!grid[r][c]) {
          complete = false;
          break;
        }
      }
      if (complete) completedLines++;
    }
    
    // Check 3x3 squares if environment supports it
    if (environment.constructor.name === 'EliteEnvironment') {
      for (let sq_row = 0; sq_row < 3; sq_row++) {
        for (let sq_col = 0; sq_col < 3; sq_col++) {
          let complete = true;
          for (let r = sq_row * 3; r < (sq_row + 1) * 3; r++) {
            for (let c = sq_col * 3; c < (sq_col + 1) * 3; c++) {
              if (!grid[r][c]) {
                complete = false;
                break;
              }
            }
            if (!complete) break;
          }
          if (complete) completedLines++;
        }
      }
    }
    
    return completedLines;
  }

  evaluateSpatialPositioning(block, row, col, environment) {
    if (!block || !environment.grid) return 0;
    
    const grid = environment.grid;
    const gridSize = grid.length;
    let score = 0;
    
    // Bonus for utilizing edges and corners
    if (row === 0 || row + block.length === gridSize) score += 20; // Edge row
    if (col === 0 || col + block[0].length === gridSize) score += 20; // Edge col
    
    // Bonus for connecting to existing blocks
    let connections = 0;
    for (let r = 0; r < block.length; r++) {
      for (let c = 0; c < block[r].length; c++) {
        if (block[r][c]) {
          const gr = row + r;
          const gc = col + c;
          
          // Check adjacent cells
          const adjacent = [
            [gr-1, gc], [gr+1, gc], [gr, gc-1], [gr, gc+1]
          ];
          
          for (const [ar, ac] of adjacent) {
            if (ar >= 0 && ar < gridSize && ac >= 0 && ac < gridSize && grid[ar][ac]) {
              connections++;
            }
          }
        }
      }
    }
    
    score += connections * 10;
    
    // Penalty for creating isolated cells
    let isolation = 0;
    for (let r = 0; r < block.length; r++) {
      for (let c = 0; c < block[r].length; c++) {
        if (block[r][c]) {
          const gr = row + r;
          const gc = col + c;
          
          const adjacent = [
            [gr-1, gc], [gr+1, gc], [gr, gc-1], [gr, gc+1]
          ];
          
          let adjacentFilled = 0;
          for (const [ar, ac] of adjacent) {
            if (ar >= 0 && ar < gridSize && ac >= 0 && ac < gridSize && grid[ar][ac]) {
              adjacentFilled++;
            }
          }
          
          if (adjacentFilled === 0) isolation++;
        }
      }
    }
    
    score -= isolation * 15;
    
    return score;
  }

  evaluateFutureOpportunities(block, row, col, environment) {
    if (!block || !environment.grid) return 0;
    
    const grid = environment.grid.map(r => [...r]); // Clone grid
    const gridSize = grid.length;
    
    // Simulate placing the block
    for (let r = 0; r < block.length; r++) {
      for (let c = 0; c < block[r].length; c++) {
        if (block[r][c] && row + r < gridSize && col + c < gridSize) {
          grid[row + r][col + c] = true;
        }
      }
    }
    
    let opportunityScore = 0;
    
    // Count near-complete lines (within 1-2 cells of completion)
    for (let r = 0; r < gridSize; r++) {
      const filled = grid[r].filter(cell => cell).length;
      if (filled >= gridSize - 2) opportunityScore += 50;
      else if (filled >= gridSize - 3) opportunityScore += 25;
    }
    
    for (let c = 0; c < gridSize; c++) {
      let filled = 0;
      for (let r = 0; r < gridSize; r++) {
        if (grid[r][c]) filled++;
      }
      if (filled >= gridSize - 2) opportunityScore += 50;
      else if (filled >= gridSize - 3) opportunityScore += 25;
    }
    
    return opportunityScore;
  }

  lookaheadEvaluation(action, environment, depth) {
    if (depth <= 0) return 0;
    
    try {
      const envClone = environment.clone();
      const stepResult = envClone.step(action);
      
      if (stepResult.done) {
        return stepResult.reward;
      }
      
      // Look ahead to next possible moves
      const nextValidActions = envClone.getValidActions();
      if (nextValidActions.length === 0) {
        return stepResult.reward;
      }
      
      let bestFutureScore = -Infinity;
      const actionsToTry = Math.min(3, nextValidActions.length); // Limit for performance
      
      for (let i = 0; i < actionsToTry; i++) {
        const nextAction = nextValidActions[i];
        const futureScore = this.lookaheadEvaluation(nextAction, envClone, depth - 1);
        bestFutureScore = Math.max(bestFutureScore, futureScore);
      }
      
      this.lookaheadNodes++;
      return stepResult.reward + bestFutureScore * 0.7; // Discount future rewards
      
    } catch (error) {
      console.warn('⚠️ Heuristic lookahead error:', error.message);
      return 0;
    }
  }

  startEpisode() {
    this.episode++;
    this.decisionTimes = [];
    this.lookaheadNodes = 0;
  }

  endEpisode(finalScore) {
    this.episode++;
    
    // CRITICAL: Track REAL game score for fair comparison
    this.scores.push(finalScore);
    
    // Update best score with REAL game score
    if (finalScore > this.bestScore) {
      this.bestScore = finalScore;
      console.log(`🧠 Heuristic NEW BEST REAL SCORE: ${this.bestScore} (Episode ${this.episode})`);
    }
    
    // Calculate average REAL game score
    const recentScores = this.scores.slice(-50);
    this.avgScore = recentScores.length > 0 ? 
      recentScores.reduce((a, b) => a + b, 0) / recentScores.length : 0;
    
    // Performance tracking
    this.performanceHistory.push({
      episode: this.episode,
      score: finalScore, // REAL game score
      avgDecisionTime: this.totalDecisionTime / Math.max(this.totalDecisions, 1),
      avgSimulations: this.totalSimulations / Math.max(this.totalDecisions, 1),
      timestamp: Date.now()
    });
    
    // Keep only recent history
    if (this.performanceHistory.length > 100) {
      this.performanceHistory = this.performanceHistory.slice(-50);
    }
    
    console.log(`🧠 Heuristic Episode ${this.episode}: REAL_Score=${finalScore}, Best_Real=${this.bestScore}, Avg_Real=${this.avgScore.toFixed(1)}, Decisions=${this.totalDecisions}, AvgTime=${(this.totalDecisionTime / Math.max(this.totalDecisions, 1)).toFixed(1)}ms`);
    
    // Reset episode counters
    this.totalDecisions = 0;
    this.totalDecisionTime = 0;
    this.totalSimulations = 0;
  }

  getStats() {
    // Calculate average REAL game score
    const recentScores = this.scores.slice(-50);
    const avgScore = recentScores.length > 0 ? 
      recentScores.reduce((a, b) => a + b, 0) / recentScores.length : 0;

    return {
      episode: this.episode,
      bestScore: this.bestScore, // REAL best game score
      avgScore: avgScore, // REAL average game score - CRITICAL for fair comparison
      scores: this.scores, // REAL game scores for visualization
      totalDecisions: this.totalDecisions,
      avgDecisionTime: this.totalDecisions > 0 ? this.totalDecisionTime / this.totalDecisions : 0,
      avgEvaluations: this.totalDecisions > 0 ? this.totalEvaluations / this.totalDecisions : 0,
      lookaheadNodes: this.lookaheadNodes,
      performanceHistory: this.performanceHistory.slice(-20),
      // ENSURE REAL PERFORMANCE IS CLEARLY MARKED
      realPerformance: {
        bestScore: this.bestScore,
        avgScore: avgScore,
        scores: this.scores.slice(-100) // Last 100 real scores
      },
      // Algorithm-specific capabilities
      capabilities: {
        supportsTraining: false,
        supportsVisualization: true,
        requiresExploration: false,
        algorithmType: 'heuristic'
      }
    };
  }

  dispose() {
    // Heuristic agent has no resources to dispose
  }
}

/**
 * ALGORITHM SELECTOR
 * 
 * Allows switching between different algorithms
 */
export class AlgorithmSelector {
  static createAgent(algorithm, stateSize, actionSize, options = {}) {
    switch (algorithm.toLowerCase()) {
      case 'mcts':
        return new MCTSAgent(stateSize, actionSize, options);
      case 'policy_gradient':
      case 'policy-gradient':
      case 'reinforce':
        return new PolicyGradientAgent(stateSize, actionSize, options);
      case 'heuristic':
      case 'hybrid':
        return new HybridHeuristicAgent(stateSize, actionSize, options);
      default:
        throw new Error(`Unknown algorithm: ${algorithm}`);
    }
  }

  static getAvailableAlgorithms() {
    return [
      {
        name: 'mcts',
        displayName: 'Monte Carlo Tree Search',
        description: 'Tree search with rollouts - excellent for sparse rewards',
        strengths: ['Handles sparse rewards', 'No training required', 'Robust decisions'],
        weaknesses: ['Computationally intensive', 'Slower per move'],
        supportsTraining: false,
        supportsVisualization: true
      },
      {
        name: 'policy_gradient',
        displayName: 'Policy Gradient (REINFORCE)',
        description: 'Direct policy optimization - learns action preferences',
        strengths: ['Direct policy learning', 'Good for large action spaces', 'Stable training'],
        weaknesses: ['Requires training', 'High variance'],
        supportsTraining: true,
        supportsVisualization: true
      },
      {
        name: 'heuristic',
        displayName: 'Hybrid Heuristic',
        description: 'Hand-crafted rules with lookahead - fast and effective',
        strengths: ['Very fast', 'No training needed', 'Interpretable decisions'],
        weaknesses: ['Limited by human knowledge', 'May miss complex patterns'],
        supportsTraining: false,
        supportsVisualization: true
      }
    ];
  }
} 