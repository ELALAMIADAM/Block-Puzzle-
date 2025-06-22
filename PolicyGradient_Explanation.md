# Policy Gradient Algorithm Explanation: Wood Block Puzzle AI

## Overview
This document explains the Policy Gradient (REINFORCE) algorithm implementation for the Wood Block Puzzle game. Policy Gradient is a reinforcement learning approach that directly optimizes a policy network to select actions, learning through gradient ascent on expected rewards.

## 1. Policy Gradient Fundamentals

### Core Concept
Unlike value-based methods (DQN), Policy Gradient directly learns a policy π(a|s) that maps states to action probabilities:
- **Policy Network**: Neural network that outputs action probabilities
- **Direct Optimization**: Updates policy parameters to increase expected reward
- **Stochastic Policy**: Samples actions from probability distribution
- **Gradient Ascent**: Maximizes expected cumulative reward

### REINFORCE Algorithm
The basic REINFORCE algorithm follows this process:
1. **Episode Collection**: Play full episodes using current policy
2. **Reward Calculation**: Compute discounted returns for each step
3. **Gradient Estimation**: Estimate policy gradient using episode data
4. **Policy Update**: Update network parameters using gradient ascent

## 2. Policy Network Architecture

### Network Structure
```javascript
buildPolicyNetwork() {
  const model = tf.sequential({
    layers: [
      // Input layer for game state
      tf.layers.dense({
        inputShape: [this.stateSize],
        units: 128,
        activation: 'relu',
        kernelInitializer: 'heUniform'
      }),
      
      tf.layers.dropout({ rate: 0.2 }),
      
      // Hidden layer for policy learning
      tf.layers.dense({
        units: 128,
        activation: 'relu',
        kernelInitializer: 'heUniform'
      }),
      
      tf.layers.dropout({ rate: 0.2 }),
      
      // Output layer with softmax for probabilities
      tf.layers.dense({
        units: this.actionSize,
        activation: 'softmax', // Ensures valid probability distribution
        kernelInitializer: 'heUniform'
      })
    ]
  });
  
  // Adam optimizer for stable gradient updates
  model.compile({
    optimizer: tf.train.adam(this.learningRate),
    loss: 'categoricalCrossentropy' // Will be overridden by custom loss
  });
  
  return model;
}
```

### Key Features
- **Softmax Output**: Ensures action probabilities sum to 1
- **Dropout Regularization**: Prevents overfitting during training
- **Adam Optimizer**: Adaptive learning rate for stable convergence
- **He Initialization**: Proper weight initialization for ReLU networks

## 3. Action Selection

### Stochastic Policy
Actions are sampled from the policy probability distribution:
```javascript
async selectAction(state, validActions = null) {
  // Forward pass through policy network
  const stateInput = state.expandDims(0);
  const actionProbs = this.policyNetwork.predict(stateInput);
  const probArray = await actionProbs.data();
  
  // Mask invalid actions
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
    // Uniform distribution fallback
    const uniformProb = 1.0 / validActions.length;
    for (const action of validActions) {
      const actionIndex = action % this.actionSize;
      validProbs[actionIndex] = uniformProb;
    }
  }
  
  // Sample action from distribution
  const selectedActionIndex = this.sampleFromDistribution(validProbs);
  const selectedAction = validActions.find(action => 
    (action % this.actionSize) === selectedActionIndex) || validActions[0];
  
  // Store for training
  this.episodeStates.push(state.clone());
  this.episodeActions.push(selectedActionIndex);
  
  return selectedAction;
}
```

### Probability Sampling
```javascript
sampleFromDistribution(probabilities) {
  const random = Math.random();
  let cumulative = 0;
  
  for (let i = 0; i < probabilities.length; i++) {
    cumulative += probabilities[i];
    if (random <= cumulative) {
      return i;
    }
  }
  
  // Fallback to last action
  return probabilities.length - 1;
}
```

### Action Masking Benefits
- **Valid Actions Only**: Prevents illegal moves by masking invalid actions
- **Probability Renormalization**: Ensures valid probability distribution
- **Graceful Fallback**: Uniform distribution when network outputs are poor
- **Dynamic Action Space**: Adapts to changing valid action sets

## 4. Episode Experience Collection

### Experience Storage
During each episode, collect:
```javascript
// In selectAction method
this.episodeStates.push(state.clone());      // Game states
this.episodeActions.push(selectedActionIndex); // Actions taken
// In remember method  
this.episodeRewards.push(reward);            // Rewards received
```

### Episode Data Structure
- **States**: Sequence of game states encountered
- **Actions**: Sequence of actions taken by policy
- **Rewards**: Sequence of rewards received from environment
- **Synchronized**: All arrays have same length for episode

## 5. Reward Processing

### Discounted Returns
Calculate discounted cumulative rewards for each timestep:
```javascript
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
```

### Reward Normalization
```javascript
// In train method
const discountedRewards = this.calculateDiscountedRewards();

// Normalize rewards to reduce variance
const mean = discountedRewards.reduce((sum, r) => sum + r, 0) / discountedRewards.length;
const std = Math.sqrt(discountedRewards.map(r => Math.pow(r - mean, 2))
  .reduce((sum, sq) => sum + sq, 0) / discountedRewards.length);
const normalizedRewards = discountedRewards.map(r => 
  std > 0 ? (r - mean) / std : r);
```

### Why Discounting and Normalization?
- **Discounting**: Future rewards matter less than immediate rewards
- **Variance Reduction**: Normalization reduces gradient variance
- **Stability**: Prevents gradient explosion from large reward values
- **Learning Speed**: Faster convergence with normalized targets

## 6. Policy Gradient Training

### REINFORCE Loss Function
The policy gradient theorem gives us the gradient:
```
∇θ J(θ) = E[∇θ log π(a|s) × R(τ)]
```

### Implementation
```javascript
async trainStep(states, actions, rewards) {
  const f = () => {
    const statesTensor = tf.stack(states);
    const actionsTensor = tf.tensor1d(actions, 'int32');
    const rewardsTensor = tf.tensor1d(rewards);
    
    // Get action probabilities from policy network
    const actionProbs = this.policyNetwork.apply(statesTensor);
    
    // Calculate log probabilities for selected actions
    const oneHotActions = tf.oneHot(actionsTensor, this.actionSize);
    const logProbs = tf.log(tf.sum(tf.mul(actionProbs, oneHotActions), 1));
    
    // Policy gradient loss: -log(prob) * reward
    const loss = tf.neg(tf.mean(tf.mul(logProbs, rewardsTensor)));
    
    // Add entropy bonus for exploration
    const entropy = tf.neg(tf.sum(tf.mul(actionProbs, 
      tf.log(tf.add(actionProbs, 1e-10))), 1));
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
  
  // Apply gradients using optimizer
  const { value: lossValue } = await this.policyNetwork.optimizer.minimize(f, true);
  this.losses.push(await lossValue.data());
  lossValue.dispose();
}
```

### Loss Components
1. **Policy Gradient Loss**: -log π(a|s) × R maximizes probability of good actions
2. **Entropy Bonus**: Encourages exploration by penalizing deterministic policies
3. **Gradient Ascent**: Minimizing negative loss = maximizing expected reward

## 7. Training Process

### Episode-Based Training
```javascript
async train() {
  if (this.episodeRewards.length === 0 || this.episodeStates.length === 0) {
    return;
  }
  
  try {
    // Calculate discounted rewards
    const discountedRewards = this.calculateDiscountedRewards();
    
    // Normalize rewards
    const mean = discountedRewards.reduce((sum, r) => sum + r, 0) / discountedRewards.length;
    const std = Math.sqrt(discountedRewards.map(r => Math.pow(r - mean, 2))
      .reduce((sum, sq) => sum + sq, 0) / discountedRewards.length);
    const normalizedRewards = discountedRewards.map(r => 
      std > 0 ? (r - mean) / std : r);
    
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
```

### Training Timing
Training occurs at the end of each episode in `endEpisode()`:
```javascript
async endEpisode(finalScore) {
  // IMPORTANT: Train the policy gradient at the end of each episode
  if (this.episodeRewards.length > 0 && this.episodeStates.length > 0) {
    await this.train();
  }
  
  // Track performance and update statistics
  // ...
}
```

## 8. Hyperparameter Configuration

### Key Parameters
```javascript
constructor(stateSize, actionSize, options = {}) {
  this.learningRate = options.learningRate || 0.001;  // Learning rate
  this.gamma = options.gamma || 0.99;                 // Discount factor
  this.entropyCoeff = options.entropyCoeff || 0.01;   // Exploration bonus
}
```

### Parameter Effects
- **Learning Rate**: Controls step size for policy updates
- **Gamma**: Determines importance of future vs immediate rewards
- **Entropy Coefficient**: Balances exploitation vs exploration

## 9. Exploration vs Exploitation

### Stochastic Policy Benefits
- **Natural Exploration**: Sampling from probabilities provides exploration
- **Adaptive Exploration**: Poor actions naturally get lower probability
- **No Epsilon-Greedy**: Exploration built into policy structure
- **Continuous Improvement**: Policy gradually shifts toward better actions

### Entropy Regularization
```javascript
// In trainStep method
const entropy = tf.neg(tf.sum(tf.mul(actionProbs, 
  tf.log(tf.add(actionProbs, 1e-10))), 1));
const entropyBonus = tf.mul(tf.scalar(this.entropyCoeff), tf.mean(entropy));
```

Benefits:
- **Prevents Premature Convergence**: Keeps policy stochastic
- **Encourages Exploration**: Rewards diverse action selection
- **Avoids Local Optima**: Helps escape suboptimal deterministic policies

## 10. Policy Gradient vs Value-Based Methods

### Advantages
- **Direct Policy Learning**: No need for value function approximation
- **Stochastic Policies**: Natural for games requiring randomization
- **Continuous Action Spaces**: Easily handles continuous actions
- **Convergence Guarantees**: Theoretical convergence to local optimum

### Disadvantages
- **High Variance**: Monte Carlo estimates have high variance
- **Sample Inefficiency**: Requires many episodes for good estimates
- **Slow Learning**: Converges slower than value-based methods
- **Local Optima**: May converge to suboptimal policies

## 11. Wood Block Puzzle Specific Features

### State Representation
Uses the same 139-feature state representation as other algorithms:
- 81 grid features
- 27 line completion features  
- 27 block analysis features
- 4 meta features

### Action Space Handling
- **Dynamic Action Masking**: Adapts to changing valid action sets
- **Probability Normalization**: Ensures valid probability distributions
- **Block Placement Focus**: Learns to prefer line-completing moves

### Reward Structure
- **Episode-Based**: Accumulates rewards throughout episode
- **Discounted Returns**: Values immediate rewards more than distant ones
- **Normalized Training**: Reduces variance in gradient estimates

## 12. Performance Characteristics

### Learning Behavior
- **Gradual Improvement**: Policy slowly shifts toward better actions
- **Exploration Maintenance**: Continues to explore even when confident
- **Episode-Level Learning**: Updates based on complete episode performance
- **Policy Smoothing**: Changes are gradual rather than abrupt

### Convergence Properties
- **Local Optima**: Guaranteed to converge to local optimum
- **Deterministic Limit**: May converge to deterministic policy
- **Plateau Behavior**: Can get stuck on performance plateaus
- **Restart Benefits**: Sometimes benefits from reinitialization

## Key Policy Gradient Innovations for Block Puzzle

1. **Action Masking**: Prevents illegal moves through probability masking
2. **Entropy Regularization**: Maintains exploration throughout training
3. **Reward Normalization**: Reduces gradient variance for stable learning
4. **Episode-Based Training**: Updates policy based on complete episodes
5. **Stochastic Policy**: Natural exploration without epsilon-greedy

This Policy Gradient implementation provides a direct approach to learning optimal play policies, offering theoretical guarantees and natural exploration while requiring careful tuning for optimal performance in the Wood Block Puzzle domain. 