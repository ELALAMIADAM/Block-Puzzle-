# Monte Carlo Tree Search Algorithm Explanation: Wood Block Puzzle

## Overview
This document explains the Monte Carlo Tree Search (MCTS) algorithm implementation for the Wood Block Puzzle game. MCTS is a strategic decision-making algorithm that uses tree search combined with Monte Carlo simulations to find optimal moves without requiring training data or neural networks.

## 1. MCTS Algorithm Fundamentals

### Core MCTS Process
MCTS operates through four main phases repeated iteratively:
1. **Selection**: Navigate down the tree using UCB1 formula
2. **Expansion**: Add new child nodes to expand the search tree
3. **Simulation**: Run random rollouts from new nodes to estimate value
4. **Backpropagation**: Update statistics back up the tree path

### UCB1 Formula
The Upper Confidence Bound formula balances exploration vs exploitation:
```
UCB1 = Q(s,a) + C × √(ln(N(s)) / N(s,a))
```
- Q(s,a): Average reward for state-action pair
- C: Exploration constant (√2 ≈ 1.414)
- N(s): Parent node visit count
- N(s,a): Child node visit count

## 2. MCTS Node Structure

### MCTSNode Class
Each node in the search tree contains:
- **Parent**: Reference to parent node
- **Action**: Action taken to reach this node
- **State**: Game state representation
- **Children**: Array of child nodes
- **Visits**: Number of times node was visited
- **Value**: Cumulative reward from all visits

### Tree Structure
```
Root (Current Game State)
├── Action A1 → Child Node 1
│   ├── Action A1.1 → Grandchild 1.1
│   └── Action A1.2 → Grandchild 1.2
├── Action A2 → Child Node 2
└── Action A3 → Child Node 3
```

## 3. Selection Phase

### UCB1-Based Selection
Starting from root node, repeatedly select child with highest UCB1 value:
```javascript
selectChild(node) {
  let bestChild = null;
  let bestUCB = -Infinity;
  
  for (const child of node.children) {
    const exploitation = child.value / child.visits;
    const exploration = this.explorationConstant * 
      Math.sqrt(Math.log(node.visits) / child.visits);
    const ucb = exploitation + exploration;
    
    if (ucb > bestUCB) {
      bestUCB = ucb;
      bestChild = child;
    }
  }
  
  return bestChild;
}
```

### Balancing Exploration vs Exploitation
- **High UCB1**: Promising actions or under-explored actions
- **Exploitation**: Focuses on actions with high average rewards
- **Exploration**: Encourages trying less-visited actions
- **Dynamic Balance**: Changes as tree grows and statistics improve

## 4. Expansion Phase

### Adding New Nodes
When reaching a leaf node that isn't terminal:
```javascript
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
  return node.children[Math.floor(Math.random() * node.children.length)];
}
```

### Expansion Strategy
- **All Valid Actions**: Creates child for every legal move
- **State Cloning**: Each child maintains its own game state
- **Random Selection**: Randomly chooses which new child to simulate
- **Incremental Growth**: Tree expands gradually over iterations

## 5. Simulation Phase (Rollout)

### Monte Carlo Rollouts
From expanded node, play game to completion using rollout policy:
```javascript
async rollout(node, environment) {
  let totalReward = 0;
  let steps = 0;
  const maxRolloutSteps = 10; // Prevent infinite rollouts
  
  while (!environment.gameOver && steps < maxRolloutSteps) {
    const validActions = environment.getValidActions();
    if (validActions.length === 0) break;
    
    let action;
    if (this.rolloutPolicy === 'heuristic') {
      action = this.selectHeuristicAction(validActions, environment);
    } else {
      action = validActions[Math.floor(Math.random() * validActions.length)];
    }
    
    const stepResult = environment.step(action);
    totalReward += stepResult.reward;
    
    if (stepResult.done) break;
    steps++;
  }
  
  return totalReward;
}
```

### Rollout Policies
1. **Random Policy**: Uniformly random action selection
2. **Heuristic Policy**: Uses domain knowledge for better rollouts
   - Prioritizes line completion
   - Considers spatial efficiency
   - Avoids obviously bad moves

### Heuristic Rollout Enhancement
```javascript
selectHeuristicAction(validActions, environment) {
  let bestAction = validActions[0];
  let bestScore = 0;
  
  for (const action of validActions) {
    let score = Math.random() * 10; // Base random score
    
    // Decode action to get placement info
    const { blockIndex, row } = environment.decodeAction(action);
    
    // Check if placement might complete a row
    if (blockIndex < environment.availableBlocks.length) {
      const block = environment.availableBlocks[blockIndex];
      if (block && environment.grid && row < environment.grid.length) {
        const rowFilled = environment.grid[row].filter(cell => cell).length;
        if (rowFilled >= environment.grid.length - 3) {
          score += 50; // Bonus for near-complete rows
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
```

## 6. Backpropagation Phase

### Updating Tree Statistics
After rollout completes, update all nodes on path from leaf to root:
```javascript
// In runSimulation method
while (currentNode !== null) {
  currentNode.visits++;
  currentNode.value += reward;
  currentNode = currentNode.parent;
}
```

### Statistical Updates
- **Visit Count**: Increment for every node on path
- **Value Accumulation**: Add rollout reward to cumulative value
- **Average Calculation**: Q(s,a) = total_value / visit_count
- **Parent Chain**: Update continues until reaching root

## 7. MCTS Configuration

### Hyperparameters
```javascript
constructor(stateSize, actionSize, options = {}) {
  this.maxSimulations = options.simulations || 50;
  this.explorationConstant = options.explorationConstant || Math.sqrt(2);
  this.maxDepth = options.maxDepth || 10;
  this.rolloutPolicy = options.rolloutPolicy || 'heuristic';
}
```

### Parameter Tuning
- **Simulations**: 50 simulations balances quality vs speed
- **Exploration Constant**: √2 is theoretically optimal
- **Max Depth**: Prevents infinite tree growth
- **Rollout Policy**: Heuristic provides better value estimates

## 8. Action Selection

### Best Action Selection
After all simulations, select action with highest visit count:
```javascript
// Select best action based on visit count (exploitation)
let bestAction = validActions[0];
let bestVisits = 0;

for (const child of rootNode.children) {
  if (child.visits > bestVisits) {
    bestVisits = child.visits;
    bestAction = child.action;
  }
}
```

### Why Visit Count vs Average Value?
- **Robust Selection**: Visit count more reliable than average
- **Convergence**: Most-visited action usually has best long-term value
- **Noise Reduction**: Averages out statistical noise in rollouts
- **Confidence**: High visit count indicates high confidence

## 9. Performance Optimizations

### Computational Efficiency
- **Early Termination**: Stop if clearly winning move found
- **Simulation Limit**: Cap rollout length to prevent slowdowns
- **Yielding**: Periodically yield to prevent UI freezing
- **Action Space Pruning**: Limit evaluations for large action spaces

### Memory Management
- **State Cloning**: Efficient environment state duplication
- **Tree Pruning**: Could implement to limit memory usage
- **Garbage Collection**: Automatic cleanup of unused nodes

## 10. MCTS vs Neural Networks

### Advantages of MCTS
- **No Training Required**: Works immediately without training data
- **Interpretable**: Can examine tree to understand decision reasoning
- **Adaptive**: Automatically adjusts to game state complexity
- **Domain Knowledge**: Can incorporate heuristics easily

### Limitations
- **Computational Cost**: Requires many simulations per move
- **Time Constraints**: Performance limited by available thinking time
- **Scalability**: May struggle with very large action/state spaces
- **No Learning**: Doesn't improve from past experience

## 11. Wood Block Puzzle Specific Features

### Game-Specific Heuristics
- **Line Completion Priority**: Heavily favors moves that complete lines
- **Spatial Awareness**: Considers board utilization efficiency
- **Block Placement**: Evaluates different block orientations
- **Future Planning**: Looks ahead for setup opportunities

### State Representation
- **Environment Cloning**: Each node maintains full game state
- **Action Encoding**: Maps actions to block placement decisions
- **Valid Action Generation**: Only considers legal moves
- **Terminal Detection**: Recognizes game over conditions

## 12. Performance Metrics

### MCTS-Specific Statistics
- **Simulations Run**: Total Monte Carlo simulations performed
- **Tree Depth**: Maximum depth reached in search tree
- **Node Count**: Total nodes in search tree
- **Decision Time**: Time spent on each decision
- **Average Visits**: Average visit count per explored action

### Quality Indicators
- **Consistent Choices**: Stable action selection across runs
- **Tree Balance**: Even exploration of promising branches
- **Rollout Quality**: Meaningful value estimates from simulations
- **Convergence**: Settling on best action with more simulations

## Key MCTS Innovations for Block Puzzle

1. **Heuristic Rollouts**: Domain knowledge improves value estimation
2. **Early Termination**: Stops when finding clearly superior moves
3. **Adaptive Depth**: Limits search depth based on game complexity
4. **Block-Aware Simulation**: Understands block placement mechanics
5. **Line Completion Focus**: Prioritizes the main game objective

This MCTS implementation provides strategic decision-making without requiring training, making it an excellent baseline algorithm for comparing against neural network approaches while offering interpretable and immediately effective gameplay. 