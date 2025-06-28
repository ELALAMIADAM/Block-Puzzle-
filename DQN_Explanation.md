# DQN Algorithm Explanation: Wood Block Puzzle AI

## Overview
This document explains the Deep Q-Network (DQN) algorithm implementation for the Wood Block Puzzle game. The DQN uses a progressive curriculum-based approach with guided exploration and creative penalty systems to learn optimal block placement strategies.

## 1. Episode Initialization

### Environment Reset
- **Grid Setup**: Always uses a fixed 9×9 grid for neural network consistency
- **Block Generation**: Creates 3 new blocks based on current curriculum complexity level
- **Curriculum System**: Progressive block complexity from simple to full game blocks
- **State Tracking**: Initializes counters for moves, scores, and line clearing statistics

### State Representation (139 Features Total)
The AI observes the game through a comprehensive 139-dimensional feature vector:

**Grid State (81 features)**
- Each cell in the 9×9 grid is encoded as 0 (empty) or 1 (filled)
- Flattened in row-major order: index = row × 9 + col
- Provides complete spatial awareness of the current board state

**Line Completion Progress (27 features)**
- 9 row completion percentages (filled cells / 9)
- 9 column completion percentages (filled cells / 9) 
- 9 three-by-three square completion percentages (filled cells / 9)
- All values normalized to range [0.0, 1.0]

**Available Blocks Analysis (27 features)**
- 3 blocks × 9 features each = 27 total features
- Per block: size, width, height, density, linear potential, corner fit, edge fit, flexibility score, strategic value
- Padded with zeros if fewer than 3 blocks available

**Meta Information (4 features)**
- Score divided by 10000 (normalized performance indicator)
- Moves since last clear divided by 20 (urgency measure)
- Available blocks divided by 3 (resource availability)
- Curriculum level divided by 3 (difficulty indicator)

## 2. Action Selection Strategy

The DQN employs multiple action selection strategies that evolve during training:

### Adaptive Epsilon-Greedy
- **Base Strategy**: Balances exploration vs exploitation
- **Adaptive Epsilon**: ε = base_ε × (1 - level × 0.2)
- **Curriculum Awareness**: Lower exploration as curriculum level increases
- **Success Bonus**: Epsilon decay accelerates with line clearing success

### Guided Exploration (30% of exploration)
When exploring, the AI has a 30% chance to use guided exploration:
- **Simulation**: Tests each valid action by simulating block placement
- **Line Evaluation**: Counts potential line completions for each action
- **Scoring**: Awards 1000 points per line that would be completed
- **Selection**: Chooses action with highest line completion potential

### Random Exploration (70% of exploration)
- **Uniform Selection**: Randomly chooses from valid actions only
- **Safety**: Ensures only legal moves are considered
- **Diversity**: Maintains exploration diversity for robust learning

### DQN Exploitation
- **Neural Network**: Forward pass through 128→128→64→action_space architecture
- **Q-Value Prediction**: Estimates long-term value for each action
- **Best Action**: Selects action with highest predicted Q-value

## 3. Environment Interaction

### Action Execution
- **Action Decoding**: actionId = block × 1000 + row × 10 + col
- **Validation**: Checks bounds and collision constraints
- **Placement**: Updates grid state and removes used block

### Line Clearing System
- **Multi-Type Clearing**: Checks rows, columns, and 3×3 squares simultaneously
- **Scoring**: Score += lines² × 50 + lines × 100 (exponential bonus for multiple lines)
- **State Update**: Clears completed lines and updates game state

### Creative Spatial Analysis
The DQN includes innovative spatial intelligence:
- **Isolated Cell Detection**: Penalizes cells with no neighbors
- **Dead Space Recognition**: Identifies unfillable areas
- **Corner/Edge Efficiency**: Rewards optimal corner and edge utilization
- **Fragmentation Measurement**: Penalizes creation of disconnected regions

## 4. Reward System

### Constant Definitions (defined once)
```
LINE_BASE = 10000        // Base reward for any line clear
LINE_MULT = 5000         // Additional reward per line
COMBO_MULT = 15000       // Bonus for multiple simultaneous lines
PLACE = 10               // Points per block cell placed
GAME_OVER = -5000        // Penalty for ending game
SURVIVAL = +1            // Small bonus per turn survived
```

### Total Reward Calculation
```
Total Reward = LINE_BASE + (lines × LINE_MULT) + combo_bonus + spatial_penalties + survival_bonus
```

### Spatial Penalties/Bonuses
- **Isolation Penalty**: -100 per isolated cell
- **Dead Space Penalty**: -75 per unfillable space
- **Wasted Corner Penalty**: -50 per inefficient corner usage
- **Fragmentation Penalty**: -40 per disconnected region
- **Compactness Bonus**: +20 for well-connected structures

## 5. Experience Storage & Prioritization

### Prioritized Experience Replay
- **Memory Size**: 5000 experiences maximum
- **Line Clearing Priority**: Line clearing experiences get 10× normal priority
- **Replacement Strategy**: Replace oldest non-line-clearing experiences first
- **Experience Format**: (state, action, reward, next_state, done)

### Batch Sampling
- **Batch Size**: 32 experiences
- **Composition**: 70% from line clearing pool, 30% from other experiences
- **Training Focus**: Ensures learning emphasizes high-reward behaviors

## 6. Neural Network Training

### Network Architecture
- **Input Layer**: 139 features
- **Hidden Layers**: 128→128→64 neurons with ReLU activation and dropout (0.2)
- **Output Layer**: Action space size (varies by valid actions)
- **Optimizer**: Adam with learning rate 0.001

### Training Process
1. **Batch Preparation**: Stack 32 states and next_states into tensors
2. **Target Calculation**: 
   - If episode done: target = reward
   - If continuing: target = reward + γ × max(Q_target)
   - Discount factor γ = 0.99
3. **Loss Computation**: Mean Squared Error between predicted and target Q-values
4. **Backpropagation**: Update weights using Adam optimizer

### Target Network Updates
- **Frequency**: Every 50 training steps
- **Method**: Hard update (complete weight copy)
- **Purpose**: Prevents target drift and stabilizes training

## 7. Curriculum Learning System

### Block Complexity Progression
- **Level 0 (Simple)**: 1×1, 2×1, 2×2 blocks only
- **Level 1 (Medium)**: Adds 3×1 blocks and basic L-shapes
- **Level 2 (Complex)**: Adds T-shapes and complex L-shapes  
- **Level 3 (Full)**: All 22 block types from the complete game

### Advancement Criteria
- **Line Threshold**: Must clear specified number of lines
- **Episode Minimum**: At least 5 episodes at current level
- **Performance Consistency**: Demonstrates mastery before advancing

### Adaptation Effects
When curriculum advances:
- **Exploration Increase**: ε × 1.1 (more exploration for new challenges)
- **Learning Rate Reduction**: lr × 0.95 (slower, more careful learning)
- **Network Recompilation**: Updates optimizer with new learning rate

## 8. Performance Tracking & Metrics

### Dual Metric System
The system carefully distinguishes between:
- **AI Internal Rewards**: Used for training the neural network
- **Real Game Scores**: Actual game performance for fair evaluation

### Episode Statistics
Logged format: "Episode X: AI_Reward=Y, REAL_Score=Z, Lines=N, Level=L, Success=P%, Epsilon=E%"

### Key Performance Indicators
- **Best Real Score**: Highest actual game score achieved
- **Line Clearing Success Rate**: Percentage of moves resulting in line clears
- **Curriculum Level**: Current difficulty level (0-3)
- **Exploration Rate**: Current epsilon value as percentage

## 9. Memory Management

### Tensor Lifecycle
- **Automatic Disposal**: Disposes old TensorFlow tensors to prevent memory leaks
- **Priority-Based Replacement**: Maintains line clearing experiences longest
- **Efficient Storage**: Optimizes memory usage while preserving important experiences

### Training Stability
- **Validation**: Ensures tensor shapes match network expectations
- **Error Handling**: Graceful handling of training failures
- **Convergence Monitoring**: Tracks loss history for training progress

## Key Innovations

1. **Guided Exploration**: 30% of exploration time uses line completion simulation
2. **Creative Spatial Penalties**: Novel penalty system for spatial inefficiency
3. **Curriculum Learning**: Progressive block complexity advancement
4. **Dual Metrics**: Separates training signals from real performance evaluation
5. **Prioritized Memory**: 10× priority for line clearing experiences

This DQN implementation combines traditional reinforcement learning with game-specific innovations to create an AI that learns both immediate tactics and long-term strategic thinking for the Wood Block Puzzle game. 