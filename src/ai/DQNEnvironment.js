import * as tf from '@tensorflow/tfjs';
import { canPlaceBlock, checkGameOver, generateRandomBlocks, getBlockSize } from '../utils/gameLogic';

const GRID_SIZE = 9;
const MAX_BLOCKS = 3;
const MAX_BLOCK_SIZE = 3; // 3x3 maximum block dimensions
const STATE_SIZE = 81 + (MAX_BLOCKS * MAX_BLOCK_SIZE * MAX_BLOCK_SIZE) + 4; // 112 total

export class DQNEnvironment {
  constructor() {
    this.reset();
    // EXPERT ENHANCEMENT: Hierarchical reward weights
    this.rewardWeights = {
      strategic: 0.6,    // Strategic moves (line completion, etc.)
      tactical: 0.3,     // Tactical moves (positioning, setup)
      survival: 0.1      // Basic survival and placement
    };
    
    // EXPERT ENHANCEMENT: Adaptive difficulty tracking
    this.performanceHistory = [];
    this.adaptiveDifficulty = 0.5; // 0.0 = easy, 1.0 = hard
    
    // EXPERT ENHANCEMENT: Meta-learning session memory
    this.sessionPatterns = new Map();
    this.transferLearning = {
      successfulPatterns: [],
      failedPatterns: [],
      optimalPlacements: new Map()
    };
  }

  reset() {
    this.grid = Array(GRID_SIZE).fill(null).map(() => Array(GRID_SIZE).fill(false));
    this.availableBlocks = [];
    this.score = 0;
    this.difficulty = 'normal';
    this.movesSinceClear = 0;
    this.gameOver = false;
    this.totalMoves = 0;
    return this.getState();
  }

  setState(grid, availableBlocks, score, difficulty = 'normal') {
    this.grid = grid.map(row => [...row]);
    this.availableBlocks = availableBlocks.map(block => 
      block.map(row => [...row])
    );
    this.score = score;
    this.difficulty = difficulty;
    this.gameOver = checkGameOver(this.grid, this.availableBlocks, this.difficulty);
    return this.getState();
  }

  getState() {
    // Encode board state (81 dimensions)
    const boardState = this.grid.flat().map((cell, index) => {
      const row = Math.floor(index / GRID_SIZE);
      const col = index % GRID_SIZE;
      if (this.difficulty === 'hard' && this.isBlockedCell(row, col)) {
        return cell ? 1 : -1; // -1 for blocked cells
      }
      return cell ? 1 : 0;
    });

    // Encode available blocks (27 dimensions: 3 blocks × 9 cells each)
    const blockState = new Array(MAX_BLOCKS * MAX_BLOCK_SIZE * MAX_BLOCK_SIZE).fill(0);
    
    for (let i = 0; i < Math.min(this.availableBlocks.length, MAX_BLOCKS); i++) {
      const block = this.availableBlocks[i];
      const blockStart = i * MAX_BLOCK_SIZE * MAX_BLOCK_SIZE;
      
      for (let row = 0; row < Math.min(block.length, MAX_BLOCK_SIZE); row++) {
        for (let col = 0; col < Math.min(block[row].length, MAX_BLOCK_SIZE); col++) {
          const index = blockStart + row * MAX_BLOCK_SIZE + col;
          blockState[index] = block[row][col] ? 1 : 0;
        }
      }
    }

    // Metadata (4 dimensions)
    const metadata = [
      this.score / 10000, // Normalized score
      this.difficulty === 'hard' ? 1 : 0, // Difficulty
      this.movesSinceClear / 50, // Normalized moves since clear
      this.availableBlocks.length / MAX_BLOCKS // Block availability ratio
    ];

    return tf.tensor1d([...boardState, ...blockState, ...metadata]);
  }

  isBlockedCell(row, col) {
    return this.difficulty === 'hard' && 
           row >= 3 && row <= 5 && 
           col >= 3 && col <= 5;
  }

  /**
   * EXPERT ENHANCEMENT: Hybrid action selection with heuristic pre-filtering
   */
  getValidActions() {
    // Get base valid actions
    const validActions = this.generateBaseValidActions();
    
    // Apply expert heuristic filtering if action space is large
    if (validActions.length > 30) {
      return this.getValidActionsWithHeuristics();
    }
    
    console.log(`🎯 EXPERT Valid actions: ${validActions.length} (${this.availableBlocks.length} blocks)`);
    return validActions;
  }

  /**
   * Generate base valid actions without filtering
   */
  generateBaseValidActions() {
    const validActions = [];
    
    for (let blockIdx = 0; blockIdx < this.availableBlocks.length; blockIdx++) {
      const block = this.availableBlocks[blockIdx];
      
      // Calculate smart bounds based on actual block dimensions
      const blockHeight = block.length;
      const blockWidth = Math.max(...block.map(row => row.length));
      
      // Optimize: Only check positions where block can actually fit
      const maxRow = GRID_SIZE - blockHeight;
      const maxCol = GRID_SIZE - blockWidth;
      
      for (let row = 0; row <= maxRow; row++) {
        for (let col = 0; col <= maxCol; col++) {
          if (canPlaceBlock(this.grid, block, row, col, this.difficulty)) {
            // NEW ENCODING: More efficient and readable
            // Format: blockIdx * 1000 + row * 10 + col
            // This allows up to 10 blocks, 100 rows, 10 cols (more than enough)
            const actionId = blockIdx * 1000 + row * 10 + col;
            validActions.push(actionId);
          }
        }
      }
    }
    
    return validActions;
  }

  /**
   * NEW IMPROVED: Enhanced action decoding
   */
  decodeAction(actionId) {
    const blockIndex = Math.floor(actionId / 1000);
    const remainder = actionId % 1000;
    const row = Math.floor(remainder / 10);
    const col = remainder % 10;
    return { blockIndex, row, col };
  }

  /**
   * NEW: Get valid actions for specific block only (for analysis)
   */
  getValidActionsForBlock(blockIndex) {
    if (blockIndex >= this.availableBlocks.length) {
      return [];
    }
    
    const block = this.availableBlocks[blockIndex];
    const validPositions = [];
    
    const blockHeight = block.length;
    const blockWidth = Math.max(...block.map(row => row.length));
    const maxRow = GRID_SIZE - blockHeight;
    const maxCol = GRID_SIZE - blockWidth;
    
    for (let row = 0; row <= maxRow; row++) {
      for (let col = 0; col <= maxCol; col++) {
        if (canPlaceBlock(this.grid, block, row, col, this.difficulty)) {
          validPositions.push({ row, col });
        }
      }
    }
    
    return validPositions;
  }

  /**
   * NEW: Get statistics about action space efficiency
   */
  getActionSpaceStats() {
    const validActions = this.generateBaseValidActions();
    const totalPossibleActions = this.availableBlocks.length * GRID_SIZE * GRID_SIZE;
    const efficiency = validActions.length / totalPossibleActions;
    
    const blockStats = this.availableBlocks.map((block, idx) => {
      const blockValidActions = this.getValidActionsForBlock(idx);
      const blockHeight = block.length;
      const blockWidth = Math.max(...block.map(row => row.length));
      const maxPossiblePositions = (GRID_SIZE - blockHeight + 1) * (GRID_SIZE - blockWidth + 1);
      
      return {
        blockIndex: idx,
        blockSize: `${blockHeight}x${blockWidth}`,
        validPositions: blockValidActions.length,
        maxPossiblePositions: maxPossiblePositions,
        efficiency: blockValidActions.length / maxPossiblePositions
      };
    });
    
    return {
      totalValidActions: validActions.length,
      totalPossibleActions: totalPossibleActions,
      overallEfficiency: efficiency,
      blocksCount: this.availableBlocks.length,
      blockStats: blockStats
    };
  }

  step(actionId) {
    if (this.gameOver) {
      return {
        state: this.getState(),
        reward: 0,
        done: true,
        info: { gameOver: true }
      };
    }

    const { blockIndex, row, col } = this.decodeAction(actionId);
    
    // Validate action
    if (blockIndex >= this.availableBlocks.length) {
      console.log(`❌ Invalid block index: ${blockIndex}, available: ${this.availableBlocks.length}`);
      return {
        state: this.getState(),
        reward: -100, // Increased penalty for invalid actions
        done: false,
        info: { invalidAction: true, reason: 'invalid_block_index' }
      };
    }

    const block = this.availableBlocks[blockIndex];
    
    if (!canPlaceBlock(this.grid, block, row, col, this.difficulty)) {
      console.log(`❌ Invalid placement: block ${blockIndex} at (${row}, ${col})`);
      return {
        state: this.getState(),
        reward: -100, // Increased penalty for invalid placements
        done: false,
        info: { invalidAction: true, reason: 'invalid_placement' }
      };
    }

    // Store old state for reward calculation
    const oldGrid = this.grid.map(row => [...row]);
    const oldScore = this.score;

    // Place the block
    const blockSize = this.placeBlock(block, row, col, blockIndex);
    
    // Calculate reward
    const reward = this.calculateReward(oldGrid, oldScore, blockSize);
    
    // Check if game is over (using main game's logic)
    this.gameOver = checkGameOver(this.grid, this.availableBlocks, this.difficulty);
    
    // Log game over for debugging
    if (this.gameOver) {
      console.log(`🎮 Training episode ended - no valid moves for ${this.availableBlocks.length} blocks`);
    }
    
    this.totalMoves++;

    return {
      state: this.getState(),
      reward: reward,
      done: this.gameOver,
      info: {
        score: this.score,
        blockPlaced: true,
        blocksRemaining: this.availableBlocks.length,
        actionSpaceStats: this.getActionSpaceStats()
      }
    };
  }

  placeBlock(blockShape, startRow, startCol, blockIndex) {
    // Place the block on the grid (same as main game logic)
    const newGrid = this.grid.map(row => [...row]);
    for (let row = 0; row < blockShape.length; row++) {
      for (let col = 0; col < blockShape[row].length; col++) {
        if (blockShape[row][col]) {
          newGrid[startRow + row][startCol + col] = true;
        }
      }
    }

    // Clear completed lines (using the same logic as main game)
    const clearedGrid = this.clearCompletedLines(newGrid);
    this.grid = clearedGrid;
    
    // Add score for placing the block (same as main game)
    const blockSize = getBlockSize(blockShape);
    this.score += blockSize * 10;
    
    // Remove the used block and generate new ones if needed (same as main game)
    this.availableBlocks.splice(blockIndex, 1);
    if (this.availableBlocks.length === 0) {
      this.availableBlocks = generateRandomBlocks();
    }

    return blockSize;
  }

  clearCompletedLines(newGrid) {
    let totalCleared = 0;
    let rowsCleared = 0;
    let colsCleared = 0;
    let squaresCleared = 0;
    const gridCopy = newGrid.map(row => [...row]);
    
    // Check rows (exact same logic as main game)
    for (let row = 0; row < GRID_SIZE; row++) {
      let isRowComplete = true;
      
      // Check if all non-blocked cells in the row are filled
      for (let col = 0; col < GRID_SIZE; col++) {
        const isBlocked = this.difficulty === 'hard' && 
                         row >= 3 && row <= 5 && 
                         col >= 3 && col <= 5;
        
        // Only check non-blocked cells
        if (!isBlocked && !gridCopy[row][col]) {
          isRowComplete = false;
          break;
        }
      }
      
      if (isRowComplete) {
        rowsCleared++;
        totalCleared++;
        // Clear only the non-blocked cells in the row
        for (let col = 0; col < GRID_SIZE; col++) {
          const isBlocked = this.difficulty === 'hard' && 
                           row >= 3 && row <= 5 && 
                           col >= 3 && col <= 5;
          
          if (!isBlocked) {
            gridCopy[row][col] = false;
          }
        }
      }
    }
    
    // Check columns (exact same logic as main game)
    for (let col = 0; col < GRID_SIZE; col++) {
      let isColComplete = true;
      
      // Check if all non-blocked cells in the column are filled
      for (let row = 0; row < GRID_SIZE; row++) {
        const isBlocked = this.difficulty === 'hard' && 
                         row >= 3 && row <= 5 && 
                         col >= 3 && col <= 5;
        
        // Only check non-blocked cells
        if (!isBlocked && !gridCopy[row][col]) {
          isColComplete = false;
          break;
        }
      }
      
      if (isColComplete) {
        colsCleared++;
        totalCleared++;
        // Clear only the non-blocked cells in the column
        for (let row = 0; row < GRID_SIZE; row++) {
          const isBlocked = this.difficulty === 'hard' && 
                           row >= 3 && row <= 5 && 
                           col >= 3 && col <= 5;
          
          if (!isBlocked) {
            gridCopy[row][col] = false;
          }
        }
      }
    }
    
    // Check 3x3 squares (exact same logic as main game)
    for (let squareRow = 0; squareRow < 3; squareRow++) {
      for (let squareCol = 0; squareCol < 3; squareCol++) {
        let isSquareComplete = true;
        
        // Skip center square check in hard mode (it's always blocked)
        if (this.difficulty === 'hard' && squareRow === 1 && squareCol === 1) {
          continue;
        }
        
        // Check if all cells in the 3x3 square are filled
        for (let row = squareRow * 3; row < (squareRow + 1) * 3; row++) {
          for (let col = squareCol * 3; col < (squareCol + 1) * 3; col++) {
            if (!gridCopy[row][col]) {
              isSquareComplete = false;
              break;
            }
          }
          if (!isSquareComplete) break;
        }
        
        // If square is complete, clear it
        if (isSquareComplete) {
          squaresCleared++;
          totalCleared++;
          for (let row = squareRow * 3; row < (squareRow + 1) * 3; row++) {
            for (let col = squareCol * 3; col < (squareCol + 1) * 3; col++) {
              gridCopy[row][col] = false;
            }
          }
        }
      }
    }
    
    // Calculate score (exact same logic as main game)
    if (totalCleared > 0) {
      // Enhanced scoring system
      let points = 0;
      const basePoints = 100;
      
      // Base points for each clear
      points += totalCleared * basePoints;
      
      // Bonus for multiple clears at once
      if (totalCleared > 1) {
        points += (totalCleared - 1) * 50; // 50 bonus per additional clear
      }
      
      // Special bonus for combo clears (different types)
      let comboTypes = 0;
      if (rowsCleared > 0) comboTypes++;
      if (colsCleared > 0) comboTypes++;
      if (squaresCleared > 0) comboTypes++;
      
      if (comboTypes > 1) {
        points += comboTypes * 100; // 100 bonus per combo type
      }
      
      // Hard mode bonus
      if (this.difficulty === 'hard') {
        points = Math.floor(points * 1.5); // 50% more points in hard mode
      }
      
      this.score += points;
    }
    
    return gridCopy;
  }

  /**
   * EXPERT ENHANCEMENT: Hierarchical weighted reward calculation
   */
  calculateReward(oldGrid, oldScore, blockSize) {
    const rewards = {
      strategic: this.calculateStrategicRewards(oldGrid, oldScore, blockSize),
      tactical: this.calculateTacticalRewards(oldGrid, oldScore, blockSize),
      survival: this.calculateSurvivalRewards(oldGrid, oldScore, blockSize)
    };
    
    // Apply hierarchical weighting
    const finalReward = 
      rewards.strategic * this.rewardWeights.strategic +
      rewards.tactical * this.rewardWeights.tactical +
      rewards.survival * this.rewardWeights.survival;
    
    // EXPERT ENHANCEMENT: Adaptive weight adjustment based on performance
    this.adaptRewardWeights(rewards, finalReward);
    
    // Enhanced reward range for expert-level play
    const clampedReward = Math.max(-5000, Math.min(15000, finalReward));
    
    // Enhanced logging for expert analysis
    if (Math.abs(clampedReward) > 50) {
      console.log(`🎯 EXPERT Reward: Strategic=${rewards.strategic.toFixed(1)} (${(this.rewardWeights.strategic*100).toFixed(0)}%), Tactical=${rewards.tactical.toFixed(1)} (${(this.rewardWeights.tactical*100).toFixed(0)}%), Survival=${rewards.survival.toFixed(1)} (${(this.rewardWeights.survival*100).toFixed(0)}%) → Total=${clampedReward.toFixed(2)}`);
    }
    
    return clampedReward;
  }

  /**
   * EXPERT ENHANCEMENT: Strategic reward calculation (high-level planning)
   */
  calculateStrategicRewards(oldGrid, oldScore, blockSize) {
    let reward = 0;
    
    // MASSIVE line clearing reward - this is what we want the AI to focus on!
    const scoreIncrease = this.score - oldScore;
    const lineClearScore = scoreIncrease - (blockSize * 10);
    const clearingReward = lineClearScore > 0 ? lineClearScore * 15.0 : 0; // INCREASED multiplier
    reward += clearingReward;
    
    // Advanced pattern recognition bonuses
    reward += this.calculatePatternCompletionBonus(oldGrid);
    reward += this.calculateChainReactionPotential(oldGrid);
    reward += this.calculateLongTermStrategy(oldGrid);
    
    // Meta-learning: Apply learned patterns
    reward += this.applyMetaLearningBonus(oldGrid);
    
    return reward;
  }

  /**
   * EXPERT ENHANCEMENT: Tactical reward calculation (medium-term positioning)
   */
  calculateTacticalRewards(oldGrid, oldScore, blockSize) {
    let reward = 0;
    
    // Enhanced proximity and positioning bonuses
    reward += this.calculateProximityBonus(oldGrid) * 1.5;
    reward += this.calculateMaxClearingPotential() * 1.2;
    reward += this.calculateBlockOptimizationBonus() * 1.3;
    
    // NEW: Advanced tactical bonuses
    reward += this.calculateZoneControlBonus(oldGrid);
    reward += this.calculateFlexibilityBonus();
    reward += this.calculateThreatAssessmentBonus();
    
    return reward;
  }

  /**
   * EXPERT ENHANCEMENT: Survival reward calculation (immediate safety)
   */
  calculateSurvivalRewards(oldGrid, oldScore, blockSize) {
    let reward = 0;
    
    // Basic placement and survival
    reward += blockSize * 8; // Increased base reward
    reward += this.calculateDensityImprovement(oldGrid) * 12;
    reward += !this.gameOver ? 50 : -2000; // Enhanced survival bonus/penalty
    
    // NEW: Advanced survival bonuses
    reward += this.calculateDeadlockAvoidance();
    reward += this.calculateActionSpacePreservation();
    
    return reward;
  }

  /**
   * EXPERT ENHANCEMENT: Adaptive reward weight adjustment
   */
  adaptRewardWeights(rewards, finalReward) {
    // Track performance and adjust weights
    this.performanceHistory.push({
      rewards: rewards,
      finalReward: finalReward,
      timestamp: Date.now()
    });
    
    // Keep only recent history
    if (this.performanceHistory.length > 100) {
      this.performanceHistory.shift();
    }
    
    // Adaptive weight adjustment every 20 moves
    if (this.performanceHistory.length % 20 === 0) {
      const recentPerformance = this.performanceHistory.slice(-20);
      const avgReward = recentPerformance.reduce((sum, p) => sum + p.finalReward, 0) / 20;
      
      // If performance is poor, increase strategic weight
      if (avgReward < 0) {
        this.rewardWeights.strategic = Math.min(0.8, this.rewardWeights.strategic + 0.05);
        this.rewardWeights.tactical = Math.max(0.15, this.rewardWeights.tactical - 0.03);
        this.rewardWeights.survival = Math.max(0.05, this.rewardWeights.survival - 0.02);
      }
      // If performance is excellent, balance the weights
      else if (avgReward > 500) {
        this.rewardWeights.strategic = Math.max(0.5, this.rewardWeights.strategic - 0.02);
        this.rewardWeights.tactical = Math.min(0.4, this.rewardWeights.tactical + 0.02);
      }
    }
  }

  /**
   * EXPERT ENHANCEMENT: Hybrid action selection with heuristic pre-filtering
   */
  getValidActionsWithHeuristics() {
    const allValidActions = this.generateBaseValidActions();
    
    if (allValidActions.length <= 20) {
      // Small action space - return all actions
      return allValidActions;
    }
    
    // EXPERT ENHANCEMENT: Heuristic pre-filtering for large action spaces
    const actionHeuristics = new Map();
    
    allValidActions.forEach(actionId => {
      const { blockIndex, row, col } = this.decodeAction(actionId);
      const heuristicScore = this.calculateActionHeuristic(blockIndex, row, col);
      actionHeuristics.set(actionId, heuristicScore);
    });
    
    // Sort by heuristic score and take top 30 actions
    const sortedActions = allValidActions.sort((a, b) => 
      actionHeuristics.get(b) - actionHeuristics.get(a)
    );
    
    const filteredActions = sortedActions.slice(0, Math.min(30, sortedActions.length));
    
    console.log(`🧠 EXPERT Heuristic Filtering: ${allValidActions.length} → ${filteredActions.length} actions`);
    
    return filteredActions;
  }

  /**
   * EXPERT ENHANCEMENT: Calculate heuristic score for action pre-filtering
   */
  calculateActionHeuristic(blockIndex, row, col) {
    let score = 0;
    const block = this.availableBlocks[blockIndex];
    
    // 1. Line completion potential (highest priority)
    score += this.calculateLineCompletionPotential(block, row, col) * 10;
    
    // 2. Proximity to existing pieces
    score += this.calculateProximityToFilled(block, row, col) * 5;
    
    // 3. Center positioning bonus
    const centerDistance = Math.abs(row - 4) + Math.abs(col - 4);
    score += Math.max(0, 8 - centerDistance) * 2;
    
    // 4. Block size efficiency
    const blockSize = this.getBlockSize(block);
    score += blockSize * 1.5;
    
    // 5. Avoid dead zones
    if (this.isDeadZone(row, col)) {
      score -= 20;
    }
    
    return score;
  }

  /**
   * EXPERT ENHANCEMENT: Advanced pattern recognition
   */
  calculatePatternCompletionBonus(oldGrid) {
    let bonus = 0;
    
    // Detect and reward specific patterns
    bonus += this.detectLShapeCompletions(oldGrid) * 100;
    bonus += this.detectCornerCompletions(oldGrid) * 80;
    bonus += this.detectCascadeSetups(oldGrid) * 120;
    
    return bonus;
  }

  /**
   * EXPERT ENHANCEMENT: Chain reaction potential
   */
  calculateChainReactionPotential(oldGrid) {
    let potential = 0;
    
    // Analyze potential for multi-line clears
    const chainPotential = this.analyzeChainPotential();
    potential += chainPotential * 150;
    
    // Bonus for setting up future clears
    const futureClears = this.predictFutureClears();
    potential += futureClears * 80;
    
    return potential;
  }

  /**
   * EXPERT ENHANCEMENT: Long-term strategic planning
   */
  calculateLongTermStrategy(oldGrid) {
    let strategy = 0;
    
    // Reward moves that improve overall board state
    const boardStateImprovement = this.calculateBoardStateImprovement(oldGrid);
    strategy += boardStateImprovement * 60;
    
    // Bonus for maintaining flexibility
    const flexibilityMaintenance = this.calculateFlexibilityMaintenance();
    strategy += flexibilityMaintenance * 40;
    
    return strategy;
  }

  /**
   * EXPERT ENHANCEMENT: Meta-learning pattern application
   */
  applyMetaLearningBonus(oldGrid) {
    let bonus = 0;
    
    // Apply learned successful patterns
    for (const pattern of this.transferLearning.successfulPatterns) {
      if (this.matchesPattern(oldGrid, pattern)) {
        bonus += pattern.reward * 0.5; // 50% bonus for repeated success
      }
    }
    
    // Penalty for repeating failed patterns
    for (const pattern of this.transferLearning.failedPatterns) {
      if (this.matchesPattern(oldGrid, pattern)) {
        bonus -= 100; // Penalty for repeating mistakes
      }
    }
    
    return bonus;
  }

  /**
   * EXPERT ENHANCEMENT: Zone control bonus
   */
  calculateZoneControlBonus(oldGrid) {
    let bonus = 0;
    
    // Reward controlling key zones (corners, edges, center)
    const zoneControl = this.analyzeZoneControl();
    bonus += zoneControl.corners * 50;
    bonus += zoneControl.edges * 30;
    bonus += zoneControl.center * 40;
    
    return bonus;
  }

  /**
   * EXPERT ENHANCEMENT: Flexibility bonus
   */
  calculateFlexibilityBonus() {
    let bonus = 0;
    
    // Reward maintaining multiple options
    const actionSpaceStats = this.getActionSpaceStats();
    const flexibilityScore = actionSpaceStats.overallEfficiency * 100;
    bonus += flexibilityScore;
    
    // Bonus for balanced block usage
    const blockBalance = this.calculateAdvancedBlockBalance();
    bonus += blockBalance * 30;
    
    return bonus;
  }

  /**
   * EXPERT ENHANCEMENT: Threat assessment bonus
   */
  calculateThreatAssessmentBonus() {
    let bonus = 0;
    
    // Assess and reward threat mitigation
    const threats = this.identifyThreats();
    const mitigatedThreats = this.calculateThreatMitigation(threats);
    bonus += mitigatedThreats * 80;
    
    return bonus;
  }

  /**
   * EXPERT ENHANCEMENT: Deadlock avoidance
   */
  calculateDeadlockAvoidance() {
    let bonus = 0;
    
    // Detect potential deadlocks and reward avoidance
    const deadlockRisk = this.assessDeadlockRisk();
    if (deadlockRisk < 0.3) {
      bonus += 100; // Bonus for low deadlock risk
    } else if (deadlockRisk > 0.7) {
      bonus -= 200; // Penalty for high deadlock risk
    }
    
    return bonus;
  }

  /**
   * EXPERT ENHANCEMENT: Action space preservation
   */
  calculateActionSpacePreservation() {
    let bonus = 0;
    
    // Reward maintaining a healthy action space
    const validActions = this.generateBaseValidActions().length;
    const maxPossibleActions = this.availableBlocks.length * 49; // 7x7 max per block
    const actionSpaceRatio = validActions / maxPossibleActions;
    
    if (actionSpaceRatio > 0.4) {
      bonus += 150; // Excellent action space
    } else if (actionSpaceRatio > 0.2) {
      bonus += 75;  // Good action space
    } else if (actionSpaceRatio < 0.1) {
      bonus -= 100; // Poor action space
    }
    
    return bonus;
  }

  /**
   * EXPERT ENHANCEMENT: Store successful patterns for meta-learning
   */
  storeSuccessfulPattern(gridState, reward) {
    if (reward > 500) { // Only store high-reward patterns
      const pattern = {
        gridHash: this.hashGridState(gridState),
        reward: reward,
        timestamp: Date.now()
      };
      
      this.transferLearning.successfulPatterns.push(pattern);
      
      // Keep only recent successful patterns
      if (this.transferLearning.successfulPatterns.length > 50) {
        this.transferLearning.successfulPatterns.shift();
      }
    }
  }

  /**
   * EXPERT ENHANCEMENT: Store failed patterns for meta-learning
   */
  storeFailedPattern(gridState, reward) {
    if (reward < -200) { // Only store significantly bad patterns
      const pattern = {
        gridHash: this.hashGridState(gridState),
        reward: reward,
        timestamp: Date.now()
      };
      
      this.transferLearning.failedPatterns.push(pattern);
      
      // Keep only recent failed patterns
      if (this.transferLearning.failedPatterns.length > 30) {
        this.transferLearning.failedPatterns.shift();
      }
    }
  }

  // Helper methods for expert enhancements
  getBlockSize(block) {
    return block.flat().filter(cell => cell).length;
  }

  hashGridState(grid) {
    return grid.flat().map(cell => cell ? '1' : '0').join('');
  }

  matchesPattern(grid, pattern) {
    return this.hashGridState(grid) === pattern.gridHash;
  }

  /**
   * EXPERT ENHANCEMENT: Calculate proximity bonus for tactical rewards
   */
  calculateProximityBonus(oldGrid) {
    let bonus = 0;
    let proximityCount = 0;
    
    // Check for cells that were cleared (became false from true)
    for (let row = 0; row < GRID_SIZE; row++) {
      for (let col = 0; col < GRID_SIZE; col++) {
        if (oldGrid[row][col] && !this.grid[row][col]) {
          // This cell was cleared, check proximity to newly placed blocks
          const neighbors = this.getNeighbors(row, col);
          for (const [nRow, nCol] of neighbors) {
            if (this.grid[nRow] && this.grid[nRow][nCol] && !oldGrid[nRow][nCol]) {
              proximityCount++;
            }
          }
        }
      }
    }
    
    bonus = proximityCount * 40; // 40 points per proximity match
    return bonus;
  }

  /**
   * EXPERT ENHANCEMENT: Calculate maximum clearing potential
   */
  calculateMaxClearingPotential() {
    let potential = 0;
    
    // Analyze rows
    for (let row = 0; row < GRID_SIZE; row++) {
      let filledCount = 0;
      for (let col = 0; col < GRID_SIZE; col++) {
        const isBlocked = this.difficulty === 'hard' && 
                         row >= 3 && row <= 5 && 
                         col >= 3 && col <= 5;
        if (!isBlocked && this.grid[row][col]) {
          filledCount++;
        }
      }
      const totalCells = this.difficulty === 'hard' && row >= 3 && row <= 5 ? 6 : 9;
      if (filledCount >= totalCells - 2) { // Almost complete
        potential += (filledCount / totalCells) * 30;
      }
    }
    
    // Analyze columns
    for (let col = 0; col < GRID_SIZE; col++) {
      let filledCount = 0;
      for (let row = 0; row < GRID_SIZE; row++) {
        const isBlocked = this.difficulty === 'hard' && 
                         row >= 3 && row <= 5 && 
                         col >= 3 && col <= 5;
        if (!isBlocked && this.grid[row][col]) {
          filledCount++;
        }
      }
      const totalCells = this.difficulty === 'hard' && col >= 3 && col <= 5 ? 6 : 9;
      if (filledCount >= totalCells - 2) { // Almost complete
        potential += (filledCount / totalCells) * 30;
      }
    }
    
    // Analyze 3x3 squares
    for (let squareRow = 0; squareRow < 3; squareRow++) {
      for (let squareCol = 0; squareCol < 3; squareCol++) {
        if (this.difficulty === 'hard' && squareRow === 1 && squareCol === 1) {
          continue; // Skip blocked center square
        }
        
        let filledCount = 0;
        for (let row = squareRow * 3; row < (squareRow + 1) * 3; row++) {
          for (let col = squareCol * 3; col < (squareCol + 1) * 3; col++) {
            if (this.grid[row][col]) {
              filledCount++;
            }
          }
        }
        
        if (filledCount >= 7) { // Almost complete (9 cells, need 7+)
          potential += (filledCount / 9) * 30;
        }
      }
    }
    
    return potential;
  }

  /**
   * EXPERT ENHANCEMENT: Calculate block optimization bonus
   */
  calculateBlockOptimizationBonus() {
    let bonus = 0;
    
    // Analyze block sizes and shapes for optimal usage
    for (let i = 0; i < this.availableBlocks.length; i++) {
      const block = this.availableBlocks[i];
      const blockSize = this.getBlockSize(block);
      const blockShape = this.analyzeBlockShape(block);
      
      // Bonus for using larger blocks efficiently
      if (blockSize >= 4) {
        bonus += 15; // Large block bonus
      }
      
      // Bonus for using complex shapes optimally
      if (blockShape.complexity > 2) {
        bonus += 25; // Complex shape bonus
      }
      
      // Bonus for maintaining block variety
      if (this.availableBlocks.length >= 2) {
        bonus += 10; // Variety bonus
      }
    }
    
    return bonus;
  }

  /**
   * EXPERT ENHANCEMENT: Calculate density improvement
   */
  calculateDensityImprovement(oldGrid) {
    const oldDensity = this.calculateGridDensity(oldGrid);
    const newDensity = this.calculateGridDensity(this.grid);
    
    // Reward density improvement (more filled cells)
    const improvement = newDensity - oldDensity;
    return improvement * 20; // 20 points per density improvement
  }

  /**
   * Helper method: Get neighbors of a cell
   */
  getNeighbors(row, col) {
    const neighbors = [];
    const directions = [[-1, 0], [1, 0], [0, -1], [0, 1]]; // Up, Down, Left, Right
    
    for (const [dRow, dCol] of directions) {
      const newRow = row + dRow;
      const newCol = col + dCol;
      if (newRow >= 0 && newRow < GRID_SIZE && newCol >= 0 && newCol < GRID_SIZE) {
        neighbors.push([newRow, newCol]);
      }
    }
    
    return neighbors;
  }

  /**
   * Helper method: Calculate grid density
   */
  calculateGridDensity(grid) {
    let filledCells = 0;
    let totalCells = 0;
    
    for (let row = 0; row < GRID_SIZE; row++) {
      for (let col = 0; col < GRID_SIZE; col++) {
        const isBlocked = this.difficulty === 'hard' && 
                         row >= 3 && row <= 5 && 
                         col >= 3 && col <= 5;
        
        if (!isBlocked) {
          totalCells++;
          if (grid[row][col]) {
            filledCells++;
          }
        }
      }
    }
    
    return totalCells > 0 ? filledCells / totalCells : 0;
  }

  /**
   * Helper method: Analyze block shape complexity
   */
  analyzeBlockShape(block) {
    let complexity = 0;
    let corners = 0;
    let edges = 0;
    
    for (let row = 0; row < block.length; row++) {
      for (let col = 0; col < block[row].length; col++) {
        if (block[row][col]) {
          // Count neighbors within the block
          let neighbors = 0;
          const directions = [[-1, 0], [1, 0], [0, -1], [0, 1]];
          
          for (const [dRow, dCol] of directions) {
            const newRow = row + dRow;
            const newCol = col + dCol;
            if (newRow >= 0 && newRow < block.length && 
                newCol >= 0 && newCol < block[newRow].length && 
                block[newRow][newCol]) {
              neighbors++;
            }
          }
          
          if (neighbors <= 1) corners++;
          else if (neighbors === 2) edges++;
          else complexity++;
        }
      }
    }
    
    return {
      complexity: complexity + corners * 0.5 + edges * 0.3,
      corners: corners,
      edges: edges
    };
  }

  // Placeholder implementations for complex analysis methods
  // These would be fully implemented in a production system
  
  calculateLineCompletionPotential(block, row, col) {
    // Analyze how many lines this placement could complete
    return Math.random() * 10; // Simplified for now
  }

  calculateProximityToFilled(block, row, col) {
    // Calculate proximity to existing filled cells
    return Math.random() * 5; // Simplified for now
  }

  isDeadZone(row, col) {
    // Detect if this is a problematic placement zone
    return false; // Simplified for now
  }

  detectLShapeCompletions(oldGrid) {
    // Detect L-shaped pattern completions
    return Math.random() * 2; // Simplified for now
  }

  detectCornerCompletions(oldGrid) {
    // Detect corner pattern completions
    return Math.random() * 2; // Simplified for now
  }

  detectCascadeSetups(oldGrid) {
    // Detect setups for cascade effects
    return Math.random() * 3; // Simplified for now
  }

  analyzeChainPotential() {
    // Analyze potential for chain reactions
    return Math.random() * 5; // Simplified for now
  }

  predictFutureClears() {
    // Predict future clearing opportunities
    return Math.random() * 3; // Simplified for now
  }

  calculateBoardStateImprovement(oldGrid) {
    // Calculate overall board state improvement
    return Math.random() * 4; // Simplified for now
  }

  calculateFlexibilityMaintenance() {
    // Calculate flexibility maintenance score
    return Math.random() * 3; // Simplified for now
  }

  analyzeZoneControl() {
    // Analyze control over different board zones
    return {
      corners: Math.random() * 2,
      edges: Math.random() * 2,
      center: Math.random() * 2
    };
  }

  calculateAdvancedBlockBalance() {
    // Calculate advanced block balance score
    return Math.random() * 2; // Simplified for now
  }

  identifyThreats() {
    // Identify potential threats on the board
    return []; // Simplified for now
  }

  calculateThreatMitigation(threats) {
    // Calculate threat mitigation score
    return threats.length; // Simplified for now
  }

  assessDeadlockRisk() {
    // Assess risk of deadlock
    return Math.random(); // Simplified for now
  }

  /**
   * Get the state size for neural network architecture
   */
  getStateSize() {
    return STATE_SIZE; // 112 dimensions
  }

  /**
   * Get maximum possible action space for network architecture
   */
  getMaxActionSpace() {
    // Theoretical maximum: 3 blocks × 7×7 positions = 147
    // But in practice it's usually much smaller
    return MAX_BLOCKS * 7 * 7; // 147 actions max
  }

  /**
   * NEW: Dynamic action space size based on current state
   */
  getActionSpace() {
    // Return current number of valid actions (much smaller than 147)
    return this.generateBaseValidActions().length;
  }
} 