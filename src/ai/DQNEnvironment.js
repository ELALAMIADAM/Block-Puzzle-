import * as tf from '@tensorflow/tfjs';
import { canPlaceBlock, checkGameOver, generateRandomBlocks, getBlockSize } from '../utils/gameLogic';

const GRID_SIZE = 9;
const MAX_BLOCKS = 3;
const MAX_BLOCK_SIZE = 3; // 3x3 maximum block dimensions
const STATE_SIZE = 81 + (MAX_BLOCKS * MAX_BLOCK_SIZE * MAX_BLOCK_SIZE) + 4; // 112 total

export class DQNEnvironment {
  constructor() {
    this.reset();
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
   * NEW IMPROVED: Generate valid actions more efficiently
   * Uses optimized bounds checking and better encoding
   */
  getValidActions() {
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
    
    console.log(`🎯 Valid actions generated: ${validActions.length} (${this.availableBlocks.length} blocks)`);
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
    const validActions = this.getValidActions();
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

  calculateReward(oldGrid, oldScore, blockSize) {
    let reward = 0;
    
    // Base reward for valid placement (enhanced based on block size)
    const placementReward = blockSize * 15; // Increased from 10 to 15
    reward += placementReward;
    
    // Line clearing reward (much higher)
    const scoreIncrease = this.score - oldScore;
    const lineClearScore = scoreIncrease - (blockSize * 10);
    const clearingReward = lineClearScore > 0 ? lineClearScore * 2.5 : 0; // Increased multiplier
    reward += clearingReward;
    
    // Efficiency bonus - reward keeping the board less cluttered
    const densityImprovement = this.calculateDensityImprovement(oldGrid);
    const densityReward = densityImprovement * 25; // Increased from 20 to 25
    reward += densityReward;
    
    // Survival bonus (higher reward for not ending the game)
    const survivalReward = !this.gameOver ? 25 : -750; // Increased survival bonus, higher penalty
    reward += survivalReward;
    
    // Strategic placement bonus (enhanced)
    const strategicReward = this.calculateStrategicBonus() * 8; // Increased from 5 to 8
    reward += strategicReward;
    
    // Action space bonus - reward moves that keep more options open
    const actionSpaceBonus = this.calculateActionSpaceBonus();
    reward += actionSpaceBonus;
    
    // Hard mode bonus
    if (this.difficulty === 'hard') {
      reward *= 1.5;
    }
    
    const finalReward = Math.max(-1500, Math.min(3000, reward)); // Expanded range
    
    // Log reward breakdown for debugging
    if (Math.abs(finalReward) > 10) { // Only log significant rewards
      console.log(`🎯 Reward: Place=${placementReward}, Clear=${clearingReward.toFixed(1)}, Density=${densityReward.toFixed(1)}, Survive=${survivalReward}, Strategic=${strategicReward.toFixed(1)}, ActionSpace=${actionSpaceBonus.toFixed(1)} → Total=${finalReward.toFixed(2)}`);
    }
    
    return finalReward;
  }

  /**
   * NEW: Calculate bonus for maintaining action space diversity
   */
  calculateActionSpaceBonus() {
    const stats = this.getActionSpaceStats();
    let bonus = 0;
    
    // Reward having multiple viable blocks
    bonus += stats.blocksCount * 5;
    
    // Reward overall action space efficiency
    bonus += stats.overallEfficiency * 10;
    
    // Bonus for blocks with good placement options
    stats.blockStats.forEach(blockStat => {
      if (blockStat.efficiency > 0.3) { // If block has good placement options
        bonus += 5;
      }
    });
    
    return bonus;
  }

  calculateDensityImprovement(oldGrid) {
    const oldDensity = this.calculateGridDensity(oldGrid);
    const newDensity = this.calculateGridDensity(this.grid);
    return (newDensity - oldDensity) * 10;
  }

  calculateGridDensity(grid) {
    let filled = 0;
    let total = 0;
    
    for (let row = 0; row < GRID_SIZE; row++) {
      for (let col = 0; col < GRID_SIZE; col++) {
        if (!this.isBlockedCell(row, col)) {
          total++;
          if (grid[row][col]) filled++;
        }
      }
    }
    
    return total > 0 ? filled / total : 0;
  }

  calculateStrategicBonus() {
    let bonus = 0;
    
    // Bonus for potential line completions
    const potentialLines = this.countPotentialLineCompletions();
    bonus += potentialLines * 4; // Increased from 3 to 4
    
    // Bonus for keeping options open
    const validMoves = this.getValidActions().length;
    bonus += Math.min(validMoves / 8, 10); // Adjusted scaling
    
    // Bonus for balanced block distribution
    const blockBalance = this.calculateBlockBalance();
    bonus += blockBalance;
    
    return bonus;
  }

  /**
   * NEW: Calculate bonus for having balanced block sizes
   */
  calculateBlockBalance() {
    if (this.availableBlocks.length === 0) return 0;
    
    const blockSizes = this.availableBlocks.map(block => getBlockSize(block));
    const avgSize = blockSizes.reduce((a, b) => a + b, 0) / blockSizes.length;
    const variance = blockSizes.reduce((sum, size) => sum + Math.pow(size - avgSize, 2), 0) / blockSizes.length;
    
    // Reward lower variance (more balanced block sizes)
    return Math.max(0, 5 - variance);
  }

  countPotentialLineCompletions() {
    let potential = 0;
    
    // Check rows
    for (let row = 0; row < GRID_SIZE; row++) {
      let filled = 0;
      let available = 0;
      
      for (let col = 0; col < GRID_SIZE; col++) {
        if (!this.isBlockedCell(row, col)) {
          available++;
          if (this.grid[row][col]) filled++;
        }
      }
      
      // Higher threshold for potential completion
      if (available > 0 && filled / available > 0.75) { // Increased from 0.7 to 0.75
        potential++;
      }
    }
    
    // Check columns
    for (let col = 0; col < GRID_SIZE; col++) {
      let filled = 0;
      let available = 0;
      
      for (let row = 0; row < GRID_SIZE; row++) {
        if (!this.isBlockedCell(row, col)) {
          available++;
          if (this.grid[row][col]) filled++;
        }
      }
      
      if (available > 0 && filled / available > 0.75) {
        potential++;
      }
    }
    
    return potential;
  }

  /**
   * NEW IMPROVED: Dynamic action space size based on current state
   */
  getActionSpace() {
    // Return current number of valid actions (much smaller than 243)
    return this.getValidActions().length;
  }

  /**
   * NEW: Get maximum possible action space for network architecture
   */
  getMaxActionSpace() {
    // Theoretical maximum: 3 blocks × 7×7 positions = 147
    // But in practice it's usually much smaller
    return MAX_BLOCKS * 7 * 7; // 147 actions max
  }

  getStateSize() {
    return STATE_SIZE; // 112 dimensions
  }
} 