import * as tf from '@tensorflow/tfjs';

// VISUAL CNN CONFIGURATION - Optimized for CNN Learning
const GRID_SIZE = 45; // Large 45x45 grid for enhanced CNN spatial learning
const MAX_BLOCKS = 3;
const CHANNELS = 4; // Multi-channel visual representation

/**
 * VISUAL CNN-DQN ENVIRONMENT - ADVANCED SPATIAL PATTERN RECOGNITION
 * 
 * Revolutionary Features:
 * - 45x45 grid for massive CNN spatial learning capacity
 * - 4-channel visual representation (grid, blocks, potentials, strategy)
 * - Advanced block shapes designed for large space utilization
 * - Visual pattern recognition rewards
 * - Spatial relationship modeling
 * - Complex strategic depth with extensive playing field
 */
export class ConvDQNEnvironment {
  constructor() {
    this.reset();
    
    // CURRICULUM SYSTEM - Like DQN but for visual learning
    this.curriculumLevel = 0; // 0: Simple, 1: Medium, 2: Complex, 3: Advanced
    this.blockComplexityLevels = ['simple', 'medium', 'complex', 'advanced'];
    this.currentComplexity = this.blockComplexityLevels[this.curriculumLevel];
    
    // Visual reward configuration
    this.rewardConfig = {
      // Pattern completion rewards (higher for visual learning)
      patternCompletionBase: 20000,     // Base reward for visual patterns
      spatialEfficiencyBonus: 12000,    // Bonus for efficient space usage
      visualHarmonyBonus: 8000,         // Bonus for balanced visual layouts
      symmetryBonus: 3000,              // Bonus for symmetrical placements
      territoryControlBonus: 2500,      // Bonus for strategic position control
      
      // Placement rewards
      placementReward: 15,              // Points per block cell placed
      
      // Visual penalties
      visualClutterPenalty: -150,       // Penalty for creating visual mess
      asymmetryPenalty: -75,            // Penalty for destroying symmetry
      inefficiencyPenalty: -100,        // Penalty for poor space usage
      
      // Standard rewards
      survivalBonus: 2,                 // Small bonus per step
      gameOverPenalty: -8000           // Penalty for ending game
    };
    
    // Curriculum tracking
    this.curriculumStats = {
      episodesAtLevel: 0,
      patternsCompletedAtLevel: 0,
      avgScoreAtLevel: 0,
      advancementThreshold: 3           // Require 3 pattern completions to advance
    };
    
    console.log('🎨 VISUAL CNN ENVIRONMENT INITIALIZED - 45x45 PATTERN RECOGNITION MODE');
    console.log(`📐 Grid Size: ${GRID_SIZE}x${GRID_SIZE} (${GRID_SIZE * GRID_SIZE} cells)`);
    console.log(`🖼️  Visual Channels: ${CHANNELS} (grid, blocks, potentials, strategy)`);
    console.log(`🧩 Max Blocks: ${MAX_BLOCKS} with size up to ${GRID_SIZE}x${GRID_SIZE}`);
  }

  reset() {
    // 45x45 grid for better CNN spatial learning
    this.grid = Array(GRID_SIZE).fill(null).map(() => Array(GRID_SIZE).fill(false));
    this.availableBlocks = this.generateCurriculumBlocks();
    this.score = 0;
    this.difficulty = 'visual';
    this.movesSinceClear = 0;
    this.gameOver = false;
    this.totalMoves = 0;
    this.lineClearsThisEpisode = 0;
    this.patternsCompletedThisEpisode = 0;
    
    // Visual intelligence metrics
    this.visualMetrics = {
      spatialEfficiency: 0,
      visualHarmony: 0,
      symmetryScore: 0,
      territoryControl: 0,
      patternComplexity: 0
    };
    
    this.currentGridSize = GRID_SIZE;
    
    return this.getVisualState();
  }

  /**
   * CURRICULUM-BASED BLOCK GENERATION - Like DQN but for visual learning
   */
  generateCurriculumBlocks() {
    const blocks = [];
    const blockShapes = this.getCurriculumBlockShapes(this.currentComplexity);
    
    for (let i = 0; i < 3; i++) {
      const randomIndex = Math.floor(Math.random() * blockShapes.length);
      blocks.push(blockShapes[randomIndex]);
    }
    
    return blocks;
  }

  /**
   * CURRICULUM BLOCK SHAPES - Progressive complexity like DQN
   */
  getCurriculumBlockShapes(complexity) {
    const shapes = [];
    
    // Level 0: Simple blocks (1x1, 2x1, 2x2) - Same as DQN Level 0
    shapes.push([[true]]); // Single cell
    shapes.push([[true, true]]); // 2x1 horizontal
    shapes.push([[true], [true]]); // 2x1 vertical
    shapes.push([[true, true], [true, true]]); // 2x2 square
    
    if (complexity === 'simple') {
      return shapes;
    }
    
    // Level 1: Medium blocks (add 3x1, basic L-shapes) - Same as DQN Level 1
    shapes.push([[true, true, true]]); // 3x1 horizontal
    shapes.push([[true], [true], [true]]); // 3x1 vertical
    shapes.push([[true, true], [true, false]]); // Simple L-shape
    shapes.push([[true, false], [true, true]]); // Simple L-shape variant
    
    if (complexity === 'medium') {
      return shapes;
    }
    
    // Level 2: Complex blocks (add T-shapes, more L-shapes) - Same as DQN Level 2
    shapes.push([[true, true, true], [false, true, false]]); // T-shape
    shapes.push([[false, true], [true, true], [false, true]]); // T-shape rotated
    shapes.push([[true, false], [true, false], [true, true]]); // L-shape
    shapes.push([[true, true, true], [true, false, false]]); // L-shape
    
    if (complexity === 'complex') {
      return shapes;
    }
    
    // Level 3: Advanced blocks (more variety for larger grid)
    shapes.push([[true, false, false], [true, false, false], [true, true, true]]); // Corner L
    shapes.push([[true, true], [false, true], [false, true]]); // L variant
    shapes.push([[false, false, true], [true, true, true]]); // L variant
    shapes.push([[false, true, false], [true, true, true], [false, true, false]]); // Plus shape
    
    return shapes;
  }

  /**
   * VISUAL STATE REPRESENTATION - 4-Channel CNN Input
   */
  getVisualState() {
    // Create 4D array structure: [batch, height, width, channels]
    const batchData = [];
    
    // Generate all channels as 2D arrays first
    const gridChannel = this.generateGridChannel();
    const blocksChannel = this.generateBlocksChannel2D();
    const potentialsChannel = this.generatePotentialsChannel2D();
    const strategyChannel = this.generateStrategyChannel2D();
    
    // Build the tensor data in the correct format
    for (let h = 0; h < GRID_SIZE; h++) {
      const heightData = [];
      for (let w = 0; w < GRID_SIZE; w++) {
        const widthData = [
          gridChannel[h][w],      // Channel 0: Grid state
          blocksChannel[h][w],    // Channel 1: Block overlay
          potentialsChannel[h][w], // Channel 2: Line potentials
          strategyChannel[h][w]   // Channel 3: Strategic importance
        ];
        heightData.push(widthData);
      }
      batchData.push(heightData);
    }
    
    // Create tensor with correct shape: [1, height, width, channels]
    return tf.tensor4d([batchData], [1, GRID_SIZE, GRID_SIZE, CHANNELS]);
  }

  generateGridChannel() {
    const channel = [];
    for (let i = 0; i < GRID_SIZE; i++) {
      const row = [];
      for (let j = 0; j < GRID_SIZE; j++) {
        row.push(this.grid[i][j] ? 1 : 0);
      }
      channel.push(row);
    }
    return channel;
  }

  generateBlocksChannel2D() {
    const channel = Array(GRID_SIZE).fill(null).map(() => Array(GRID_SIZE).fill(0));
    
    // Mark potential placement areas for available blocks
    for (const block of this.availableBlocks) {
      for (let row = 0; row <= GRID_SIZE - block.length; row++) {
        for (let col = 0; col <= GRID_SIZE - block[0].length; col++) {
          if (this.canPlaceBlockAtPosition(block, row, col)) {
            for (let br = 0; br < block.length; br++) {
              for (let bc = 0; bc < block[br].length; bc++) {
                if (block[br][bc]) {
                  channel[row + br][col + bc] = Math.min(1, channel[row + br][col + bc] + 0.3);
                }
              }
            }
          }
        }
      }
    }
    
    return channel;
  }

  generatePotentialsChannel2D() {
    const channel = Array(GRID_SIZE).fill(null).map(() => Array(GRID_SIZE).fill(0));
    
    // Mark cells that could complete lines
    for (let row = 0; row < GRID_SIZE; row++) {
      for (let col = 0; col < GRID_SIZE; col++) {
        if (!this.grid[row][col]) {
          let potential = 0;
          
          // Row completion potential
          const rowFilled = this.grid[row].filter(cell => cell).length;
          if (rowFilled >= GRID_SIZE - 2) {
            potential += 0.8;
          } else if (rowFilled >= GRID_SIZE - 4) {
            potential += 0.4;
          }
          
          // Column completion potential
          let colFilled = 0;
          for (let r = 0; r < GRID_SIZE; r++) {
            if (this.grid[r][col]) colFilled++;
          }
          if (colFilled >= GRID_SIZE - 2) {
            potential += 0.8;
          } else if (colFilled >= GRID_SIZE - 4) {
            potential += 0.4;
          }
          
          channel[row][col] = Math.min(1, potential);
        }
      }
    }
    
    return channel;
  }

  generateStrategyChannel2D() {
    const channel = Array(GRID_SIZE).fill(null).map(() => Array(GRID_SIZE).fill(0));
    
    for (let row = 0; row < GRID_SIZE; row++) {
      for (let col = 0; col < GRID_SIZE; col++) {
        let strategic = 0;
        
        // Corner control (high value)
        if ((row === 0 || row === GRID_SIZE - 1) && (col === 0 || col === GRID_SIZE - 1)) {
          strategic += 0.9;
        }
        // Edge control (medium value)
        else if (row === 0 || row === GRID_SIZE - 1 || col === 0 || col === GRID_SIZE - 1) {
          strategic += 0.6;
        }
        // Center control (high value for larger boards)
        else if (row >= GRID_SIZE / 3 && row < 2 * GRID_SIZE / 3 && 
                 col >= GRID_SIZE / 3 && col < 2 * GRID_SIZE / 3) {
          strategic += 0.7;
        }
        // Near center
        else {
          strategic += 0.3;
        }
        
        channel[row][col] = strategic;
      }
    }
    
    return channel;
  }

  /**
   * COMPATIBILITY METHOD - For non-CNN agents
   */
  getState() {
    // Flatten visual state for compatibility
    const visualState = this.getVisualState();
    const flattened = visualState.reshape([GRID_SIZE * GRID_SIZE * CHANNELS]);
    visualState.dispose();
    return flattened;
  }

  getVisualStateSize() {
    return [CHANNELS, GRID_SIZE, GRID_SIZE]; // For CNN
  }

  getStateSize() {
    return GRID_SIZE * GRID_SIZE * CHANNELS; // For flattened state
  }

  getMaxActionSpace() {
    return GRID_SIZE * GRID_SIZE * MAX_BLOCKS; // 432 max actions
  }

  getValidActions() {
    const validActions = [];
    
    for (let blockIndex = 0; blockIndex < this.availableBlocks.length; blockIndex++) {
      const block = this.availableBlocks[blockIndex];
      
      for (let row = 0; row <= GRID_SIZE - block.length; row++) {
        for (let col = 0; col <= GRID_SIZE - block[0].length; col++) {
          if (this.canPlaceBlockAtPosition(block, row, col)) {
            const actionId = row * GRID_SIZE * MAX_BLOCKS + col * MAX_BLOCKS + blockIndex;
            validActions.push(actionId);
          }
        }
      }
    }
    
    return validActions;
  }

  canPlaceBlockAtPosition(blockShape, startRow, startCol) {
    for (let row = 0; row < blockShape.length; row++) {
      for (let col = 0; col < blockShape[row].length; col++) {
        if (blockShape[row][col]) {
          const gridRow = startRow + row;
          const gridCol = startCol + col;
          
          if (gridRow >= GRID_SIZE || gridCol >= GRID_SIZE || this.grid[gridRow][gridCol]) {
            return false;
          }
        }
      }
    }
    return true;
  }

  countLinesCleared(oldGrid, newGrid) {
    let linesCleared = 0;
    
    // Check rows
    for (let row = 0; row < GRID_SIZE; row++) {
      if (oldGrid[row].every(cell => cell) && newGrid[row].every(cell => !cell)) {
        linesCleared++;
      }
    }
    
    // Check columns
    for (let col = 0; col < GRID_SIZE; col++) {
      if (oldGrid.every(row => row[col]) && newGrid.every(row => !row[col])) {
        linesCleared++;
      }
    }
    
    return linesCleared;
  }

  step(actionId) {
    const oldGrid = this.grid.map(row => [...row]);
    const oldScore = this.score;
    
    // Decode action
    const blockIndex = actionId % MAX_BLOCKS;
    const col = Math.floor(actionId / MAX_BLOCKS) % GRID_SIZE;
    const row = Math.floor(actionId / (MAX_BLOCKS * GRID_SIZE));
    
    // Validate action
    if (blockIndex >= this.availableBlocks.length) {
      return {
        state: this.getVisualState(),
        reward: -1000,
        done: true,
        info: { error: 'Invalid block index' }
      };
    }

    const block = this.availableBlocks[blockIndex];
    
    if (!this.canPlaceBlockAtPosition(block, row, col)) {
      return {
        state: this.getVisualState(),
        reward: -1000,
        done: true,
        info: { error: 'Invalid placement' }
      };
    }

    // Place block
    const blockSize = this.placeBlock(block, row, col, blockIndex);
    this.totalMoves++;
    this.movesSinceClear++;
    
    // Clear lines and update score
    const newGrid = this.clearCompletedLines(this.grid);
    const linesCleared = this.countLinesCleared(oldGrid, newGrid);
    
    if (linesCleared > 0) {
      this.movesSinceClear = 0;
      this.grid = newGrid;
      this.score += linesCleared * 100 * (1 + linesCleared); // Standard scoring like DQN
      this.lineClearsThisEpisode += linesCleared;
      this.patternsCompletedThisEpisode += linesCleared;
    }
    
    // Remove used block
    this.availableBlocks.splice(blockIndex, 1);
    
    // Check if need new blocks
    if (this.availableBlocks.length === 0) {
      this.availableBlocks = this.generateCurriculumBlocks();
    }
    
    // Check game over
    this.gameOver = this.checkGameOver();
    
    // Calculate visual reward
    const reward = this.calculateVisualReward(oldGrid, oldScore, blockSize, linesCleared);
    
    // Update visual metrics
    this.updateVisualMetrics();
    
    return {
      state: this.getVisualState(),
      reward: reward,
      done: this.gameOver,
      info: {
        linesCleared: linesCleared,
        score: this.score,
        moves: this.totalMoves,
        visualMetrics: this.visualMetrics
      }
    };
  }

  calculateVisualReward(oldGrid, oldScore, blockSize, linesCleared) {
    let totalReward = 0;
    
    // Pattern completion rewards (like DQN but with visual bonuses)
    if (linesCleared > 0) {
      const baseReward = this.rewardConfig.patternCompletionBase;
      const lineReward = linesCleared * 5000; // Bonus per line
      const comboReward = linesCleared > 1 ? Math.pow(linesCleared, 2) * 8000 : 0;
      
      totalReward = baseReward + lineReward + comboReward;
      
      console.log(`🎨 VISUAL PATTERN COMPLETION! Lines: ${linesCleared}, Total: ${totalReward}`);
    }
    
    // Placement reward (like DQN)
    totalReward += blockSize * this.rewardConfig.placementReward;
    
    // Visual intelligence bonuses
    const visualAnalysis = this.analyzeVisualIntelligence();
    
    // Spatial efficiency bonus
    if (visualAnalysis.spatialEfficiency > 0.7) {
      totalReward += this.rewardConfig.spatialEfficiencyBonus * visualAnalysis.spatialEfficiency;
    }
    
    // Visual harmony bonus
    if (visualAnalysis.visualHarmony > 0.6) {
      totalReward += this.rewardConfig.visualHarmonyBonus * visualAnalysis.visualHarmony;
    }
    
    // Symmetry bonus
    if (visualAnalysis.symmetryScore > 0.5) {
      totalReward += this.rewardConfig.symmetryBonus * visualAnalysis.symmetryScore;
    }
    
    // Territory control bonus
    if (visualAnalysis.territoryControl > 0.4) {
      totalReward += this.rewardConfig.territoryControlBonus * visualAnalysis.territoryControl;
    }
    
    // Survival bonus
    if (!this.gameOver) {
      totalReward += this.rewardConfig.survivalBonus;
    } else {
      totalReward += this.rewardConfig.gameOverPenalty;
    }
    
    // Curriculum multiplier
    const curriculumMultiplier = 1 + (this.curriculumLevel * 0.2);
    totalReward *= curriculumMultiplier;
    
    return Math.max(-10000, Math.min(100000, totalReward));
  }

  analyzeVisualIntelligence() {
    // Simplified visual analysis for CNN learning
    let filledCells = 0;
    let totalCells = GRID_SIZE * GRID_SIZE;
    let edgeCells = 0;
    let centerCells = 0;
    
    for (let row = 0; row < GRID_SIZE; row++) {
      for (let col = 0; col < GRID_SIZE; col++) {
        if (this.grid[row][col]) {
          filledCells++;
          
          // Count edge vs center occupation
          if (row === 0 || row === GRID_SIZE - 1 || col === 0 || col === GRID_SIZE - 1) {
            edgeCells++;
          } else if (row >= GRID_SIZE / 3 && row < 2 * GRID_SIZE / 3 && 
                     col >= GRID_SIZE / 3 && col < 2 * GRID_SIZE / 3) {
            centerCells++;
          }
        }
      }
    }
    
    return {
      spatialEfficiency: filledCells / totalCells,
      visualHarmony: Math.min(edgeCells, centerCells) / Math.max(edgeCells, centerCells + 1),
      symmetryScore: this.calculateSimpleSymmetry(),
      territoryControl: (edgeCells + centerCells) / filledCells || 0,
      patternComplexity: this.curriculumLevel / 3
    };
  }

  calculateSimpleSymmetry() {
    let symmetryScore = 0;
    const center = Math.floor(GRID_SIZE / 2);
    
    // Check horizontal symmetry
    for (let row = 0; row < GRID_SIZE; row++) {
      for (let col = 0; col < center; col++) {
        const leftCell = this.grid[row][col];
        const rightCell = this.grid[row][GRID_SIZE - 1 - col];
        if (leftCell === rightCell) {
          symmetryScore += 1;
        }
      }
    }
    
    return symmetryScore / (GRID_SIZE * center);
  }

  updateVisualMetrics() {
    const analysis = this.analyzeVisualIntelligence();
    this.visualMetrics = analysis;
  }

  placeBlock(blockShape, startRow, startCol, blockIndex) {
    let blockSize = 0;
    
    for (let row = 0; row < blockShape.length; row++) {
      for (let col = 0; col < blockShape[row].length; col++) {
        if (blockShape[row][col]) {
          this.grid[startRow + row][startCol + col] = true;
          blockSize++;
        }
      }
    }
    
    return blockSize;
  }

  clearCompletedLines(grid) {
    const newGrid = grid.map(row => [...row]);
    
    // Clear completed rows
    for (let row = 0; row < GRID_SIZE; row++) {
      if (newGrid[row].every(cell => cell)) {
        newGrid[row].fill(false);
      }
    }
    
    // Clear completed columns
    for (let col = 0; col < GRID_SIZE; col++) {
      if (newGrid.every(row => row[col])) {
        for (let row = 0; row < GRID_SIZE; row++) {
          newGrid[row][col] = false;
        }
      }
    }
    
    return newGrid;
  }

  checkGameOver() {
    return this.getValidActions().length === 0;
  }

  /**
   * CURRICULUM LEARNING - Like DQN but for visual patterns
   */
  updateCurriculum(patternsCompletedThisEpisode, episodeScore) {
    this.curriculumStats.episodesAtLevel++;
    this.curriculumStats.patternsCompletedAtLevel += patternsCompletedThisEpisode;
    this.curriculumStats.avgScoreAtLevel = 
      (this.curriculumStats.avgScoreAtLevel * (this.curriculumStats.episodesAtLevel - 1) + episodeScore) / 
      this.curriculumStats.episodesAtLevel;
    
    // Check for advancement
    const canAdvance = 
      this.curriculumLevel < this.blockComplexityLevels.length - 1 && // Not at max level
      this.curriculumStats.patternsCompletedAtLevel >= this.curriculumStats.advancementThreshold && // Enough patterns
      this.curriculumStats.episodesAtLevel >= 5; // Minimum episodes at level
    
    if (canAdvance) {
      this.curriculumLevel++;
      this.currentComplexity = this.blockComplexityLevels[this.curriculumLevel];
      
      console.log(`🎨 VISUAL CURRICULUM ADVANCED! Level ${this.curriculumLevel}: ${this.currentComplexity} blocks on ${GRID_SIZE}x${GRID_SIZE} grid`);
      console.log(`📊 Previous level stats: ${this.curriculumStats.patternsCompletedAtLevel} patterns in ${this.curriculumStats.episodesAtLevel} episodes, avg score: ${this.curriculumStats.avgScoreAtLevel.toFixed(1)}`);
      
      // Reset stats for new level
      this.curriculumStats = {
        episodesAtLevel: 0,
        patternsCompletedAtLevel: 0,
        avgScoreAtLevel: 0,
        advancementThreshold: Math.min(6, this.curriculumStats.advancementThreshold + 1)
      };
      
      return true;
    }
    
    return false;
  }

  // Compatibility methods
  setState(grid, availableBlocks, score, difficulty = 'visual') {
    // Adapt to 45x45 grid
    this.grid = Array(GRID_SIZE).fill(null).map((_, row) => 
      Array(GRID_SIZE).fill(null).map((_, col) => {
        if (row < grid.length && col < grid[0].length) {
          return grid[row][col];
        }
        return false;
      })
    );
    
    this.availableBlocks = this.generateCurriculumBlocks();
    this.score = score;
    this.difficulty = difficulty;
    this.gameOver = this.checkGameOver();
    this.currentGridSize = GRID_SIZE;
    return this.getVisualState();
  }

  generateVisualBlocks() {
    return this.generateCurriculumBlocks();
  }

  decodeAction(actionId) {
    const blockIndex = actionId % MAX_BLOCKS;
    const col = Math.floor(actionId / MAX_BLOCKS) % GRID_SIZE;
    const row = Math.floor(actionId / (MAX_BLOCKS * GRID_SIZE));
    
    return { blockIndex, row, col };
  }

  clone() {
    const cloned = new ConvDQNEnvironment();
    
    cloned.grid = this.grid.map(row => [...row]);
    cloned.availableBlocks = this.availableBlocks.map(block => 
      block.map(row => [...row])
    );
    cloned.score = this.score;
    cloned.difficulty = this.difficulty;
    cloned.movesSinceClear = this.movesSinceClear;
    cloned.gameOver = this.gameOver;
    cloned.totalMoves = this.totalMoves;
    cloned.lineClearsThisEpisode = this.lineClearsThisEpisode;
    cloned.patternsCompletedThisEpisode = this.patternsCompletedThisEpisode;
    cloned.curriculumLevel = this.curriculumLevel;
    cloned.currentComplexity = this.currentComplexity;
    cloned.currentGridSize = this.currentGridSize;
    cloned.visualMetrics = { ...this.visualMetrics };
    
    return cloned;
  }
} 