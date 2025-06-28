import * as tf from '@tensorflow/tfjs';

// DQN CONFIGURATION - Optimized for Learning

const GRID_SIZE = 9; // FIXED: Always 9x9 for neural network consistency
const MAX_BLOCKS = 3;
const MAX_BLOCK_SIZE = 3; // 3x3 maximum block dimensions

export class DQNEnvironment {
  constructor() {
    this.reset();
    
    // FIXED CURRICULUM SYSTEM: 9x9 grid always, but progressive block complexity
    this.curriculumLevel = 0; // 0: Simple blocks, 1: Medium blocks, 2: Complex blocks, 3: All blocks
    this.blockComplexityLevels = ['simple', 'medium', 'complex', 'full'];
    this.currentComplexity = this.blockComplexityLevels[this.curriculumLevel];
    
    // CREATIVE PENALTY-BASED REWARD SYSTEM
    this.rewardConfig = {
      // MASSIVE line clearing rewards (keep these high)
      lineClearBase: 10000,      // Base reward for any line clear
      lineMultiplier: 5000,      // Additional reward per line cleared
      comboMultiplier: 15000,    // Massive bonus for multiple lines
      
      // REALISTIC placement rewards (like actual game)
      placementReward: 10,       // Points per block cell placed (realistic)
      
      // CREATIVE SPATIAL PENALTIES (the innovation!)
      isolatedCellPenalty: -100,     // Penalty for cells with no neighbors
      cornerWastePenalty: -50,       // Penalty for leaving corners empty when they could be filled
      deadSpacePenalty: -75,         // Penalty for creating unfillable spaces
      inefficientGapPenalty: -25,    // Penalty for creating awkward gaps
      edgeWastePenalty: -30,         // Penalty for not utilizing edges efficiently
      fragmentationPenalty: -40,     // Penalty for creating disconnected regions
      
      // Bonuses for good spatial organization
      compactnessBonus: 20,          // Bonus for keeping filled areas compact
      edgeUtilizationBonus: 15,      // Bonus for efficiently using edges
      cornerUtilizationBonus: 25,    // Bonus for smart corner usage
      
      // Survival and game over
      survivalBonus: 1,              // Tiny bonus per step alive
      gameOverPenalty: -5000         // Major penalty for ending game
    };
    
    // Performance tracking for curriculum advancement
    this.curriculumStats = {
      episodesAtLevel: 0,
      lineClearsAtLevel: 0,
      avgScoreAtLevel: 0,
      advancementThreshold: 5    // Require 5 line clears to advance
    };
  }

  reset() {
    // FIXED: Always use full 9x9 grid for neural network consistency
    this.grid = Array(GRID_SIZE).fill(null).map(() => Array(GRID_SIZE).fill(false));
    this.availableBlocks = this.generateCurriculumBlocks();
    this.score = 0;
    this.difficulty = 'normal';
    this.movesSinceClear = 0;
    this.gameOver = false;
    this.totalMoves = 0;
    this.lineClearsThisEpisode = 0;
    
    // Add current grid size for display purposes (always 9)
    this.currentGridSize = GRID_SIZE;
    
    return this.getState();
  }

  /**
   * FIXED CURRICULUM: Generate blocks with progressive complexity on 9x9 grid
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
   * FIXED CURRICULUM: Block shapes with progressive complexity
   */
  getCurriculumBlockShapes(complexity) {
    const shapes = [];
    
    // Level 0: Simple blocks (1x1, 2x1, 2x2)
    shapes.push([[true]]); // Single cell
    shapes.push([[true, true]]); // 2x1 horizontal
    shapes.push([[true], [true]]); // 2x1 vertical
    
    if (complexity === 'simple') {
      shapes.push([[true, true], [true, true]]); // 2x2 square
      return shapes;
    }
    
    // Level 1: Medium blocks (add 3x1, basic L-shapes)
    shapes.push([[true, true], [true, true]]); // 2x2 square
    shapes.push([[true, true, true]]); // 3x1 horizontal
    shapes.push([[true], [true], [true]]); // 3x1 vertical
    shapes.push([[true, true], [true, false]]); // Simple L-shape
    shapes.push([[true, false], [true, true]]); // Simple L-shape variant
    
    if (complexity === 'medium') {
      return shapes;
    }
    
    // Level 2: Complex blocks (add T-shapes, more L-shapes)
    shapes.push([[true, true, true], [false, true, false]]); // T-shape
    shapes.push([[false, true], [true, true], [false, true]]); // T-shape rotated
    shapes.push([[true, false], [true, false], [true, true]]); // L-shape
    shapes.push([[true, true, true], [true, false, false]]); // L-shape
    shapes.push([[true, true], [false, true], [false, true]]); // L-shape
    shapes.push([[false, false, true], [true, true, true]]); // L-shape
    
    if (complexity === 'complex') {
      return shapes;
    }
    
    // Level 3: Full game blocks (all shapes from gameLogic)
    return this.getAllBlockShapes();
  }

  /**
   * Get all block shapes for full complexity
   */
  getAllBlockShapes() {
    return [
      // Single cell
      [[true]],
      
      // 2x1 blocks
      [[true, true]],
      [[true], [true]],
      
      // 3x1 blocks
      [[true, true, true]],
      [[true], [true], [true]],
      
      // L-shapes
      [[true, false], [true, false], [true, true]],
      [[true, true, true], [true, false, false]],
      [[true, true], [false, true], [false, true]],
      [[false, false, true], [true, true, true]],
      
      // T-shapes
      [[true, true, true], [false, true, false]],
      [[false, true], [true, true], [false, true]],
      [[false, true, false], [true, true, true]],
      [[true, false], [true, true], [true, false]],
      
      // Squares
      [[true, true], [true, true]],
      
      // Z-shapes
      [[true, true, false], [false, true, true]],
      [[false, true], [true, true], [true, false]],
      
      // Plus shape
      [[false, true, false], [true, true, true], [false, true, false]],
      
      // Corner shapes
      [[true, false, false], [true, false, false], [true, true, true]],
      [[true, true, true], [true, false, false], [true, false, false]],
      [[true, true, true], [false, false, true], [false, false, true]],
      [[false, false, true], [false, false, true], [true, true, true]],
    ];
  }

  setState(grid, availableBlocks, score, difficulty = 'normal') {
    // FIXED: Always use full 9x9 grid, pad if necessary
    this.grid = Array(GRID_SIZE).fill(null).map((_, row) => 
      Array(GRID_SIZE).fill(null).map((_, col) => {
        if (row < grid.length && col < grid[0].length) {
          return grid[row][col];
        }
        return false;
      })
    );
    
    // Use curriculum-appropriate blocks
    this.availableBlocks = this.generateCurriculumBlocks();
    this.score = score;
    this.difficulty = difficulty;
    this.gameOver = this.checkGameOver();
    this.currentGridSize = GRID_SIZE; // Always 9 for display
    return this.getState();
  }

  /**
   * FIXED STATE REPRESENTATION: Always 9x9 grid
   */
  getState() {
    // 1. Full 9x9 grid state (no padding needed)
    const gridState = [];
    for (let i = 0; i < GRID_SIZE; i++) {
      for (let j = 0; j < GRID_SIZE; j++) {
        gridState.push(this.grid[i][j] ? 1 : 0);
      }
    }
    
    // 2. LINE COMPLETION PROGRESS FEATURES (for full 9x9 grid)
    const lineFeatures = [];
    
    // Row completion progress
    for (let row = 0; row < GRID_SIZE; row++) {
      const filled = this.grid[row].filter(cell => cell).length;
      lineFeatures.push(filled / GRID_SIZE);
    }
    
    // Column completion progress
    for (let col = 0; col < GRID_SIZE; col++) {
      let filled = 0;
      for (let row = 0; row < GRID_SIZE; row++) {
        if (this.grid[row][col]) filled++;
      }
      lineFeatures.push(filled / GRID_SIZE);
    }
    
    // 3x3 square completion progress
    for (let sq_row = 0; sq_row < 3; sq_row++) {
      for (let sq_col = 0; sq_col < 3; sq_col++) {
        let filled = 0;
        for (let r = sq_row * 3; r < (sq_row + 1) * 3; r++) {
          for (let c = sq_col * 3; c < (sq_col + 1) * 3; c++) {
            if (this.grid[r][c]) filled++;
          }
        }
        lineFeatures.push(filled / 9); // Always 9 cells per 3x3 square
      }
    }
    
    // 3. Block encoding (unchanged)
    const blockState = [];
    for (let i = 0; i < MAX_BLOCKS; i++) {
      if (i < this.availableBlocks.length) {
        const block = this.availableBlocks[i];
        for (let r = 0; r < MAX_BLOCK_SIZE; r++) {
          for (let c = 0; c < MAX_BLOCK_SIZE; c++) {
            if (r < block.length && c < block[r].length) {
              blockState.push(block[r][c] ? 1 : 0);
            } else {
              blockState.push(0);
            }
          }
        }
      } else {
        // No block available
        for (let j = 0; j < MAX_BLOCK_SIZE * MAX_BLOCK_SIZE; j++) {
          blockState.push(0);
        }
      }
    }
    
    // 4. Meta features
    const metaFeatures = [
      this.score / 10000, // Normalized score
      this.movesSinceClear / 20, // Moves since last clear
      this.availableBlocks.length / 3, // Block availability
      this.curriculumLevel / 3, // Current curriculum level
    ];
    
    const fullState = [...gridState, ...lineFeatures, ...blockState, ...metaFeatures];
    return tf.tensor1d(fullState);
  }

  /**
   * FIXED LINE COUNTING: Always for 9x9 grid
   */
  countLinesCleared(oldGrid, newGrid) {
    let linesCleared = 0;
    
    // Count cleared rows
    for (let row = 0; row < GRID_SIZE; row++) {
      let wasComplete = true;
      let nowEmpty = true;
      
      for (let col = 0; col < GRID_SIZE; col++) {
        if (!oldGrid[row][col]) wasComplete = false;
        if (newGrid[row][col]) nowEmpty = false;
      }
      
      if (wasComplete && nowEmpty) {
        linesCleared++;
        console.log(`✅ Row ${row} cleared!`);
      }
    }
    
    // Count cleared columns
    for (let col = 0; col < GRID_SIZE; col++) {
      let wasComplete = true;
      let nowEmpty = true;
      
      for (let row = 0; row < GRID_SIZE; row++) {
        if (!oldGrid[row][col]) wasComplete = false;
        if (newGrid[row][col]) nowEmpty = false;
      }
      
      if (wasComplete && nowEmpty) {
        linesCleared++;
        console.log(`✅ Column ${col} cleared!`);
      }
    }
    
    // Count cleared 3x3 squares
    for (let sq_row = 0; sq_row < 3; sq_row++) {
      for (let sq_col = 0; sq_col < 3; sq_col++) {
        let wasComplete = true;
        let nowEmpty = true;
        
        for (let r = sq_row * 3; r < (sq_row + 1) * 3; r++) {
          for (let c = sq_col * 3; c < (sq_col + 1) * 3; c++) {
            if (!oldGrid[r][c]) wasComplete = false;
            if (newGrid[r][c]) nowEmpty = false;
          }
        }
        
        if (wasComplete && nowEmpty) {
          linesCleared++;
          console.log(`✅ Square (${sq_row}, ${sq_col}) cleared!`);
        }
      }
    }
    
    return linesCleared;
  }

  /**
   * FIXED ACTION SPACE: Always for 9x9 grid
   */
  getValidActions() {
    const validActions = [];
    
    for (let blockIdx = 0; blockIdx < this.availableBlocks.length; blockIdx++) {
      const block = this.availableBlocks[blockIdx];
      
      const blockHeight = block.length;
      const blockWidth = Math.max(...block.map(row => row.length));
      
      const maxRow = GRID_SIZE - blockHeight;
      const maxCol = GRID_SIZE - blockWidth;
      
      for (let row = 0; row <= maxRow; row++) {
        for (let col = 0; col <= maxCol; col++) {
          if (this.canPlaceBlockAtPosition(block, row, col)) {
            const actionId = blockIdx * 1000 + row * 10 + col;
            validActions.push(actionId);
          }
        }
      }
    }
    
    return validActions;
  }

  /**
   * FIXED: Block placement for 9x9 grid
   */
  canPlaceBlockAtPosition(blockShape, startRow, startCol) {
    for (let row = 0; row < blockShape.length; row++) {
      for (let col = 0; col < blockShape[row].length; col++) {
        if (blockShape[row][col]) {
          const gridRow = startRow + row;
          const gridCol = startCol + col;
          
          if (gridRow < 0 || gridRow >= GRID_SIZE || gridCol < 0 || gridCol >= GRID_SIZE) {
            return false;
          }
          
          if (this.grid[gridRow][gridCol]) {
            return false;
          }
        }
      }
    }
    
    return true;
  }

  /**
   * FIXED: Check game over for 9x9 grid
   */
  checkGameOver() {
    for (const block of this.availableBlocks) {
      for (let row = 0; row <= GRID_SIZE - block.length; row++) {
        for (let col = 0; col <= GRID_SIZE - Math.max(...block.map(r => r.length)); col++) {
          if (this.canPlaceBlockAtPosition(block, row, col)) {
            return false; // Game can continue
          }
        }
      }
    }
    
    return true; // Game over
  }

  /**
   * FIXED: Line clearing for 9x9 grid
   */
  clearCompletedLines(newGrid) {
    let totalCleared = 0;
    const gridCopy = newGrid.map(row => [...row]);
    
    // Clear completed rows
    for (let row = 0; row < GRID_SIZE; row++) {
      let isComplete = true;
      for (let col = 0; col < GRID_SIZE; col++) {
        if (!gridCopy[row][col]) {
          isComplete = false;
          break;
        }
      }
      
      if (isComplete) {
        totalCleared++;
        for (let col = 0; col < GRID_SIZE; col++) {
          gridCopy[row][col] = false;
        }
      }
    }
    
    // Clear completed columns
    for (let col = 0; col < GRID_SIZE; col++) {
      let isComplete = true;
      for (let row = 0; row < GRID_SIZE; row++) {
        if (!gridCopy[row][col]) {
          isComplete = false;
          break;
        }
      }
      
      if (isComplete) {
        totalCleared++;
        for (let row = 0; row < GRID_SIZE; row++) {
          gridCopy[row][col] = false;
        }
      }
    }
    
    // Clear completed 3x3 squares
    for (let sq_row = 0; sq_row < 3; sq_row++) {
      for (let sq_col = 0; sq_col < 3; sq_col++) {
        let isComplete = true;
        
        for (let r = sq_row * 3; r < (sq_row + 1) * 3; r++) {
          for (let c = sq_col * 3; c < (sq_col + 1) * 3; c++) {
            if (!gridCopy[r][c]) {
              isComplete = false;
              break;
            }
          }
          if (!isComplete) break;
        }
        
        if (isComplete) {
          totalCleared++;
          for (let r = sq_row * 3; r < (sq_row + 1) * 3; r++) {
            for (let c = sq_col * 3; c < (sq_col + 1) * 3; c++) {
              gridCopy[r][c] = false;
            }
          }
        }
      }
    }
    
    // Score calculation
    if (totalCleared > 0) {
      const basePoints = 100;
      let points = totalCleared * basePoints;
      
      if (totalCleared > 1) {
        points += Math.pow(totalCleared, 2) * 50;
      }
      
      this.score += points;
      this.movesSinceClear = 0;
    } else {
      this.movesSinceClear++;
    }
    
    return gridCopy;
  }

  placeBlock(blockShape, startRow, startCol, blockIndex) {
    // Place the block on 9x9 grid
    for (let row = 0; row < blockShape.length; row++) {
      for (let col = 0; col < blockShape[row].length; col++) {
        if (blockShape[row][col]) {
          const gridRow = startRow + row;
          const gridCol = startCol + col;
          
          if (gridRow >= 0 && gridRow < GRID_SIZE && gridCol >= 0 && gridCol < GRID_SIZE) {
            this.grid[gridRow][gridCol] = true;
          }
        }
      }
    }
    
    // Remove the used block
    this.availableBlocks.splice(blockIndex, 1);
    
    // Clear completed lines
    this.grid = this.clearCompletedLines(this.grid);
    
    // Check if we need new blocks
    if (this.availableBlocks.length === 0) {
      this.availableBlocks = this.generateCurriculumBlocks();
    }
    
    // Check game over
    this.gameOver = this.checkGameOver();
    this.totalMoves++;
  }

  getStateSize() {
    // FIXED: Grid (81) + line features (27) + block features (27) + meta features (4) = 139
    return 81 + 27 + 27 + 4; // 139 total
  }

  getMaxActionSpace() {
    return 3 * 9 * 9; // 243 max actions for 9x9 grid
  }

  getActionSpace() {
    return this.getMaxActionSpace();
  }

  /**
   * FIXED CURRICULUM: Update based on block complexity progression
   */
  updateCurriculum(lineClearsThisEpisode, episodeScore) {
    this.curriculumStats.episodesAtLevel++;
    this.curriculumStats.lineClearsAtLevel += lineClearsThisEpisode;
    this.curriculumStats.avgScoreAtLevel = 
      (this.curriculumStats.avgScoreAtLevel * (this.curriculumStats.episodesAtLevel - 1) + episodeScore) / 
      this.curriculumStats.episodesAtLevel;
    
    // Check for advancement (easier criteria for block complexity progression)
    const canAdvance = 
      this.curriculumLevel < this.blockComplexityLevels.length - 1 && // Not at max level
      this.curriculumStats.lineClearsAtLevel >= this.curriculumStats.advancementThreshold && // Enough line clears
      this.curriculumStats.episodesAtLevel >= 5; // Minimum episodes at level
    
    if (canAdvance) {
      this.curriculumLevel++;
      this.currentComplexity = this.blockComplexityLevels[this.curriculumLevel];
      
      console.log(`🎓 BLOCK COMPLEXITY ADVANCEMENT! Level ${this.curriculumLevel}: ${this.currentComplexity} blocks on 9x9 grid`);
      console.log(`📊 Previous level stats: ${this.curriculumStats.lineClearsAtLevel} line clears in ${this.curriculumStats.episodesAtLevel} episodes, avg score: ${this.curriculumStats.avgScoreAtLevel.toFixed(1)}`);
      
      // Reset stats for new level
      this.curriculumStats = {
        episodesAtLevel: 0,
        lineClearsAtLevel: 0,
        avgScoreAtLevel: 0,
        advancementThreshold: Math.min(8, this.curriculumStats.advancementThreshold + 1) // Gradual increase
      };
      
      return true; // Signal that curriculum advanced
    }
    
    return false;
  }

  /**
   * CREATIVE PENALTY-BASED REWARD SYSTEM
   */
  calculateReward(oldGrid, oldScore, blockSize) {
    // 1. COUNT ACTUAL LINES CLEARED (keep massive rewards)
    const linesCleared = this.countLinesCleared(oldGrid, this.grid);
    let totalReward = 0;
    
    // Track line clears for curriculum
    this.lineClearsThisEpisode += linesCleared;
    
    // 2. MASSIVE LINE CLEARING REWARDS (unchanged)
    if (linesCleared > 0) {
      const baseReward = this.rewardConfig.lineClearBase;
      const lineReward = linesCleared * this.rewardConfig.lineMultiplier;
      const comboReward = linesCleared > 1 ? Math.pow(linesCleared, 2) * this.rewardConfig.comboMultiplier : 0;
      
      totalReward = baseReward + lineReward + comboReward;
      
      console.log(`🎯 MASSIVE LINE CLEAR REWARD! Lines: ${linesCleared}, Base: ${baseReward}, Line: ${lineReward}, Combo: ${comboReward}, Total: ${totalReward}`);
    }
    
    // 3. REALISTIC PLACEMENT REWARD (like actual game)
    const placementPoints = blockSize * this.rewardConfig.placementReward;
    totalReward += placementPoints;
    
    // 4. CREATIVE SPATIAL ANALYSIS & PENALTIES
    const spatialAnalysis = this.analyzeSpatialPatterns();
    const spatialPenalties = this.calculateSpatialPenalties(spatialAnalysis);
    const spatialBonuses = this.calculateSpatialBonuses(spatialAnalysis);
    
    totalReward += spatialPenalties + spatialBonuses;
    
    // 5. SURVIVAL
    if (!this.gameOver) {
      totalReward += this.rewardConfig.survivalBonus;
    } else {
      totalReward += this.rewardConfig.gameOverPenalty;
    }
    
    // 6. CURRICULUM BONUS
    const curriculumMultiplier = 1 + (this.curriculumLevel * 0.3);
    totalReward *= curriculumMultiplier;
    
    // Log detailed breakdown for significant rewards/penalties
    if (Math.abs(totalReward) > 50 || Math.abs(spatialPenalties) > 25) {
      console.log(`💰 CREATIVE REWARD BREAKDOWN:`);
      console.log(`  📦 Placement: +${placementPoints}`);
      console.log(`  🎯 Line Clears: +${linesCleared > 0 ? (this.rewardConfig.lineClearBase + linesCleared * this.rewardConfig.lineMultiplier) : 0}`);
      console.log(`  ❌ Spatial Penalties: ${spatialPenalties.toFixed(1)}`);
      console.log(`  ✅ Spatial Bonuses: ${spatialBonuses.toFixed(1)}`);
      console.log(`  🏆 Total: ${totalReward.toFixed(1)} (Level ${this.curriculumLevel}: ${this.currentComplexity})`);
      
      if (spatialAnalysis.isolatedCells > 0) console.log(`    🔸 Isolated cells: ${spatialAnalysis.isolatedCells}`);
      if (spatialAnalysis.wastedCorners > 0) console.log(`    🔸 Wasted corners: ${spatialAnalysis.wastedCorners}`);
      if (spatialAnalysis.deadSpaces > 0) console.log(`    🔸 Dead spaces: ${spatialAnalysis.deadSpaces}`);
      if (spatialAnalysis.inefficientGaps > 0) console.log(`    🔸 Inefficient gaps: ${spatialAnalysis.inefficientGaps}`);
    }
    
    return Math.max(-10000, Math.min(50000, totalReward));
  }

  /**
   * CREATIVE SPATIAL PATTERN ANALYSIS
   */
  analyzeSpatialPatterns() {
    const analysis = {
      isolatedCells: 0,
      wastedCorners: 0,
      deadSpaces: 0,
      inefficientGaps: 0,
      wastedEdges: 0,
      fragmentedRegions: 0,
      compactRegions: 0,
      edgeUtilization: 0,
      cornerUtilization: 0
    };
    
    // Analyze each cell and its spatial context
    for (let row = 0; row < GRID_SIZE; row++) {
      for (let col = 0; col < GRID_SIZE; col++) {
        if (this.grid[row][col]) {
          // Check for isolated cells (no adjacent neighbors)
          if (this.isIsolatedCell(row, col)) {
            analysis.isolatedCells++;
          }
          
          // Check edge and corner utilization
          if (this.isCorner(row, col)) {
            analysis.cornerUtilization++;
          }
          if (this.isEdge(row, col)) {
            analysis.edgeUtilization++;
          }
        } else {
          // Empty cell analysis
          
          // Check for wasted corners (corners that could be easily filled)
          if (this.isCorner(row, col) && this.isWastedCorner(row, col)) {
            analysis.wastedCorners++;
          }
          
          // Check for dead spaces (small gaps that can't be filled efficiently)
          if (this.isDeadSpace(row, col)) {
            analysis.deadSpaces++;
          }
          
          // Check for inefficient gaps (single empty cells surrounded by filled cells)
          if (this.isInefficientGap(row, col)) {
            analysis.inefficientGaps++;
          }
          
          // Check for wasted edges
          if (this.isEdge(row, col) && this.isWastedEdge(row, col)) {
            analysis.wastedEdges++;
          }
        }
      }
    }
    
    // Analyze fragmentation vs compactness
    const regions = this.findFilledRegions();
    analysis.fragmentedRegions = regions.filter(region => region.size < 4).length;
    analysis.compactRegions = regions.filter(region => region.size >= 4 && region.compactness > 0.7).length;
    
    return analysis;
  }

  /**
   * CREATIVE SPATIAL PENALTIES
   */
  calculateSpatialPenalties(analysis) {
    let penalties = 0;
    
    // Penalty for isolated cells (cells with no neighbors)
    penalties += analysis.isolatedCells * this.rewardConfig.isolatedCellPenalty;
    
    // Penalty for wasted corners
    penalties += analysis.wastedCorners * this.rewardConfig.cornerWastePenalty;
    
    // Penalty for dead spaces
    penalties += analysis.deadSpaces * this.rewardConfig.deadSpacePenalty;
    
    // Penalty for inefficient gaps
    penalties += analysis.inefficientGaps * this.rewardConfig.inefficientGapPenalty;
    
    // Penalty for wasted edges
    penalties += analysis.wastedEdges * this.rewardConfig.edgeWastePenalty;
    
    // Penalty for fragmentation
    penalties += analysis.fragmentedRegions * this.rewardConfig.fragmentationPenalty;
    
    return penalties;
  }

  /**
   * CREATIVE SPATIAL BONUSES
   */
  calculateSpatialBonuses(analysis) {
    let bonuses = 0;
    
    // Bonus for compact regions
    bonuses += analysis.compactRegions * this.rewardConfig.compactnessBonus;
    
    // Bonus for edge utilization
    const edgeUtilizationRatio = analysis.edgeUtilization / (GRID_SIZE * 4 - 4); // Total edge cells
    bonuses += edgeUtilizationRatio * this.rewardConfig.edgeUtilizationBonus;
    
    // Bonus for corner utilization
    const cornerUtilizationRatio = analysis.cornerUtilization / 4; // 4 corners total
    bonuses += cornerUtilizationRatio * this.rewardConfig.cornerUtilizationBonus;
    
    return bonuses;
  }

  /**
   * SPATIAL PATTERN DETECTION METHODS
   */
  isIsolatedCell(row, col) {
    const neighbors = this.getNeighbors(row, col);
    return neighbors.every(([r, c]) => !this.grid[r] || !this.grid[r][c]);
  }

  isWastedCorner(row, col) {
    if (!this.isCorner(row, col)) return false;
    
    // A corner is "wasted" if it's empty but has filled adjacent cells
    const neighbors = this.getNeighbors(row, col);
    const filledNeighbors = neighbors.filter(([r, c]) => this.grid[r] && this.grid[r][c]).length;
    
    return filledNeighbors >= 1; // Corner could be utilized
  }

  isDeadSpace(row, col) {
    // A dead space is a small isolated empty area that's hard to fill
    const emptyNeighbors = this.getNeighbors(row, col).filter(([r, c]) => 
      r >= 0 && r < GRID_SIZE && c >= 0 && c < GRID_SIZE && !this.grid[r][c]
    );
    
    const filledNeighbors = this.getNeighbors(row, col).filter(([r, c]) => 
      r >= 0 && r < GRID_SIZE && c >= 0 && c < GRID_SIZE && this.grid[r][c]
    );
    
    // Single empty cell surrounded by mostly filled cells
    return emptyNeighbors.length <= 1 && filledNeighbors.length >= 3;
  }

  isInefficientGap(row, col) {
    // Single empty cell completely surrounded by filled cells
    const neighbors = this.getNeighbors(row, col);
    const validNeighbors = neighbors.filter(([r, c]) => r >= 0 && r < GRID_SIZE && c >= 0 && c < GRID_SIZE);
    const filledNeighbors = validNeighbors.filter(([r, c]) => this.grid[r][c]);
    
    return filledNeighbors.length === validNeighbors.length && validNeighbors.length >= 3;
  }

  isWastedEdge(row, col) {
    if (!this.isEdge(row, col)) return false;
    
    // Edge is wasted if it's empty but could connect to existing structures
    const neighbors = this.getNeighbors(row, col);
    const filledNeighbors = neighbors.filter(([r, c]) => 
      r >= 0 && r < GRID_SIZE && c >= 0 && c < GRID_SIZE && this.grid[r][c]
    ).length;
    
    return filledNeighbors >= 2; // Edge could connect structures
  }

  /**
   * UTILITY METHODS FOR SPATIAL ANALYSIS
   */
  isCorner(row, col) {
    return (row === 0 || row === GRID_SIZE - 1) && (col === 0 || col === GRID_SIZE - 1);
  }

  isEdge(row, col) {
    return row === 0 || row === GRID_SIZE - 1 || col === 0 || col === GRID_SIZE - 1;
  }

  getNeighbors(row, col) {
    return [
      [row - 1, col], [row + 1, col],  // Vertical neighbors
      [row, col - 1], [row, col + 1]   // Horizontal neighbors
    ];
  }

  findFilledRegions() {
    const visited = Array(GRID_SIZE).fill(null).map(() => Array(GRID_SIZE).fill(false));
    const regions = [];
    
    for (let row = 0; row < GRID_SIZE; row++) {
      for (let col = 0; col < GRID_SIZE; col++) {
        if (this.grid[row][col] && !visited[row][col]) {
          const region = this.exploreRegion(row, col, visited);
          regions.push(region);
        }
      }
    }
    
    return regions;
  }

  exploreRegion(startRow, startCol, visited) {
    const region = { cells: [], size: 0, compactness: 0 };
    const stack = [[startRow, startCol]];
    
    while (stack.length > 0) {
      const [row, col] = stack.pop();
      
      if (row < 0 || row >= GRID_SIZE || col < 0 || col >= GRID_SIZE) continue;
      if (visited[row][col] || !this.grid[row][col]) continue;
      
      visited[row][col] = true;
      region.cells.push([row, col]);
      region.size++;
      
      // Add neighbors to explore
      const neighbors = this.getNeighbors(row, col);
      for (const [nr, nc] of neighbors) {
        if (nr >= 0 && nr < GRID_SIZE && nc >= 0 && nc < GRID_SIZE && 
            !visited[nr][nc] && this.grid[nr][nc]) {
          stack.push([nr, nc]);
        }
      }
    }
    
    // Calculate compactness (how close to square the region is)
    if (region.size > 1) {
      const minRow = Math.min(...region.cells.map(([r]) => r));
      const maxRow = Math.max(...region.cells.map(([r]) => r));
      const minCol = Math.min(...region.cells.map(([, c]) => c));
      const maxCol = Math.max(...region.cells.map(([, c]) => c));
      
      const boundingArea = (maxRow - minRow + 1) * (maxCol - minCol + 1);
      region.compactness = region.size / boundingArea;
    }
    
    return region;
  }

  // Essential compatibility methods
  isBlockedCell(row, col) {
    return this.difficulty === 'hard' && 
           row >= 3 && row <= 5 && 
           col >= 3 && col <= 5;
  }

  decodeAction(actionId) {
    const blockIndex = Math.floor(actionId / 1000);
    const remainder = actionId % 1000;
    const row = Math.floor(remainder / 10);
    const col = remainder % 10;
    
    return { blockIndex, row, col };
  }

  step(actionId) {
    const { blockIndex, row, col } = this.decodeAction(actionId);
    
    if (blockIndex >= this.availableBlocks.length) {
      return {
        state: this.getState(),
        reward: -100,
        done: true
      };
    }
    
    const blockShape = this.availableBlocks[blockIndex];
    const oldGrid = this.grid.map(row => [...row]);
    const oldScore = this.score;
    
    if (this.canPlaceBlockAtPosition(blockShape, row, col)) {
      this.placeBlock(blockShape, row, col, blockIndex);
      
      const reward = this.calculateReward(oldGrid, oldScore, this.getBlockSize(blockShape));
      const newState = this.getState();
      
      return {
        state: newState,
        reward: reward,
        done: this.gameOver
      };
    } else {
      return {
        state: this.getState(),
        reward: -100,
        done: true
      };
    }
  }

  /**
   * CLONE ENVIRONMENT for MCTS simulations
   */
  clone() {
    const cloned = new DQNEnvironment();
    
    // Copy state
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
    cloned.curriculumLevel = this.curriculumLevel;
    cloned.currentComplexity = this.currentComplexity;
    cloned.currentGridSize = this.currentGridSize;
    
    return cloned;
  }

  /**
   * CLONE STATE for MCTS nodes
   */
  cloneState() {
    return {
      grid: this.grid.map(row => [...row]),
      blocks: this.availableBlocks.map(block => block.map(row => [...row])),
      score: this.score,
      done: this.gameOver
    };
  }

  getBlockSize(blockShape) {
    // Simple implementation to count filled cells in block
    if (!blockShape || !Array.isArray(blockShape)) return 1;
    
    let size = 0;
    for (let row = 0; row < blockShape.length; row++) {
      for (let col = 0; col < blockShape[row].length; col++) {
        if (blockShape[row][col]) {
          size++;
        }
      }
    }
    return size || 1; // Ensure at least 1
  }
} 