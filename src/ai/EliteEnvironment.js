import * as tf from '@tensorflow/tfjs';
import { canPlaceBlock, checkGameOver, generateRandomBlocks, getBlockSize } from '../utils/gameLogic';

const GRID_SIZE = 9; // 9x9 grid for consistency
const MAX_BLOCKS = 3;
const MAX_BLOCK_SIZE = 3;
// ELITE STATE SIZE: 81 (grid) + 27 (line completion) + 27 (blocks) + 4 (meta) = 139 features
const ELITE_STATE_SIZE = 139;

/**
 * ELITE ENVIRONMENT - MAXIMUM PERFORMANCE WOOD BLOCK PUZZLE AI
 * 
 * Features:
 * - Enhanced 139-feature state representation with spatial intelligence
 * - Massive line-clearing rewards with exponential combos
 * - Sophisticated spatial reasoning and pattern recognition
 * - Advanced chain detection and opportunity evaluation
 * - Strategic penalties for dead ends and wasted space
 * - Multi-level curriculum system
 * - Comprehensive performance analytics
 */
export class EliteEnvironment {
  constructor() {
    this.reset();
    
    // ELITE REWARD SYSTEM - MAXIMUM PERFORMANCE FOCUS
    this.rewardConfig = {
      // MASSIVE LINE CLEARING INCENTIVES
      lineClearBase: 15000,        // Massive base reward for ANY line clear
      lineMultiplier: 8000,        // Exponential reward per line cleared
      comboMultiplier: 25000,      // ENORMOUS combo bonuses
      maxComboBonus: 40000,        // Maximum combo reward cap
      
      // STRATEGIC PLACEMENT REWARDS
      placementBase: 15,           // Points per cell placed
      efficiencyBonus: 50,         // Bonus for efficient placements
      connectivityBonus: 100,      // Bonus for connecting to existing blocks
      
      // ADVANCED SPATIAL INTELLIGENCE
      nearCompletionBonus: 1500,   // Massive bonus for near-complete lines
      almostCompleteBonus: 500,    // Bonus for lines that are 80%+ full
      patternFormationBonus: 200,  // Bonus for creating patterns
      futureOpportunityBonus: 150, // Bonus for creating future opportunities
      
      // SOPHISTICATED PENALTIES
      isolationPenalty: -200,      // Heavy penalty for isolated placements
      wastedSpacePenalty: -150,    // Penalty for creating wasted space
      deadEndPenalty: -300,        // Severe penalty for dead ends
      fragmentationPenalty: -100,  // Penalty for fragmentation
      
      // CHAIN REACTION BONUSES
      chainSetupBonus: 300,        // Bonus for setting up chain reactions
      chainExecutionBonus: 800,    // Massive bonus for executing chains
      sequentialLineBonus: 1200,   // Bonus for sequential line clears
      
      // SURVIVAL AND PROGRESSION
      survivalBonus: 2,            // Small bonus per step
      gameOverPenalty: -8000,      // Heavy penalty for game over
      longevityBonus: 5,           // Bonus for long survival
      
      // ELITE PERFORMANCE MULTIPLIERS
      masterMultiplier: 2.0,       // Multiplier for elite performance
      perfectPlayBonus: 5000       // Bonus for perfect play sequences
    };
    
    // ELITE PERFORMANCE TRACKING
    this.performanceMetrics = {
      totalLineClearsThisEpisode: 0,
      maxChainLength: 0,
      currentChainLength: 0,
      perfectMoves: 0,
      wastedMoves: 0,
      spatialEfficiency: 0,
      strategicValue: 0
    };
    
    // CURRICULUM SYSTEM
    this.curriculumLevel = 0;
    this.blockComplexityLevels = ['simple', 'medium', 'complex', 'master'];
    this.currentComplexity = this.blockComplexityLevels[this.curriculumLevel];
    
    console.log('🏆 ELITE ENVIRONMENT INITIALIZED - MAXIMUM PERFORMANCE MODE');
    console.log(`📊 Enhanced State Size: ${ELITE_STATE_SIZE} features`);
    console.log(`💎 Reward System: Advanced spatial intelligence + massive line rewards`);
  }

  reset() {
    this.grid = Array(GRID_SIZE).fill(null).map(() => Array(GRID_SIZE).fill(false));
    this.availableBlocks = this.generateEliteBlocks();
    this.score = 0;
    this.difficulty = 'elite';
    this.movesSinceClear = 0;
    this.gameOver = false;
    this.totalMoves = 0;
    this.lineClearsThisEpisode = 0;
    this.currentGridSize = GRID_SIZE;
    
    // Reset performance metrics
    this.performanceMetrics = {
      totalLineClearsThisEpisode: 0,
      maxChainLength: 0,
      currentChainLength: 0,
      perfectMoves: 0,
      wastedMoves: 0,
      spatialEfficiency: 0,
      strategicValue: 0
    };
    
    // Track line completion states for chain detection
    this.previousLineStates = this.analyzeLineCompletionStates();
    
    return this.getEliteState();
  }

  /**
   * ELITE BLOCK GENERATION with sophisticated variety
   */
  generateEliteBlocks() {
    const blocks = [];
    const allShapes = this.getEliteBlockShapes();
    
    // Ensure variety in block selection for strategic depth
    const selectedShapes = [];
    for (let i = 0; i < 3; i++) {
      let shape;
      let attempts = 0;
      do {
        shape = allShapes[Math.floor(Math.random() * allShapes.length)];
        attempts++;
      } while (selectedShapes.some(s => this.areShapesSimilar(s, shape)) && attempts < 10);
      
      selectedShapes.push(shape);
      blocks.push(shape);
    }
    
    return blocks;
  }

  /**
   * ELITE BLOCK SHAPES with strategic variety
   */
  getEliteBlockShapes() {
    const shapes = [
      // Single cells - highest utility
      [[true]],
      
      // Straight pieces - line completion focused
      [[true, true]],
      [[true], [true]],
      [[true, true, true]],
      [[true], [true], [true]],
      
      // L-shapes - corner utilization
      [[true, false], [true, false], [true, true]],
      [[true, true, true], [true, false, false]],
      [[true, true], [false, true], [false, true]],
      [[false, false, true], [true, true, true]],
      
      // T-shapes - versatile placement
      [[true, true, true], [false, true, false]],
      [[false, true], [true, true], [false, true]],
      [[false, true, false], [true, true, true]],
      [[true, false], [true, true], [true, false]],
      
      // Squares - stable foundations
      [[true, true], [true, true]],
      
      // Z-shapes - gap filling
      [[true, true, false], [false, true, true]],
      [[false, true], [true, true], [true, false]],
      
      // Plus shape - central positioning
      [[false, true, false], [true, true, true], [false, true, false]],
      
      // Corner shapes - edge optimization
      [[true, false, false], [true, false, false], [true, true, true]],
      [[true, true, true], [true, false, false], [true, false, false]],
      [[true, true, true], [false, false, true], [false, false, true]],
      [[false, false, true], [false, false, true], [true, true, true]]
    ];
    
    return shapes;
  }

  /**
   * ELITE STATE REPRESENTATION - 139 FEATURES
   * 
   * Features breakdown:
   * - 81 features: 9x9 grid state (0/1 for each cell)
   * - 27 features: Line completion analysis (9 rows + 9 cols + 9 diagonals/patterns)
   * - 27 features: Available blocks encoded (3 blocks × 9 features each)
   * - 4 features: Meta information (score, moves, difficulty, curriculum)
   */
  getEliteState() {
    const state = [];
    
    // 1. GRID STATE (81 features) - Basic grid representation
    for (let row = 0; row < GRID_SIZE; row++) {
      for (let col = 0; col < GRID_SIZE; col++) {
        state.push(this.grid[row][col] ? 1 : 0);
      }
    }
    
    // 2. LINE COMPLETION ANALYSIS (27 features) - Advanced spatial intelligence
    const lineAnalysis = this.analyzeLineCompletionStates();
    
    // Row completion percentages (9 features)
    for (let row = 0; row < GRID_SIZE; row++) {
      const filled = this.grid[row].filter(cell => cell).length;
      state.push(filled / GRID_SIZE);
    }
    
    // Column completion percentages (9 features)
    for (let col = 0; col < GRID_SIZE; col++) {
      const filled = this.grid.filter(row => row[col]).length;
      state.push(filled / GRID_SIZE);
    }
    
    // Advanced spatial patterns (9 features)
    state.push(lineAnalysis.nearCompleteLines / (GRID_SIZE * 2)); // Near complete lines ratio
    state.push(lineAnalysis.almostCompleteLines / (GRID_SIZE * 2)); // Almost complete lines ratio
    state.push(lineAnalysis.chainPotential); // Chain reaction potential
    state.push(lineAnalysis.spatialEfficiency); // How efficiently space is used
    state.push(lineAnalysis.connectivity); // How well blocks are connected
    state.push(lineAnalysis.fragmentationScore); // Fragmentation level
    state.push(lineAnalysis.cornerUtilization); // Corner usage efficiency
    state.push(lineAnalysis.edgeUtilization); // Edge usage efficiency
    state.push(lineAnalysis.deadSpaceRatio); // Ratio of dead/wasted space
    
    // 3. AVAILABLE BLOCKS ANALYSIS (27 features) - 3 blocks × 9 features each
    for (let i = 0; i < 3; i++) {
      if (i < this.availableBlocks.length) {
        const blockFeatures = this.analyzeBlockFeatures(this.availableBlocks[i]);
        state.push(blockFeatures.size / 9); // Normalized block size
        state.push(blockFeatures.width / 3); // Normalized width
        state.push(blockFeatures.height / 3); // Normalized height
        state.push(blockFeatures.density); // How dense the block is
        state.push(blockFeatures.linearPotential); // Potential for line completion
        state.push(blockFeatures.cornerFit); // How well it fits in corners
        state.push(blockFeatures.edgeFit); // How well it fits on edges
        state.push(blockFeatures.flexibilityScore); // How flexible placement is
        state.push(blockFeatures.strategicValue); // Overall strategic value
      } else {
        // No block available - all zeros
        for (let j = 0; j < 9; j++) state.push(0);
      }
    }
    
    // 4. META INFORMATION (4 features)
    state.push(Math.min(this.score / 10000, 1)); // Normalized score
    state.push(Math.min(this.totalMoves / 100, 1)); // Normalized moves
    state.push(this.curriculumLevel / 3); // Normalized curriculum level
    state.push(this.movesSinceClear / 20); // Normalized moves since last clear
    
    // Ensure exactly 139 features
    if (state.length !== ELITE_STATE_SIZE) {
      console.error(`⚠️ ELITE STATE SIZE MISMATCH: Expected ${ELITE_STATE_SIZE}, got ${state.length}`);
      // Pad or trim to ensure consistency
      while (state.length < ELITE_STATE_SIZE) state.push(0);
      if (state.length > ELITE_STATE_SIZE) state.splice(ELITE_STATE_SIZE);
    }
    
    return new Float32Array(state);
  }

  /**
   * COMPATIBILITY METHOD: getState() calls getEliteState()
   * This ensures compatibility with training systems that expect getState()
   */
  getState() {
    return this.getEliteState();
  }

  /**
   * ADVANCED LINE COMPLETION ANALYSIS
   */
  analyzeLineCompletionStates() {
    const analysis = {
      nearCompleteLines: 0,      // Lines with 7-8/9 cells filled
      almostCompleteLines: 0,    // Lines with 6/9 cells filled
      chainPotential: 0,         // Potential for chain reactions
      spatialEfficiency: 0,      // How efficiently space is used
      connectivity: 0,           // How well blocks are connected
      fragmentationScore: 0,     // Level of fragmentation
      cornerUtilization: 0,      // How well corners are used
      edgeUtilization: 0,        // How well edges are used
      deadSpaceRatio: 0          // Ratio of unusable space
    };
    
    // Analyze rows
    for (let row = 0; row < GRID_SIZE; row++) {
      const filled = this.grid[row].filter(cell => cell).length;
      if (filled >= 7) analysis.nearCompleteLines++;
      else if (filled >= 6) analysis.almostCompleteLines++;
    }
    
    // Analyze columns
    for (let col = 0; col < GRID_SIZE; col++) {
      const filled = this.grid.filter(row => row[col]).length;
      if (filled >= 7) analysis.nearCompleteLines++;
      else if (filled >= 6) analysis.almostCompleteLines++;
    }
    
    // Calculate spatial metrics
    analysis.chainPotential = this.calculateChainPotential();
    analysis.spatialEfficiency = this.calculateSpatialEfficiency();
    analysis.connectivity = this.calculateConnectivity();
    analysis.fragmentationScore = this.calculateFragmentation();
    analysis.cornerUtilization = this.calculateCornerUtilization();
    analysis.edgeUtilization = this.calculateEdgeUtilization();
    analysis.deadSpaceRatio = this.calculateDeadSpaceRatio();
    
    return analysis;
  }

  /**
   * ANALYZE BLOCK FEATURES for intelligent placement
   */
  analyzeBlockFeatures(block) {
    const features = {
      size: 0,
      width: block[0].length,
      height: block.length,
      density: 0,
      linearPotential: 0,
      cornerFit: 0,
      edgeFit: 0,
      flexibilityScore: 0,
      strategicValue: 0
    };
    
    // Calculate size and density directly
    let filledCells = 0;
    for (let row = 0; row < block.length; row++) {
      for (let col = 0; col < block[row].length; col++) {
        if (block[row][col]) filledCells++;
      }
    }
    
    features.size = filledCells;
    features.density = filledCells / (features.width * features.height);
    
    // Simplified calculations to avoid circular dependencies
    // Linear potential: straight lines are good for line completion
    if (block.length === 1 || block[0].length === 1) {
      features.linearPotential = 0.8;
    } else {
      features.linearPotential = features.density * 0.5;
    }
    
    // Corner fit: small blocks fit corners well
    if (filledCells <= 3) {
      features.cornerFit = 0.7;
    } else if (filledCells === 4 && (features.width === 3 || features.height === 3)) {
      features.cornerFit = 0.8; // L-shaped
    } else {
      features.cornerFit = 0.3;
    }
    
    // Edge fit: straight blocks fit edges well
    if (block.length === 1 || block[0].length === 1) {
      features.edgeFit = 0.8;
    } else {
      features.edgeFit = 0.4;
    }
    
    // Flexibility: smaller blocks are more flexible
    features.flexibilityScore = Math.max(0.1, 1.0 - (filledCells / 9));
    
    // Strategic value: combination of all factors
    features.strategicValue = (
      features.linearPotential * 0.3 + 
      features.cornerFit * 0.2 + 
      features.edgeFit * 0.2 + 
      features.flexibilityScore * 0.3
    );
    
    return features;
  }

  /**
   * SOPHISTICATED SPATIAL CALCULATIONS
   */
  calculateChainPotential() {
    let potential = 0;
    
    // Look for potential chain reactions
    for (let row = 0; row < GRID_SIZE; row++) {
      const filled = this.grid[row].filter(cell => cell).length;
      if (filled >= 6) {
        // Check if completing this line would enable others
        for (let col = 0; col < GRID_SIZE; col++) {
          if (!this.grid[row][col]) {
            const colFilled = this.grid.filter(r => r[col]).length;
            if (colFilled >= 6) potential += 0.5;
          }
        }
      }
    }
    
    return Math.min(potential, 1.0);
  }

  calculateSpatialEfficiency() {
    let totalCells = GRID_SIZE * GRID_SIZE;
    let filledCells = 0;
    let efficientCells = 0;
    
    for (let row = 0; row < GRID_SIZE; row++) {
      for (let col = 0; col < GRID_SIZE; col++) {
        if (this.grid[row][col]) {
          filledCells++;
          // Cell is efficient if it's part of a larger connected group
          const neighbors = this.getNeighbors(row, col).filter(([r, c]) => 
            r >= 0 && r < GRID_SIZE && c >= 0 && c < GRID_SIZE && this.grid[r][c]
          );
          if (neighbors.length >= 2) efficientCells++;
        }
      }
    }
    
    return filledCells > 0 ? efficientCells / filledCells : 0;
  }

  calculateConnectivity() {
    const regions = this.findConnectedRegions();
    if (regions.length === 0) return 0;
    
    // Prefer fewer, larger regions over many small ones
    const totalCells = regions.reduce((sum, region) => sum + region.size, 0);
    const avgRegionSize = totalCells / regions.length;
    
    return Math.min(avgRegionSize / 10, 1.0);
  }

  calculateFragmentation() {
    const regions = this.findConnectedRegions();
    if (regions.length <= 1) return 0;
    
    // More regions = more fragmentation
    return Math.min(regions.length / 10, 1.0);
  }

  calculateCornerUtilization() {
    const corners = [[0,0], [0,8], [8,0], [8,8]];
    const filledCorners = corners.filter(([r,c]) => this.grid[r][c]).length;
    return filledCorners / 4;
  }

  calculateEdgeUtilization() {
    let edgeCells = 0;
    let filledEdgeCells = 0;
    
    for (let i = 0; i < GRID_SIZE; i++) {
      // Top and bottom edges
      if (this.grid[0][i]) filledEdgeCells++; edgeCells++;
      if (this.grid[8][i]) filledEdgeCells++; edgeCells++;
      // Left and right edges (excluding corners already counted)
      if (i > 0 && i < 8) {
        if (this.grid[i][0]) filledEdgeCells++; edgeCells++;
        if (this.grid[i][8]) filledEdgeCells++; edgeCells++;
      }
    }
    
    return edgeCells > 0 ? filledEdgeCells / edgeCells : 0;
  }

  calculateDeadSpaceRatio() {
    let deadSpaces = 0;
    
    for (let row = 0; row < GRID_SIZE; row++) {
      for (let col = 0; col < GRID_SIZE; col++) {
        if (!this.grid[row][col] && this.isDeadSpace(row, col)) {
          deadSpaces++;
        }
      }
    }
    
    return deadSpaces / (GRID_SIZE * GRID_SIZE);
  }

  /**
   * HELPER METHODS
   */
  areShapesSimilar(shape1, shape2) {
    if (shape1.length !== shape2.length) return false;
    if (shape1[0].length !== shape2[0].length) return false;
    
    for (let i = 0; i < shape1.length; i++) {
      for (let j = 0; j < shape1[i].length; j++) {
        if (shape1[i][j] !== shape2[i][j]) return false;
      }
    }
    return true;
  }

  isLShaped(block) {
    // Simple L-shape detection - calculate features directly
    let filledCells = 0;
    for (let row = 0; row < block.length; row++) {
      for (let col = 0; col < block[row].length; col++) {
        if (block[row][col]) filledCells++;
      }
    }
    
    const width = block[0].length;
    const height = block.length;
    
    return filledCells === 4 && (width === 3 || height === 3);
  }

  isTShaped(block) {
    // Simple T-shape detection - calculate features directly
    let filledCells = 0;
    for (let row = 0; row < block.length; row++) {
      for (let col = 0; col < block[row].length; col++) {
        if (block[row][col]) filledCells++;
      }
    }
    
    const width = block[0].length;
    const height = block.length;
    
    return filledCells === 4 && width === 3 && height === 2;
  }

  getValidPlacementsForBlock(block) {
    const placements = [];
    for (let row = 0; row <= GRID_SIZE - block.length; row++) {
      for (let col = 0; col <= GRID_SIZE - block[0].length; col++) {
        if (this.canPlaceBlockAtPosition(block, row, col)) {
          placements.push([row, col]);
        }
      }
    }
    return placements;
  }

  isDeadSpace(row, col) {
    // A space is "dead" if it's isolated and cannot be efficiently filled
    const neighbors = this.getNeighbors(row, col);
    const filledNeighbors = neighbors.filter(([r, c]) => 
      r >= 0 && r < GRID_SIZE && c >= 0 && c < GRID_SIZE && this.grid[r][c]
    ).length;
    
    const emptyNeighbors = neighbors.filter(([r, c]) => 
      r >= 0 && r < GRID_SIZE && c >= 0 && c < GRID_SIZE && !this.grid[r][c]
    ).length;
    
    return filledNeighbors >= 3 && emptyNeighbors <= 1;
  }

  findConnectedRegions() {
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
      
      // Add neighbors to stack
      this.getNeighbors(row, col).forEach(([r, c]) => {
        if (r >= 0 && r < GRID_SIZE && c >= 0 && c < GRID_SIZE && !visited[r][c] && this.grid[r][c]) {
          stack.push([r, c]);
        }
      });
    }
    
    // Calculate compactness (ratio of filled area to bounding box area)
    if (region.cells.length > 0) {
      const minRow = Math.min(...region.cells.map(([r, c]) => r));
      const maxRow = Math.max(...region.cells.map(([r, c]) => r));
      const minCol = Math.min(...region.cells.map(([r, c]) => c));
      const maxCol = Math.max(...region.cells.map(([r, c]) => c));
      
      const boundingBoxArea = (maxRow - minRow + 1) * (maxCol - minCol + 1);
      region.compactness = region.size / boundingBoxArea;
    }
    
    return region;
  }

  getNeighbors(row, col) {
    return [
      [row - 1, col], [row + 1, col],
      [row, col - 1], [row, col + 1]
    ];
  }

  /**
   * ELITE REWARD CALCULATION - MAXIMUM PERFORMANCE FOCUS
   */
  calculateEliteReward(oldGrid, oldScore, blockSize, action) {
    const linesCleared = this.countLinesCleared(oldGrid, this.grid);
    let totalReward = 0;
    
    // Track performance metrics
    this.performanceMetrics.totalLineClearsThisEpisode += linesCleared;
    
    // 1. MASSIVE LINE CLEARING REWARDS
    if (linesCleared > 0) {
      const baseReward = this.rewardConfig.lineClearBase;
      const lineReward = linesCleared * this.rewardConfig.lineMultiplier;
      
      // EXPONENTIAL COMBO BONUSES
      let comboReward = 0;
      if (linesCleared === 2) comboReward = this.rewardConfig.comboMultiplier * 0.5;
      else if (linesCleared === 3) comboReward = this.rewardConfig.comboMultiplier * 1.0;
      else if (linesCleared >= 4) comboReward = this.rewardConfig.comboMultiplier * 2.0;
      
      // CHAIN BONUSES for sequential clears
      if (this.performanceMetrics.currentChainLength > 0) {
        comboReward += this.rewardConfig.chainExecutionBonus * this.performanceMetrics.currentChainLength;
        this.performanceMetrics.currentChainLength++;
      } else {
        this.performanceMetrics.currentChainLength = 1;
      }
      
      totalReward = baseReward + lineReward + comboReward;
      
      // Cap at maximum but allow for extraordinary performance
      totalReward = Math.min(totalReward, this.rewardConfig.maxComboBonus);
      
      console.log(`🏆 ELITE LINE CLEAR! Lines: ${linesCleared}, Chain: ${this.performanceMetrics.currentChainLength}, Total: ${totalReward}`);
    } else {
      // Reset chain if no lines cleared
      this.performanceMetrics.currentChainLength = 0;
    }
    
    // 2. STRATEGIC PLACEMENT REWARDS
    const placementReward = blockSize * this.rewardConfig.placementBase;
    totalReward += placementReward;
    
    // 3. ADVANCED SPATIAL INTELLIGENCE BONUSES
    const spatialAnalysis = this.analyzeCurrentSpatialState();
    
    // Near completion bonuses
    if (spatialAnalysis.nearCompleteLines > 0) {
      totalReward += spatialAnalysis.nearCompleteLines * this.rewardConfig.nearCompletionBonus;
    }
    
    if (spatialAnalysis.almostCompleteLines > 0) {
      totalReward += spatialAnalysis.almostCompleteLines * this.rewardConfig.almostCompleteBonus;
    }
    
    // Pattern formation and connectivity bonuses
    if (spatialAnalysis.improvesConnectivity) {
      totalReward += this.rewardConfig.connectivityBonus;
    }
    
    if (spatialAnalysis.formsPattern) {
      totalReward += this.rewardConfig.patternFormationBonus;
    }
    
    if (spatialAnalysis.createsFutureOpportunities) {
      totalReward += this.rewardConfig.futureOpportunityBonus;
    }
    
    // 4. CHAIN SETUP DETECTION
    if (spatialAnalysis.setsUpChain) {
      totalReward += this.rewardConfig.chainSetupBonus;
    }
    
    // 5. EFFICIENCY AND PLACEMENT QUALITY
    if (spatialAnalysis.isEfficient) {
      totalReward += this.rewardConfig.efficiencyBonus;
      this.performanceMetrics.perfectMoves++;
    } else if (spatialAnalysis.isWasteful) {
      totalReward += this.rewardConfig.wastedSpacePenalty;
      this.performanceMetrics.wastedMoves++;
    }
    
    // 6. SOPHISTICATED PENALTIES
    if (spatialAnalysis.createsIsolation) {
      totalReward += this.rewardConfig.isolationPenalty;
    }
    
    if (spatialAnalysis.createsDeadEnd) {
      totalReward += this.rewardConfig.deadEndPenalty;
    }
    
    if (spatialAnalysis.increasesFragmentation) {
      totalReward += this.rewardConfig.fragmentationPenalty;
    }
    
    // 7. SURVIVAL AND PROGRESSION
    if (!this.gameOver) {
      totalReward += this.rewardConfig.survivalBonus;
      
      // Longevity bonus for long-term survival
      if (this.totalMoves > 50) {
        totalReward += this.rewardConfig.longevityBonus;
      }
    } else {
      totalReward += this.rewardConfig.gameOverPenalty;
    }
    
    // 8. ELITE PERFORMANCE MULTIPLIERS
    if (this.performanceMetrics.perfectMoves > this.performanceMetrics.wastedMoves * 2) {
      totalReward *= this.rewardConfig.masterMultiplier;
      
      if (this.performanceMetrics.perfectMoves >= 10) {
        totalReward += this.rewardConfig.perfectPlayBonus;
      }
    }
    
    // Log significant rewards for analysis
    if (Math.abs(totalReward) > 100) {
      console.log(`💎 ELITE REWARD BREAKDOWN:`);
      console.log(`  📦 Base Placement: +${placementReward}`);
      console.log(`  🎯 Line Clear Bonus: +${linesCleared > 0 ? (this.rewardConfig.lineClearBase + linesCleared * this.rewardConfig.lineMultiplier) : 0}`);
      console.log(`  🔥 Spatial Intelligence: ${(spatialAnalysis.nearCompleteLines * this.rewardConfig.nearCompletionBonus).toFixed(1)}`);
      console.log(`  ⚡ Chain Potential: ${spatialAnalysis.setsUpChain ? this.rewardConfig.chainSetupBonus : 0}`);
      console.log(`  🏆 TOTAL: ${totalReward.toFixed(1)}`);
    }
    
    return Math.max(-15000, Math.min(75000, totalReward));
  }

  /**
   * ANALYZE CURRENT SPATIAL STATE for advanced intelligence
   */
  analyzeCurrentSpatialState() {
    const analysis = {
      nearCompleteLines: 0,
      almostCompleteLines: 0,
      improvesConnectivity: false,
      formsPattern: false,
      createsFutureOpportunities: false,
      setsUpChain: false,
      isEfficient: false,
      isWasteful: false,
      createsIsolation: false,
      createsDeadEnd: false,
      increasesFragmentation: false
    };
    
    // Analyze line completion states
    const lineStates = this.analyzeLineCompletionStates();
    analysis.nearCompleteLines = lineStates.nearCompleteLines;
    analysis.almostCompleteLines = lineStates.almostCompleteLines;
    
    // Compare with previous state to detect improvements
    if (this.previousLineStates) {
      analysis.improvesConnectivity = lineStates.connectivity > this.previousLineStates.connectivity;
      analysis.createsFutureOpportunities = lineStates.chainPotential > this.previousLineStates.chainPotential;
      analysis.setsUpChain = lineStates.chainPotential > 0.7;
    }
    
    // Detect patterns and efficiency
    analysis.isEfficient = lineStates.spatialEfficiency > 0.7;
    analysis.isWasteful = lineStates.deadSpaceRatio > 0.3;
    analysis.createsIsolation = lineStates.fragmentationScore > 0.5;
    analysis.formsPattern = lineStates.connectivity > 0.6;
    
    // Update previous state for next comparison
    this.previousLineStates = lineStates;
    
    return analysis;
  }

  /**
   * GAME STATE MANAGEMENT
   */
  getStateSize() {
    return ELITE_STATE_SIZE;
  }

  getMaxActionSpace() {
    return GRID_SIZE * GRID_SIZE * MAX_BLOCKS; // 243 possible actions
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
      const oldRowFull = oldGrid[row].every(cell => cell);
      const newRowEmpty = newGrid[row].every(cell => !cell);
      if (!oldRowFull && newRowEmpty) linesCleared++;
    }
    
    // Check columns
    for (let col = 0; col < GRID_SIZE; col++) {
      const oldColFull = oldGrid.every(row => row[col]);
      const newColEmpty = newGrid.every(row => !row[col]);
      if (!oldColFull && newColEmpty) linesCleared++;
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
        state: this.getEliteState(),
        reward: -1000,
        done: true,
        info: { error: 'Invalid block index' }
      };
    }
    
    const block = this.availableBlocks[blockIndex];
    
    if (!this.canPlaceBlockAtPosition(block, row, col)) {
      return {
        state: this.getEliteState(),
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
      this.score += linesCleared * 100 * (1 + linesCleared);
    }
    
    // Remove used block
    this.availableBlocks.splice(blockIndex, 1);
    
    // Check if need new blocks
    if (this.availableBlocks.length === 0) {
      this.availableBlocks = this.generateEliteBlocks();
    }
    
    // Check game over
    this.gameOver = this.checkGameOver();
    
    // Calculate elite reward
    const reward = this.calculateEliteReward(oldGrid, oldScore, blockSize, actionId);
    
    return {
      state: this.getEliteState(),
      reward: reward,
      done: this.gameOver,
      info: {
        linesCleared: linesCleared,
        score: this.score,
        moves: this.totalMoves,
        chainLength: this.performanceMetrics.currentChainLength
      }
    };
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

  updateCurriculum(episodeScore, linesCleared) {
    // Elite curriculum advancement based on mastery
    if (linesCleared >= 3 && episodeScore > 1000 && this.curriculumLevel < 3) {
      this.curriculumLevel++;
      this.currentComplexity = this.blockComplexityLevels[this.curriculumLevel];
      console.log(`🏆 CURRICULUM ADVANCED to Level ${this.curriculumLevel}: ${this.currentComplexity}`);
    }
  }

  setState(grid, availableBlocks, score, difficulty = 'elite') {
    this.grid = grid.map(row => [...row]);
    this.availableBlocks = availableBlocks || this.generateEliteBlocks();
    this.score = score;
    this.difficulty = difficulty;
    this.gameOver = this.checkGameOver();
    return this.getEliteState();
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

  /**
   * CLONE ENVIRONMENT for MCTS simulations
   */
  clone() {
    const cloned = new EliteEnvironment();
    
    // Copy grid state
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
      blocks: this.availableBlocks.map(block => 
        block.map(row => [...row])
      ),
      score: this.score,
      done: this.gameOver
    };
  }
} 