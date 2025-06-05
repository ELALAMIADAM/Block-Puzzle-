// Block shapes for the Wood Block Puzzle
const BLOCK_SHAPES = [
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
  [[true, true, false], [false, true, true]],
  [[false, true], [true, true], [true, false]],
  
  // Plus shape
  [[false, true, false], [true, true, true], [false, true, false]],
  
  // 4x1 blocks
  [[true, true, true, true]],
  [[true], [true], [true], [true]],
  
  // 5x1 blocks
  [[true, true, true, true, true]],
  [[true], [true], [true], [true], [true]],
  
  // Corner shapes
  [[true, false, false], [true, false, false], [true, true, true]],
  [[true, true, true], [true, false, false], [true, false, false]],
  [[true, true, true], [false, false, true], [false, false, true]],
  [[false, false, true], [false, false, true], [true, true, true]],
  
  // Step shapes
  [[true, false, false], [true, true, false], [false, true, true]],
  [[false, false, true], [false, true, true], [true, true, false]],
];

export function generateRandomBlocks() {
  const blocks = [];
  for (let i = 0; i < 3; i++) {
    const randomIndex = Math.floor(Math.random() * BLOCK_SHAPES.length);
    blocks.push(BLOCK_SHAPES[randomIndex]);
  }
  return blocks;
}

export function canPlaceBlock(grid, blockShape, startRow, startCol, difficulty = 'normal') {
  const gridSize = grid.length;
  
  for (let row = 0; row < blockShape.length; row++) {
    for (let col = 0; col < blockShape[row].length; col++) {
      if (blockShape[row][col]) {
        const gridRow = startRow + row;
        const gridCol = startCol + col;
        
        // Check bounds
        if (gridRow < 0 || gridRow >= gridSize || gridCol < 0 || gridCol >= gridSize) {
          return false;
        }
        
        // Check if cell is already occupied
        if (grid[gridRow][gridCol]) {
          return false;
        }
        
        // Check if trying to place in blocked center square (hard mode)
        if (difficulty === 'hard' && 
            gridRow >= 3 && gridRow <= 5 && 
            gridCol >= 3 && gridCol <= 5) {
          return false;
        }
      }
    }
  }
  
  return true;
}

export function checkGameOver(grid, availableBlocks, difficulty = 'normal') {
  const gridSize = grid.length;
  
  // Check if any block can be placed anywhere on the grid
  for (const block of availableBlocks) {
    for (let row = 0; row < gridSize; row++) {
      for (let col = 0; col < gridSize; col++) {
        if (canPlaceBlock(grid, block, row, col, difficulty)) {
          return false; // Game can continue
        }
      }
    }
  }
  
  return true; // Game over
}

export function getBlockSize(blockShape) {
  return blockShape.flat().filter(cell => cell).length;
}

export function calculateScore(blockShape, linesCleared) {
  const blockSize = getBlockSize(blockShape);
  const blockPoints = blockSize * 10;
  const linePoints = linesCleared * 100;
  return blockPoints + linePoints;
} 