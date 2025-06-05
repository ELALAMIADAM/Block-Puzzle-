import React, { useState, useEffect, useCallback } from 'react';
import { DndProvider } from 'react-dnd';
import { HTML5Backend } from 'react-dnd-html5-backend';
import GameBoard from './components/GameBoard';
import BlockTray from './components/BlockTray';
import ScoreDisplay from './components/ScoreDisplay';
import GameOverModal from './components/GameOverModal';
import { generateRandomBlocks, checkGameOver } from './utils/gameLogic';

const GRID_SIZE = 9;

function App() {
  const [grid, setGrid] = useState(() => 
    Array(GRID_SIZE).fill(null).map(() => Array(GRID_SIZE).fill(false))
  );
  const [availableBlocks, setAvailableBlocks] = useState([]);
  const [score, setScore] = useState(0);
  const [bestScore, setBestScore] = useState(() => {
    const saved = localStorage.getItem('woodBlockPuzzleBestScore');
    return saved ? parseInt(saved, 10) : 0;
  });
  const [gameOver, setGameOver] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [linesCleared, setLinesCleared] = useState(0);
  const [difficulty, setDifficulty] = useState('normal'); // 'normal' or 'hard'
  const [clearingMessage, setClearingMessage] = useState('');
  const [isAIMode, setIsAIMode] = useState(false);
  const [aiSpeed, setAiSpeed] = useState(1000); // AI delay in milliseconds

  // Initialize game
  useEffect(() => {
    resetGame();
  }, []);

  // Check for game over when blocks change
  useEffect(() => {
    if (availableBlocks.length > 0) {
      const isGameOver = checkGameOver(grid, availableBlocks);
      if (isGameOver) {
        setGameOver(true);
        if (score > bestScore) {
          setBestScore(score);
          localStorage.setItem('woodBlockPuzzleBestScore', score.toString());
        }
      }
    }
  }, [grid, availableBlocks, score, bestScore]);

  const resetGame = useCallback(() => {
    const newGrid = Array(GRID_SIZE).fill(null).map(() => Array(GRID_SIZE).fill(false));
    setGrid(newGrid);
    setAvailableBlocks(generateRandomBlocks());
    setScore(0);
    setGameOver(false);
    setIsPaused(false);
    setLinesCleared(0);
    setIsAIMode(false); // Reset AI mode on new game
  }, []);

  const clearCompletedLines = useCallback((newGrid) => {
    let totalCleared = 0;
    let rowsCleared = 0;
    let colsCleared = 0;
    let squaresCleared = 0;
    const gridCopy = newGrid.map(row => [...row]);
    const clearedMessages = [];
    
    // Check rows
    for (let row = 0; row < GRID_SIZE; row++) {
      if (gridCopy[row].every(cell => cell)) {
        rowsCleared++;
        totalCleared++;
        // Clear the row (including blocked center in hard mode)
        for (let col = 0; col < GRID_SIZE; col++) {
          gridCopy[row][col] = false;
        }
      }
    }
    
    // Check columns
    for (let col = 0; col < GRID_SIZE; col++) {
      if (gridCopy.every(row => row[col])) {
        colsCleared++;
        totalCleared++;
        // Clear the column (including blocked center in hard mode)
        for (let row = 0; row < GRID_SIZE; row++) {
          gridCopy[row][col] = false;
        }
      }
    }
    
    // Check 3x3 squares (9x9 grid has 9 squares)
    for (let squareRow = 0; squareRow < 3; squareRow++) {
      for (let squareCol = 0; squareCol < 3; squareCol++) {
        let isSquareComplete = true;
        
        // Skip center square check in hard mode (it's always blocked)
        if (difficulty === 'hard' && squareRow === 1 && squareCol === 1) {
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
    
    if (totalCleared > 0) {
      // Create clearing message
      if (rowsCleared > 0) clearedMessages.push(`${rowsCleared} row${rowsCleared > 1 ? 's' : ''}`);
      if (colsCleared > 0) clearedMessages.push(`${colsCleared} column${colsCleared > 1 ? 's' : ''}`);
      if (squaresCleared > 0) clearedMessages.push(`${squaresCleared} square${squaresCleared > 1 ? 's' : ''}`);
      
      const message = `Cleared: ${clearedMessages.join(', ')}!`;
      setClearingMessage(message);
      
      // Clear message after 2 seconds
      setTimeout(() => setClearingMessage(''), 2000);
      
      setLinesCleared(prev => prev + totalCleared);
      
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
      if (difficulty === 'hard') {
        points = Math.floor(points * 1.5); // 50% more points in hard mode
      }
      
      setScore(prev => prev + points);
    }
    
    return gridCopy;
  }, [difficulty]);

  const placeBlock = useCallback((blockShape, startRow, startCol, blockIndex) => {
    if (isPaused || gameOver) return false;

    // Check if placement is valid
    for (let row = 0; row < blockShape.length; row++) {
      for (let col = 0; col < blockShape[row].length; col++) {
        if (blockShape[row][col]) {
          const gridRow = startRow + row;
          const gridCol = startCol + col;
          
          if (gridRow >= GRID_SIZE || gridCol >= GRID_SIZE || grid[gridRow][gridCol]) {
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

    // Place the block
    const newGrid = grid.map(row => [...row]);
    for (let row = 0; row < blockShape.length; row++) {
      for (let col = 0; col < blockShape[row].length; col++) {
        if (blockShape[row][col]) {
          newGrid[startRow + row][startCol + col] = true;
        }
      }
    }

    // Clear completed lines
    const clearedGrid = clearCompletedLines(newGrid);
    setGrid(clearedGrid);

    // Add score for placing the block
    const blockSize = blockShape.flat().filter(cell => cell).length;
    setScore(prev => prev + blockSize * 10);

    // Remove the used block and generate new ones if needed
    const newBlocks = availableBlocks.filter((_, index) => index !== blockIndex);
    if (newBlocks.length === 0) {
      setAvailableBlocks(generateRandomBlocks());
    } else {
      setAvailableBlocks(newBlocks);
    }

    return true;
  }, [grid, availableBlocks, isPaused, gameOver, clearCompletedLines, difficulty]);

  const togglePause = () => {
    setIsPaused(!isPaused);
  };

  const toggleAIMode = () => {
    setIsAIMode(!isAIMode);
    if (!isAIMode) {
      setIsPaused(false); // Unpause when starting AI
    }
  };

  // AI logic to find valid positions for a block
  const findValidPositions = useCallback((blockShape, currentGrid) => {
    const validPositions = [];
    
    for (let row = 0; row < GRID_SIZE; row++) {
      for (let col = 0; col < GRID_SIZE; col++) {
        let canPlace = true;
        
        // Check if block can be placed at this position
        for (let blockRow = 0; blockRow < blockShape.length; blockRow++) {
          for (let blockCol = 0; blockCol < blockShape[blockRow].length; blockCol++) {
            if (blockShape[blockRow][blockCol]) {
              const gridRow = row + blockRow;
              const gridCol = col + blockCol;
              
              // Check bounds
              if (gridRow >= GRID_SIZE || gridCol >= GRID_SIZE || 
                  gridRow < 0 || gridCol < 0 || currentGrid[gridRow][gridCol]) {
                canPlace = false;
                break;
              }
              
              // Check blocked center square in hard mode
              if (difficulty === 'hard' && 
                  gridRow >= 3 && gridRow <= 5 && 
                  gridCol >= 3 && gridCol <= 5) {
                canPlace = false;
                break;
              }
            }
          }
          if (!canPlace) break;
        }
        
        if (canPlace) {
          validPositions.push({ row, col });
        }
      }
    }
    
    return validPositions;
  }, [difficulty]);

  // AI automatic play logic
  useEffect(() => {
    if (!isAIMode || isPaused || gameOver || availableBlocks.length === 0) return;

    const aiInterval = setInterval(() => {
      // Try to place a random block
      const blockIndex = Math.floor(Math.random() * availableBlocks.length);
      const blockShape = availableBlocks[blockIndex];
      const validPositions = findValidPositions(blockShape, grid);
      
      if (validPositions.length > 0) {
        // Pick a random valid position
        const randomPosition = validPositions[Math.floor(Math.random() * validPositions.length)];
        placeBlock(blockShape, randomPosition.row, randomPosition.col, blockIndex);
      } else {
        // Try other blocks if current one can't be placed
        let placed = false;
        for (let i = 0; i < availableBlocks.length && !placed; i++) {
          if (i !== blockIndex) {
            const otherBlockShape = availableBlocks[i];
            const otherValidPositions = findValidPositions(otherBlockShape, grid);
            
            if (otherValidPositions.length > 0) {
              const randomPosition = otherValidPositions[Math.floor(Math.random() * otherValidPositions.length)];
              placeBlock(otherBlockShape, randomPosition.row, randomPosition.col, i);
              placed = true;
            }
          }
        }
      }
    }, aiSpeed);

    return () => clearInterval(aiInterval);
  }, [isAIMode, isPaused, gameOver, availableBlocks, grid, findValidPositions, placeBlock, aiSpeed]);

  return (
    <DndProvider backend={HTML5Backend}>
      <div className="game-container">
        <h1 style={{ color: 'white', fontSize: '36px', margin: '0 0 20px 0', textAlign: 'center' }}>
          Wood Block Puzzle
        </h1>
        
        <div className="difficulty-selector">
          <label style={{ color: 'white', marginRight: '15px' }}>Difficulty:</label>
          <select 
            value={difficulty} 
            onChange={(e) => setDifficulty(e.target.value)}
            className="difficulty-select"
            disabled={!gameOver && score > 0}
          >
            <option value="normal">Normal</option>
            <option value="hard">Hard (Blocked Center)</option>
          </select>
        </div>
        
        {clearingMessage && (
          <div className="clearing-message">
            {clearingMessage}
          </div>
        )}
        
        <ScoreDisplay 
          score={score} 
          bestScore={bestScore} 
          linesCleared={linesCleared}
          difficulty={difficulty}
          isAIMode={isAIMode}
        />
        
        <div className="game-main">
          <div className="game-board-container">
            <GameBoard 
              grid={grid} 
              onBlockPlace={placeBlock}
              availableBlocks={availableBlocks}
              isPaused={isPaused || isAIMode}
              difficulty={difficulty}
            />
          </div>
          
          <BlockTray 
            blocks={availableBlocks} 
            onBlockPlace={placeBlock}
            disabled={isPaused || gameOver || isAIMode}
          />
        </div>
        
        <div className="ai-controls">
          <button 
            className={`btn ai-btn ${isAIMode ? 'ai-active' : ''}`}
            onClick={toggleAIMode}
            disabled={gameOver}
          >
            {isAIMode ? '🤖 Stop AI' : '🤖 Start AI'}
          </button>
          
          {isAIMode && (
            <div className="ai-speed-control">
              <label style={{ color: 'white', marginRight: '10px' }}>AI Speed:</label>
              <select 
                value={aiSpeed} 
                onChange={(e) => setAiSpeed(Number(e.target.value))}
                className="speed-select"
              >
                <option value={3000}>Slow (3s)</option>
                <option value={1500}>Medium (1.5s)</option>
                <option value={1000}>Normal (1s)</option>
                <option value={500}>Fast (0.5s)</option>
                <option value={200}>Very Fast (0.2s)</option>
              </select>
            </div>
          )}
        </div>

        <div className="controls">
          <button 
            className="btn" 
            onClick={togglePause}
            disabled={gameOver || isAIMode}
          >
            {isPaused ? '▶️ Resume' : '⏸️ Pause'}
          </button>
          <button className="btn" onClick={resetGame}>
            🔄 New Game
          </button>
        </div>

        {gameOver && (
          <GameOverModal 
            score={score}
            bestScore={bestScore}
            isNewBest={score === bestScore}
            onRestart={resetGame}
          />
        )}
      </div>
    </DndProvider>
  );
}

export default App; 