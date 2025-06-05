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
  }, []);

  const clearCompletedLines = useCallback((newGrid) => {
    let clearedCount = 0;
    const gridCopy = newGrid.map(row => [...row]);
    
    // Check rows
    for (let row = 0; row < GRID_SIZE; row++) {
      if (gridCopy[row].every(cell => cell)) {
        clearedCount++;
        // Clear the row
        for (let col = 0; col < GRID_SIZE; col++) {
          gridCopy[row][col] = false;
        }
      }
    }
    
    // Check columns
    for (let col = 0; col < GRID_SIZE; col++) {
      if (gridCopy.every(row => row[col])) {
        clearedCount++;
        // Clear the column
        for (let row = 0; row < GRID_SIZE; row++) {
          gridCopy[row][col] = false;
        }
      }
    }
    
    // Check 3x3 squares (9x9 grid has 9 squares)
    for (let squareRow = 0; squareRow < 3; squareRow++) {
      for (let squareCol = 0; squareCol < 3; squareCol++) {
        let isSquareComplete = true;
        
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
          clearedCount++;
          for (let row = squareRow * 3; row < (squareRow + 1) * 3; row++) {
            for (let col = squareCol * 3; col < (squareCol + 1) * 3; col++) {
              gridCopy[row][col] = false;
            }
          }
        }
      }
    }
    
    if (clearedCount > 0) {
      setLinesCleared(prev => prev + clearedCount);
      setScore(prev => prev + clearedCount * 100);
    }
    
    return gridCopy;
  }, []);

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
  }, [grid, availableBlocks, isPaused, gameOver, clearCompletedLines]);

  const togglePause = () => {
    setIsPaused(!isPaused);
  };

  return (
    <DndProvider backend={HTML5Backend}>
      <div className="game-container">
        <h1 style={{ color: 'white', fontSize: '36px', margin: '0 0 30px 0', textAlign: 'center' }}>
          Wood Block Puzzle
        </h1>
        
        <ScoreDisplay 
          score={score} 
          bestScore={bestScore} 
          linesCleared={linesCleared}
        />
        
        <div className="game-main">
          <div className="game-board-container">
            <GameBoard 
              grid={grid} 
              onBlockPlace={placeBlock}
              availableBlocks={availableBlocks}
              isPaused={isPaused}
            />
          </div>
          
          <BlockTray 
            blocks={availableBlocks} 
            onBlockPlace={placeBlock}
            disabled={isPaused || gameOver}
          />
        </div>
        
        <div className="controls">
          <button 
            className="btn" 
            onClick={togglePause}
            disabled={gameOver}
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