import React, { useState, useEffect, useCallback } from 'react';
import { DndProvider } from 'react-dnd';
import { HTML5Backend } from 'react-dnd-html5-backend';
import GameBoard from './components/GameBoard';
import BlockTray from './components/BlockTray';
import ScoreDisplay from './components/ScoreDisplay';
import GameOverModal from './components/GameOverModal';
import AITrainingPanel from './components/AITrainingPanel';
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
  const [showAIPanel, setShowAIPanel] = useState(false);

  // Initialize game
  useEffect(() => {
    resetGame();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Check for game over when blocks change
  useEffect(() => {
    if (availableBlocks.length > 0) {
      const isGameOver = checkGameOver(grid, availableBlocks, difficulty);
      if (isGameOver) {
        setGameOver(true);
        if (score > bestScore) {
          setBestScore(score);
          localStorage.setItem('woodBlockPuzzleBestScore', score.toString());
        }
      }
    }
  }, [grid, availableBlocks, score, bestScore, difficulty]);

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
    let totalCleared = 0;
    let rowsCleared = 0;
    let colsCleared = 0;
    let squaresCleared = 0;
    const gridCopy = newGrid.map(row => [...row]);
    const clearedMessages = [];
    
    // Check rows
    for (let row = 0; row < GRID_SIZE; row++) {
      let isRowComplete = true;
      
      // Check if all non-blocked cells in the row are filled
      for (let col = 0; col < GRID_SIZE; col++) {
        const isBlocked = difficulty === 'hard' && 
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
          const isBlocked = difficulty === 'hard' && 
                           row >= 3 && row <= 5 && 
                           col >= 3 && col <= 5;
          
          if (!isBlocked) {
            gridCopy[row][col] = false;
          }
        }
      }
    }
    
    // Check columns
    for (let col = 0; col < GRID_SIZE; col++) {
      let isColComplete = true;
      
      // Check if all non-blocked cells in the column are filled
      for (let row = 0; row < GRID_SIZE; row++) {
        const isBlocked = difficulty === 'hard' && 
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
          const isBlocked = difficulty === 'hard' && 
                           row >= 3 && row <= 5 && 
                           col >= 3 && col <= 5;
          
          if (!isBlocked) {
            gridCopy[row][col] = false;
          }
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

  const handleAIGameStateUpdate = useCallback((aiMove) => {
    // Execute AI move in the game
    const { blockIndex, row, col, blockShape } = aiMove;
    const success = placeBlock(blockShape, row, col, blockIndex);
    console.log(`🎮 AI move result: ${success ? 'Success' : 'Failed'}`);
    return success;
  }, [placeBlock]);

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
        />
        
        <div className="game-main">
          <div className="game-board-container">
            <GameBoard 
              grid={grid} 
              onBlockPlace={placeBlock}
              availableBlocks={availableBlocks}
              isPaused={isPaused}
              difficulty={difficulty}
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
          <button 
            className="btn ai-panel-toggle"
            onClick={() => {
              setShowAIPanel(!showAIPanel);
              if (!showAIPanel) {
                // Scroll to AI panel after it appears
                setTimeout(() => {
                  const aiPanel = document.querySelector('.ai-training-panel');
                  if (aiPanel) {
                    aiPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });
                  }
                }, 100);
              }
            }}
            style={{
              background: showAIPanel ? 'linear-gradient(145deg, #DC143C, #8B0000)' : 'linear-gradient(145deg, #228B22, #006400)',
              fontSize: '16px',
              fontWeight: 'bold'
            }}
          >
            {showAIPanel ? '🤖 Hide AI Panel' : '🤖 Show AI Panel'}
          </button>
        </div>

        {showAIPanel && (
          <div style={{ 
            width: '100%', 
            display: 'flex', 
            justifyContent: 'center',
            marginTop: '20px',
            animation: 'fadeIn 0.5s ease-in-out'
          }}>
            <AITrainingPanel
              grid={grid}
              availableBlocks={availableBlocks}
              score={score}
              difficulty={difficulty}
              onGameStateUpdate={handleAIGameStateUpdate}
              gameOver={gameOver}
              onResetGame={resetGame}
            />
          </div>
        )}

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