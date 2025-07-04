import React, { useState } from 'react';
import { useDrop } from 'react-dnd';

const GRID_SIZE = 9;

function GameBoard({ grid, onBlockPlace, availableBlocks, isPaused, difficulty }) {
  const [dragPreview, setDragPreview] = useState(null);
  const [previewPosition, setPreviewPosition] = useState({ row: -1, col: -1 });

  const [{ isOver }, drop] = useDrop({
    accept: 'block',
    drop: (item, monitor) => {
      const clientOffset = monitor.getClientOffset();
      const dropPosition = getDropPosition(clientOffset);
      
      if (dropPosition) {
        // Calculate the actual placement position based on the dragged cell
        const actualRow = dropPosition.row - (item.dragCellRow || 0);
        const actualCol = dropPosition.col - (item.dragCellCol || 0);
        
        const success = onBlockPlace(
          item.shape, 
          actualRow, 
          actualCol, 
          item.index
        );
        if (success) {
          setDragPreview(null);
          setPreviewPosition({ row: -1, col: -1 });
        }
      }
    },
    hover: (item, monitor) => {
      const clientOffset = monitor.getClientOffset();
      const dropPosition = getDropPosition(clientOffset);
      
      if (dropPosition) {
        // Calculate the actual placement position based on the dragged cell
        const actualRow = dropPosition.row - (item.dragCellRow || 0);
        const actualCol = dropPosition.col - (item.dragCellCol || 0);
        
        if (canPlaceBlock(item.shape, actualRow, actualCol)) {
          setDragPreview(item.shape);
          setPreviewPosition({ row: actualRow, col: actualCol });
        } else {
          setDragPreview(null);
          setPreviewPosition({ row: -1, col: -1 });
        }
      } else {
        setDragPreview(null);
        setPreviewPosition({ row: -1, col: -1 });
      }
    },
    collect: (monitor) => ({
      isOver: !!monitor.isOver(),
    }),
  });

  const getDropPosition = (clientOffset) => {
    if (!clientOffset) return null;
    
    const boardElement = document.querySelector('.game-board');
    if (!boardElement) return null;
    
    const boardRect = boardElement.getBoundingClientRect();
    const cellSize = (boardRect.width - 20 - (GRID_SIZE - 1) * 2) / GRID_SIZE; // Account for padding and gaps
    
    const relativeX = clientOffset.x - boardRect.left - 10; // Account for padding
    const relativeY = clientOffset.y - boardRect.top - 10;
    
    const col = Math.floor(relativeX / (cellSize + 2)); // Account for gap
    const row = Math.floor(relativeY / (cellSize + 2));
    
    if (row >= 0 && row < GRID_SIZE && col >= 0 && col < GRID_SIZE) {
      return { row, col };
    }
    
    return null;
  };

  const canPlaceBlock = (blockShape, startRow, startCol) => {
    if (isPaused) return false;
    
    for (let row = 0; row < blockShape.length; row++) {
      for (let col = 0; col < blockShape[row].length; col++) {
        if (blockShape[row][col]) {
          const gridRow = startRow + row;
          const gridCol = startCol + col;
          
          if (gridRow >= GRID_SIZE || gridCol >= GRID_SIZE || 
              gridRow < 0 || gridCol < 0 || grid[gridRow][gridCol]) {
            return false;
          }
          
          // Check blocked center square in hard mode
          if (difficulty === 'hard' && 
              gridRow >= 3 && gridRow <= 5 && 
              gridCol >= 3 && gridCol <= 5) {
            return false;
          }
        }
      }
    }
    return true;
  };

  const shouldHighlightCell = (row, col) => {
    if (!dragPreview || previewPosition.row === -1) return false;
    
    const relativeRow = row - previewPosition.row;
    const relativeCol = col - previewPosition.col;
    
    if (relativeRow >= 0 && relativeRow < dragPreview.length &&
        relativeCol >= 0 && relativeCol < dragPreview[0].length) {
      return dragPreview[relativeRow][relativeCol];
    }
    
    return false;
  };

  const getSquareBorderClasses = (rowIndex, colIndex) => {
    let classes = '';
    // Add right border for 3x3 square separation (columns 2 and 5)
    if (colIndex === 2 || colIndex === 5) {
      classes += ' square-border-right';
    }
    // Add bottom border for 3x3 square separation (rows 2 and 5)
    if (rowIndex === 2 || rowIndex === 5) {
      classes += ' square-border-bottom';
    }
    return classes;
  };

  const isCenterBlocked = (rowIndex, colIndex) => {
    return difficulty === 'hard' && 
           rowIndex >= 3 && rowIndex <= 5 && 
           colIndex >= 3 && colIndex <= 5;
  };

  return (
    <div 
      ref={drop}
      className={`game-board ${isOver ? 'drag-over' : ''}`}
    >
      {grid.map((row, rowIndex) =>
        row.map((cell, colIndex) => (
          <div
            key={`${rowIndex}-${colIndex}`}
            className={`grid-cell ${
              cell ? 'occupied' : ''
            } ${
              shouldHighlightCell(rowIndex, colIndex) ? 'highlight' : ''
            } ${
              isCenterBlocked(rowIndex, colIndex) ? 'blocked' : ''
            }${getSquareBorderClasses(rowIndex, colIndex)}`}
          />
        ))
      )}
    </div>
  );
}

export default GameBoard; 