import React from 'react';
import DraggableBlockCell from './DraggableBlockCell';

function DraggableBlock({ shape, index, disabled }) {
  const renderBlock = () => {
    const maxCols = Math.max(...shape.map(row => row.length));
    let cellIndex = 0;
    
    return (
      <div 
        className="block" 
        style={{ 
          gridTemplateColumns: `repeat(${maxCols}, 1fr)`,
          gridTemplateRows: `repeat(${shape.length}, 1fr)`
        }}
      >
        {shape.map((row, rowIndex) =>
          row.map((cell, colIndex) => {
            const currentCellIndex = cellIndex++;
            return (
              <DraggableBlockCell
                key={`${rowIndex}-${colIndex}`}
                shape={shape}
                blockIndex={index}
                cellRow={rowIndex}
                cellCol={colIndex}
                isVisible={cell}
                disabled={disabled}
              />
            );
          })
        )}
      </div>
    );
  };

  return (
    <div
      className={`block-container ${disabled ? 'disabled' : ''}`}
      style={{
        opacity: disabled ? 0.5 : 1,
      }}
    >
      {renderBlock()}
    </div>
  );
}

export default DraggableBlock; 