import React from 'react';
import { useDrag } from 'react-dnd';

function DraggableBlock({ shape, index, disabled }) {
  const renderBlock = () => {
    const maxCols = Math.max(...shape.map(row => row.length));
    
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

  // Individual cell component with drag functionality
  function DraggableBlockCell({ shape, blockIndex, cellRow, cellCol, isVisible, disabled }) {
    const [{ isDragging }, drag] = useDrag({
      type: 'block',
      item: { 
        shape, 
        index: blockIndex,
        dragCellRow: cellRow,
        dragCellCol: cellCol
      },
      canDrag: !disabled && isVisible,
      collect: (monitor) => ({
        isDragging: !!monitor.isDragging(),
      }),
    });

    if (!isVisible) {
      return (
        <div
          className="block-cell empty"
          style={{ 
            opacity: 0,
            visibility: 'hidden'
          }}
        />
      );
    }

    return (
      <div
        ref={drag}
        className={`block-cell filled ${isDragging ? 'dragging' : ''}`}
        style={{
          opacity: isDragging ? 0.7 : 1,
          cursor: disabled ? 'not-allowed' : (isDragging ? 'grabbing' : 'grab'),
        }}
      />
    );
  }

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