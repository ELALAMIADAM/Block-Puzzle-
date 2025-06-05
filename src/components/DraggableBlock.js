import React from 'react';
import { useDrag } from 'react-dnd';

function DraggableBlock({ shape, index, disabled }) {
  const [{ isDragging }, drag] = useDrag({
    type: 'block',
    item: { shape, index },
    canDrag: !disabled,
    collect: (monitor) => ({
      isDragging: !!monitor.isDragging(),
    }),
  });

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
        {shape.flat().map((cell, cellIndex) => (
          <div
            key={cellIndex}
            className={`block-cell ${cell ? 'filled' : 'empty'}`}
            style={{ 
              opacity: cell ? 1 : 0,
              visibility: cell ? 'visible' : 'hidden'
            }}
          />
        ))}
      </div>
    );
  };

  return (
    <div
      ref={drag}
      className={`block-container ${isDragging ? 'dragging' : ''} ${disabled ? 'disabled' : ''}`}
      style={{
        opacity: disabled ? 0.5 : (isDragging ? 0.7 : 1),
        cursor: disabled ? 'not-allowed' : (isDragging ? 'grabbing' : 'grab'),
      }}
    >
      {renderBlock()}
    </div>
  );
}

export default DraggableBlock; 