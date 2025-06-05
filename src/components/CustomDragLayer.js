import React from 'react';
import { useDragLayer } from 'react-dnd';

const layerStyles = {
  position: 'fixed',
  pointerEvents: 'none',
  zIndex: 1000,
  left: 0,
  top: 0,
  width: '100%',
  height: '100%',
};

function getItemStyles(initialOffset, currentOffset, dragCellRow, dragCellCol) {
  if (!initialOffset || !currentOffset) {
    return {
      display: 'none',
    };
  }

  // Calculate the offset to position the block correctly relative to the dragged cell
  const cellSize = 25; // Same as block-cell size
  const gap = 0; // Gap between cells
  const offsetX = -(dragCellCol * (cellSize + gap));
  const offsetY = -(dragCellRow * (cellSize + gap));

  const { x, y } = currentOffset;

  const transform = `translate(${x + offsetX}px, ${y + offsetY}px)`;
  return {
    transform,
    WebkitTransform: transform,
  };
}

function CustomDragLayer() {
  const {
    itemType,
    isDragging,
    item,
    initialOffset,
    currentOffset,
  } = useDragLayer((monitor) => ({
    item: monitor.getItem(),
    itemType: monitor.getItemType(),
    initialOffset: monitor.getInitialSourceClientOffset(),
    currentOffset: monitor.getSourceClientOffset(),
    isDragging: monitor.isDragging(),
  }));

  function renderItem() {
    if (itemType !== 'block' || !item) {
      return null;
    }

    const { shape, dragCellRow, dragCellCol } = item;
    const maxCols = Math.max(...shape.map(row => row.length));

    return (
      <div 
        className="block drag-preview" 
        style={{ 
          gridTemplateColumns: `repeat(${maxCols}, 1fr)`,
          gridTemplateRows: `repeat(${shape.length}, 1fr)`,
          gap: '0px',
          padding: '5px',
          background: 'rgba(139, 69, 19, 0.8)',
          borderRadius: '8px',
          boxShadow: '0 8px 16px rgba(0, 0, 0, 0.3)',
        }}
      >
        {shape.map((row, rowIndex) =>
          row.map((cell, colIndex) => (
            <div
              key={`${rowIndex}-${colIndex}`}
              className={`block-cell ${cell ? 'filled' : 'empty'}`}
              style={{ 
                width: '25px',
                height: '25px',
                background: cell 
                  ? (rowIndex === dragCellRow && colIndex === dragCellCol)
                    ? 'linear-gradient(145deg, #FFD700, #DAA520)' // Highlight the dragged cell
                    : 'linear-gradient(145deg, #CD853F, #8B4513)'
                  : 'transparent',
                border: cell ? '1px solid #654321' : 'none',
                borderRadius: '3px',
                opacity: cell ? (rowIndex === dragCellRow && colIndex === dragCellCol ? 1 : 0.9) : 0,
                visibility: cell ? 'visible' : 'hidden',
                boxShadow: cell ? '0 2px 4px rgba(0, 0, 0, 0.2)' : 'none',
              }}
            />
          ))
        )}
      </div>
    );
  }

  if (!isDragging) {
    return null;
  }

  return (
    <div style={layerStyles}>
      <div
        style={getItemStyles(
          initialOffset, 
          currentOffset, 
          item?.dragCellRow || 0, 
          item?.dragCellCol || 0
        )}
      >
        {renderItem()}
      </div>
    </div>
  );
}

export default CustomDragLayer; 