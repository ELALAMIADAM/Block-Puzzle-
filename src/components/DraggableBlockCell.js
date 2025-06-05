import React, { useEffect } from 'react';
import { useDrag } from 'react-dnd';
import { getEmptyImage } from 'react-dnd-html5-backend';

function DraggableBlockCell({ shape, blockIndex, cellRow, cellCol, isVisible, disabled }) {
  const [{ isDragging }, drag, preview] = useDrag({
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

  // Use empty image as drag preview to create custom preview
  useEffect(() => {
    preview(getEmptyImage(), { captureDraggingState: true });
  }, [preview]);

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

export default DraggableBlockCell; 