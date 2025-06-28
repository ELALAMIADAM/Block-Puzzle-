import React from 'react';
import DraggableBlock from './DraggableBlock';

function BlockTray({ blocks, disabled }) {
  return (
    <div className="block-tray" style={{
      display: 'flex',
      flexDirection: 'column',
      gap: '10px',
      width: '100%',
      height: '100%',
      justifyContent: 'flex-start',
      alignItems: 'center',
      minHeight: '150px' // Increased minimum height to work with larger containers
    }}>
      {blocks.map((block, index) => (
        <DraggableBlock
          key={`${index}-${Date.now()}`}
          shape={block}
          index={index}
          disabled={disabled}
        />
      ))}
      {/* Fill remaining space if there are fewer than 3 blocks */}
      {blocks.length < 3 && Array.from({ length: 3 - blocks.length }).map((_, index) => (
        <div 
          key={`placeholder-${index}`}
          style={{
            width: '80px',
            height: '40px',
            minHeight: '40px',
            border: '2px dashed rgba(139, 69, 19, 0.3)',
            borderRadius: '10px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: 'rgba(139, 69, 19, 0.5)',
            fontSize: '12px',
            fontStyle: 'italic'
          }}
        >
          {disabled ? 'Loading...' : 'Empty'}
        </div>
      ))}
    </div>
  );
}

export default BlockTray; 