import React from 'react';
import DraggableBlock from './DraggableBlock';

function BlockTray({ blocks, disabled }) {
  return (
    <div className="block-tray">
      {blocks.map((block, index) => (
        <DraggableBlock
          key={`${index}-${Date.now()}`}
          shape={block}
          index={index}
          disabled={disabled}
        />
      ))}
    </div>
  );
}

export default BlockTray; 