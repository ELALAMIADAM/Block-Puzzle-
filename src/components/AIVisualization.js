import React, { useEffect, useRef, useCallback } from 'react';

function AIVisualization({ trainingStats, isTraining, compact = false, showOnlyNetwork = false }) {
  const scoreChartRef = useRef(null);
  const rewardChartRef = useRef(null);
  const lossChartRef = useRef(null);
  const networkRef = useRef(null);

  const getMovingAverage = useCallback((data, windowSize = 10) => {
    const result = [];
    for (let i = windowSize - 1; i < data.length; i++) {
      const window = data.slice(i - windowSize + 1, i + 1);
      const average = window.reduce((sum, val) => sum + val, 0) / windowSize;
      result.push(average);
    }
    return result;
  }, []);

  const formatNumber = useCallback((num) => {
    if (typeof num !== 'number' || isNaN(num)) return '0';
    if (num >= 1000000) {
      return (num / 1000000).toFixed(1) + 'M';
    } else if (num >= 1000) {
      return (num / 1000).toFixed(1) + 'K';
    } else if (num >= 1) {
      return num.toFixed(0);
    } else {
      return num.toFixed(2);
    }
  }, []);

  const drawScoreChart = useCallback(() => {
    if (!trainingStats.scores || trainingStats.scores.length === 0) return;
    if (!scoreChartRef.current) return;
    
    const canvas = document.createElement('canvas');
    canvas.width = compact ? 300 : 400;
    canvas.height = compact ? 150 : 200;
    scoreChartRef.current.appendChild(canvas);
    
    const ctx = canvas.getContext('2d');
    const scores = trainingStats.scores.slice(-100); // Show last 100 episodes
    
    if (scores.length === 0) return;
    
    // Chart setup
    const padding = 30;
    const chartWidth = canvas.width - 2 * padding;
    const chartHeight = canvas.height - 2 * padding;
    
    const maxScore = Math.max(...scores, 1);
    const minScore = Math.min(...scores, 0);
    const scoreRange = maxScore - minScore || 1;
    
    // Clear and setup
    ctx.fillStyle = '#1a1a1a';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Grid
    ctx.strokeStyle = '#333';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 5; i++) {
      const y = padding + (i / 5) * chartHeight;
      ctx.beginPath();
      ctx.moveTo(padding, y);
      ctx.lineTo(padding + chartWidth, y);
      ctx.stroke();
    }
    
    // Score line
    ctx.strokeStyle = '#4CAF50';
    ctx.lineWidth = 2;
    ctx.beginPath();
    
    scores.forEach((score, index) => {
      const x = padding + (index / (scores.length - 1)) * chartWidth;
      const y = padding + (1 - (score - minScore) / scoreRange) * chartHeight;
      
      if (index === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });
    ctx.stroke();
    
    // Moving average
    const movingAvg = getMovingAverage(scores, 10);
    if (movingAvg.length > 1) {
      ctx.strokeStyle = '#FFD700';
      ctx.lineWidth = 3;
      ctx.beginPath();
      
      movingAvg.forEach((avg, index) => {
        const x = padding + ((index + 9) / (scores.length - 1)) * chartWidth;
        const y = padding + (1 - (avg - minScore) / scoreRange) * chartHeight;
        
        if (index === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      });
      ctx.stroke();
    }
    
    // Best score line
    if (trainingStats.bestScore > 0) {
      const bestY = padding + (1 - (trainingStats.bestScore - minScore) / scoreRange) * chartHeight;
      ctx.strokeStyle = '#FF4444';
      ctx.lineWidth = 1;
      ctx.setLineDash([5, 5]);
      ctx.beginPath();
      ctx.moveTo(padding, bestY);
      ctx.lineTo(padding + chartWidth, bestY);
      ctx.stroke();
      ctx.setLineDash([]);
    }
    
    // Labels
    ctx.fillStyle = '#FFD700';
    ctx.font = '12px Arial';
    ctx.textAlign = 'left';
    
    // Y-axis labels
    for (let i = 0; i <= 5; i++) {
      const y = padding + (i / 5) * chartHeight;
      const value = maxScore - (i / 5) * scoreRange;
      ctx.fillText(formatNumber(value), 2, y + 4);
    }
    
    // Title and current value
    ctx.textAlign = 'center';
    ctx.font = 'bold 14px Arial';
    ctx.fillText('Score Progress', canvas.width / 2, 20);
    
    const currentScore = scores[scores.length - 1] || 0;
    const avgScore = trainingStats.avgScore || (scores.reduce((a, b) => a + b, 0) / scores.length);
    
    ctx.font = '12px Arial';
    ctx.fillText(`Current: ${formatNumber(currentScore)} | Avg: ${formatNumber(avgScore)} | Best: ${formatNumber(trainingStats.bestScore)}`, 
                canvas.width / 2, canvas.height - 5);
  }, [trainingStats, compact, getMovingAverage, formatNumber]);

  const drawRewardChart = useCallback(() => {
    if (!trainingStats.rewards || trainingStats.rewards.length === 0) return;
    if (!rewardChartRef.current) return;
    
    const canvas = document.createElement('canvas');
    canvas.width = compact ? 300 : 400;
    canvas.height = compact ? 150 : 200;
    rewardChartRef.current.appendChild(canvas);
    
    const ctx = canvas.getContext('2d');
    const rewards = trainingStats.rewards.slice(-100);
    
    if (rewards.length === 0) return;
    
    // Chart setup
    const padding = 30;
    const chartWidth = canvas.width - 2 * padding;
    const chartHeight = canvas.height - 2 * padding;
    
    const maxReward = Math.max(...rewards, 1);
    const minReward = Math.min(...rewards, 0);
    const rewardRange = maxReward - minReward || 1;
    
    // Clear and setup
    ctx.fillStyle = '#1a1a1a';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Grid
    ctx.strokeStyle = '#333';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 5; i++) {
      const y = padding + (i / 5) * chartHeight;
      ctx.beginPath();
      ctx.moveTo(padding, y);
      ctx.lineTo(padding + chartWidth, y);
      ctx.stroke();
    }
    
    // Zero line
    if (minReward < 0 && maxReward > 0) {
      const zeroY = padding + (1 - (0 - minReward) / rewardRange) * chartHeight;
      ctx.strokeStyle = '#666';
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(padding, zeroY);
      ctx.lineTo(padding + chartWidth, zeroY);
      ctx.stroke();
      ctx.setLineDash([]);
    }
    
    // Reward line
    ctx.strokeStyle = '#2196F3';
    ctx.lineWidth = 2;
    ctx.beginPath();
    
    rewards.forEach((reward, index) => {
      const x = padding + (index / (rewards.length - 1)) * chartWidth;
      const y = padding + (1 - (reward - minReward) / rewardRange) * chartHeight;
      
      if (index === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });
    ctx.stroke();
    
    // Moving average
    const movingAvg = getMovingAverage(rewards, 10);
    if (movingAvg.length > 1) {
      ctx.strokeStyle = '#FF9800';
      ctx.lineWidth = 2;
      ctx.beginPath();
      
      movingAvg.forEach((avg, index) => {
        const x = padding + ((index + 9) / (rewards.length - 1)) * chartWidth;
        const y = padding + (1 - (avg - minReward) / rewardRange) * chartHeight;
        
        if (index === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      });
      ctx.stroke();
    }
    
    // Labels
    ctx.fillStyle = '#FFD700';
    ctx.font = '12px Arial';
    ctx.textAlign = 'left';
    
    // Y-axis labels
    for (let i = 0; i <= 5; i++) {
      const y = padding + (i / 5) * chartHeight;
      const value = maxReward - (i / 5) * rewardRange;
      ctx.fillText(formatNumber(value), 2, y + 4);
    }
    
    // Title and current value
    ctx.textAlign = 'center';
    ctx.font = 'bold 14px Arial';
    ctx.fillText('Reward Progress', canvas.width / 2, 20);
    
    const currentReward = rewards[rewards.length - 1] || 0;
    const avgReward = trainingStats.avgReward || (rewards.reduce((a, b) => a + b, 0) / rewards.length);
    
    ctx.font = '12px Arial';
    ctx.fillText(`Current: ${formatNumber(currentReward)} | Avg: ${formatNumber(avgReward)}`, 
                canvas.width / 2, canvas.height - 5);
  }, [trainingStats, compact, getMovingAverage, formatNumber]);

  const drawLossChart = useCallback(() => {
    if (!trainingStats.losses || trainingStats.losses.length === 0) return;
    if (!lossChartRef.current) return;
    
    const canvas = document.createElement('canvas');
    canvas.width = compact ? 300 : 400;
    canvas.height = compact ? 150 : 200;
    lossChartRef.current.appendChild(canvas);
    
    const ctx = canvas.getContext('2d');
    const losses = trainingStats.losses.slice(-100);
    
    if (losses.length === 0) return;
    
    // Chart setup
    const padding = 30;
    const chartWidth = canvas.width - 2 * padding;
    const chartHeight = canvas.height - 2 * padding;
    
    const maxLoss = Math.max(...losses, 1);
    const minLoss = Math.min(...losses, 0);
    const lossRange = maxLoss - minLoss || 1;
    
    // Clear and setup
    ctx.fillStyle = '#1a1a1a';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Grid
    ctx.strokeStyle = '#333';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 5; i++) {
      const y = padding + (i / 5) * chartHeight;
      ctx.beginPath();
      ctx.moveTo(padding, y);
      ctx.lineTo(padding + chartWidth, y);
      ctx.stroke();
    }
    
    // Loss line
    ctx.strokeStyle = '#F44336';
    ctx.lineWidth = 2;
    ctx.beginPath();
    
    losses.forEach((loss, index) => {
      const x = padding + (index / (losses.length - 1)) * chartWidth;
      const y = padding + (1 - (loss - minLoss) / lossRange) * chartHeight;
      
      if (index === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });
    ctx.stroke();
    
    // Moving average
    const movingAvg = getMovingAverage(losses, 10);
    if (movingAvg.length > 1) {
      ctx.strokeStyle = '#9C27B0';
      ctx.lineWidth = 2;
      ctx.beginPath();
      
      movingAvg.forEach((avg, index) => {
        const x = padding + ((index + 9) / (losses.length - 1)) * chartWidth;
        const y = padding + (1 - (avg - minLoss) / lossRange) * chartHeight;
        
        if (index === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      });
      ctx.stroke();
    }
    
    // Labels
    ctx.fillStyle = '#FFD700';
    ctx.font = '12px Arial';
    ctx.textAlign = 'left';
    
    // Y-axis labels
    for (let i = 0; i <= 5; i++) {
      const y = padding + (i / 5) * chartHeight;
      const value = maxLoss - (i / 5) * lossRange;
      ctx.fillText(formatNumber(value), 2, y + 4);
    }
    
    // Title and current value
    ctx.textAlign = 'center';
    ctx.font = 'bold 14px Arial';
    ctx.fillText('Training Loss', canvas.width / 2, 20);
    
    const currentLoss = losses[losses.length - 1] || 0;
    const avgLoss = trainingStats.avgLoss || (losses.reduce((a, b) => a + b, 0) / losses.length);
    
    ctx.font = '12px Arial';
    ctx.fillText(`Current: ${formatNumber(currentLoss)} | Avg: ${formatNumber(avgLoss)}`, 
                canvas.width / 2, canvas.height - 5);
  }, [trainingStats, compact, getMovingAverage, formatNumber]);

  const drawNetworkVisualization = useCallback(() => {
    if (!trainingStats.algorithm) return;
    if (!networkRef.current) return;
    
    const canvas = document.createElement('canvas');
    canvas.width = compact ? 250 : 300;
    canvas.height = compact ? 200 : 250;
    networkRef.current.appendChild(canvas);
    
    const ctx = canvas.getContext('2d');
    
    // Clear canvas
    ctx.fillStyle = '#1a1a1a';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Network architecture visualization
    const layers = [];
    
    if (trainingStats.algorithm.includes('DQN')) {
      // DQN architecture
      layers.push({ name: 'Input', neurons: 'State', color: '#4CAF50' });
      layers.push({ name: 'Hidden 1', neurons: '128', color: '#2196F3' });
      layers.push({ name: 'Hidden 2', neurons: '128', color: '#2196F3' });
      layers.push({ name: 'Hidden 3', neurons: '64', color: '#2196F3' });
      layers.push({ name: 'Output', neurons: 'Actions', color: '#FF9800' });
    } else if (trainingStats.algorithm.includes('Policy')) {
      // Policy Gradient architecture
      layers.push({ name: 'Input', neurons: 'State', color: '#4CAF50' });
      layers.push({ name: 'Hidden 1', neurons: '128', color: '#9C27B0' });
      layers.push({ name: 'Hidden 2', neurons: '128', color: '#9C27B0' });
      layers.push({ name: 'Hidden 3', neurons: '64', color: '#9C27B0' });
      layers.push({ name: 'Policy', neurons: 'Softmax', color: '#E91E63' });
    } else {
      // Generic network
      layers.push({ name: 'Input', neurons: 'State', color: '#4CAF50' });
      layers.push({ name: 'Processing', neurons: 'Rules', color: '#607D8B' });
      layers.push({ name: 'Output', neurons: 'Action', color: '#FF9800' });
    }
    
    const layerWidth = canvas.width / layers.length;
    const centerY = canvas.height / 2;

    // Draw connections
    ctx.strokeStyle = '#444';
    ctx.lineWidth = 1;
    for (let i = 0; i < layers.length - 1; i++) {
      const x1 = (i + 0.5) * layerWidth;
      const x2 = (i + 1.5) * layerWidth;
      
      for (let j = 0; j < 3; j++) {
        const y1 = centerY - 30 + j * 30;
        for (let k = 0; k < 3; k++) {
          const y2 = centerY - 30 + k * 30;
          ctx.beginPath();
          ctx.moveTo(x1, y1);
          ctx.lineTo(x2, y2);
          ctx.stroke();
        }
      }
    }
    
    // Draw layers
    layers.forEach((layer, index) => {
      const x = (index + 0.5) * layerWidth;
      
      // Draw neurons
      ctx.fillStyle = layer.color;
      for (let i = 0; i < 3; i++) {
        const y = centerY - 30 + i * 30;
        ctx.beginPath();
        ctx.arc(x, y, 8, 0, 2 * Math.PI);
        ctx.fill();
      }
      
      // Draw layer label
      ctx.fillStyle = '#FFD700';
      ctx.font = '12px Arial';
      ctx.textAlign = 'center';
      ctx.fillText(layer.name, x, centerY + 60);
      ctx.fillText(layer.neurons, x, centerY + 75);
    });
    
    // Title
    ctx.fillStyle = '#FFD700';
    ctx.font = 'bold 16px Arial';
    ctx.textAlign = 'center';
    ctx.fillText(`${trainingStats.algorithm} Architecture`, canvas.width / 2, 25);
    
    // Status indicators
    const indicators = [];
    
    if (trainingStats.supportsTraining) {
      indicators.push({
        text: `Training: ${isTraining ? 'ACTIVE' : 'STOPPED'}`,
        color: isTraining ? '#4CAF50' : '#F44336'
      });
    }
    
    if (trainingStats.epsilon !== undefined) {
      indicators.push({
        text: `Exploration: ${(trainingStats.epsilon * 100).toFixed(1)}%`,
        color: '#2196F3'
      });
    }
    
    if (trainingStats.avgDecisionTime !== undefined) {
      indicators.push({
        text: `Decision: ${trainingStats.avgDecisionTime.toFixed(1)}ms`,
        color: '#FF9800'
      });
    }
    
    indicators.forEach((indicator, index) => {
      ctx.fillStyle = indicator.color;
      ctx.font = '10px Arial';
      ctx.textAlign = 'left';
      ctx.fillText(indicator.text, 10, canvas.height - 30 + index * 12);
    });
  }, [trainingStats, isTraining, compact]);

  // Algorithm-specific performance indicators
  const getPerformanceIndicator = () => {
    if (!trainingStats || !trainingStats.algorithm) return { text: 'Unknown', color: '#666' };
    
    const algorithm = trainingStats.algorithm;
    
    if (algorithm.includes('DQN')) {
      const avgScore = trainingStats.avgScore || 0;
      if (avgScore > 1000) return { text: 'Excellent', color: '#4CAF50' };
      if (avgScore > 500) return { text: 'Good', color: '#8BC34A' };
      if (avgScore > 100) return { text: 'Learning', color: '#FFC107' };
      return { text: 'Training', color: '#FF9800' };
    } else if (algorithm.includes('MCTS')) {
      const avgDecisionTime = trainingStats.avgDecisionTime || 0;
      if (avgDecisionTime < 100) return { text: 'Fast', color: '#4CAF50' };
      if (avgDecisionTime < 500) return { text: 'Normal', color: '#8BC34A' };
      return { text: 'Slow', color: '#FF9800' };
    } else if (algorithm.includes('Heuristic')) {
      const avgDecisionTime = trainingStats.avgDecisionTime || 0;
      if (avgDecisionTime < 50) return { text: 'Instant', color: '#4CAF50' };
      return { text: 'Fast', color: '#8BC34A' };
    } else if (algorithm.includes('Policy')) {
      const avgLoss = trainingStats.avgLoss || 0;
      if (avgLoss < 0.1) return { text: 'Converged', color: '#4CAF50' };
      if (avgLoss < 0.5) return { text: 'Learning', color: '#FFC107' };
      return { text: 'Training', color: '#FF9800' };
    }
    
    return { text: 'Active', color: '#2196F3' };
  };

  const getLearningIndicator = () => {
    if (!trainingStats.supportsTraining) return null;
    
    if (trainingStats.losses && trainingStats.losses.length > 10) {
      const recentLosses = trainingStats.losses.slice(-10);
      const trend = recentLosses[recentLosses.length - 1] - recentLosses[0];
      
      if (trend < -0.01) return { text: 'Improving', color: '#4CAF50' };
      if (trend > 0.01) return { text: 'Diverging', color: '#F44336' };
      return { text: 'Stable', color: '#FFC107' };
    }
    
    return { text: 'Learning', color: '#2196F3' };
  };

  const getLearningSpeed = () => {
    if (!trainingStats.episode || trainingStats.episode < 10) return null;
    
    if (trainingStats.episode < 50) return { text: 'Starting', color: '#FF9800' };
    if (trainingStats.episode < 200) return { text: 'Training', color: '#2196F3' };
    if (trainingStats.episode < 500) return { text: 'Learning', color: '#9C27B0' };
    return { text: 'Experienced', color: '#4CAF50' };
  };

  useEffect(() => {
    // Clear previous charts
    if (scoreChartRef.current) scoreChartRef.current.innerHTML = '';
    if (rewardChartRef.current) rewardChartRef.current.innerHTML = '';
    if (lossChartRef.current) lossChartRef.current.innerHTML = '';
    if (networkRef.current) networkRef.current.innerHTML = '';

    // Draw charts if we have data
    if (trainingStats && trainingStats.scores && trainingStats.scores.length > 0) {
      drawScoreChart();
      drawRewardChart();
    }
    
    if (trainingStats && trainingStats.losses && trainingStats.losses.length > 0) {
      drawLossChart();
    }
    
    // Draw network visualization for neural network algorithms
    if (!showOnlyNetwork && trainingStats && trainingStats.algorithm && (
      trainingStats.algorithm.includes('DQN') || 
      trainingStats.algorithm.includes('Policy')
    )) {
      drawNetworkVisualization();
    }
  }, [trainingStats, compact, showOnlyNetwork, drawScoreChart, drawRewardChart, drawLossChart, drawNetworkVisualization]);

  if (!trainingStats || Object.keys(trainingStats).length === 0) {
    return (
      <div className="ai-visualization" style={{
        background: 'rgba(139, 69, 19, 0.2)',
        padding: '20px',
        borderRadius: '10px',
        border: '2px solid #8B4513',
        textAlign: 'center'
      }}>
        <h4 style={{ color: '#FFD700', marginBottom: '10px' }}>🤖 AI Visualization</h4>
        <p style={{ color: '#D2B48C' }}>No training data available yet. Start training to see visualizations!</p>
      </div>
    );
  }

  const performanceIndicator = getPerformanceIndicator();
  const learningIndicator = getLearningIndicator();
  const learningSpeed = getLearningSpeed();

  return (
    <div className="ai-visualization" style={{
      background: 'rgba(139, 69, 19, 0.2)',
      padding: '20px',
      borderRadius: '10px',
      border: '2px solid #8B4513'
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
        <h4 style={{ color: '#FFD700', margin: 0 }}>
          🤖 {trainingStats.algorithm || 'AI'} Visualization
        </h4>
        
        <div style={{ display: 'flex', gap: '15px', alignItems: 'center' }}>
          <div style={{
            padding: '4px 8px',
            borderRadius: '12px',
            background: performanceIndicator.color,
            color: 'white',
            fontSize: '12px',
            fontWeight: 'bold'
          }}>
            {performanceIndicator.text}
            </div>
          
          {learningIndicator && (
            <div style={{
              padding: '4px 8px',
              borderRadius: '12px',
              background: learningIndicator.color,
              color: 'white',
              fontSize: '12px',
              fontWeight: 'bold'
            }}>
              {learningIndicator.text}
              </div>
            )}
            
          {learningSpeed && (
            <div style={{
              padding: '4px 8px',
              borderRadius: '12px',
              background: learningSpeed.color,
              color: 'white',
              fontSize: '12px',
              fontWeight: 'bold'
            }}>
              {learningSpeed.text}
              </div>
            )}
          </div>
        </div>

      {/* Algorithm-specific stats */}
        <div style={{ 
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))',
        gap: '15px',
        marginBottom: '20px'
      }}>
        <div style={{ textAlign: 'center' }}>
          <div style={{ color: '#D2B48C', fontSize: '12px' }}>Episodes</div>
          <div style={{ color: '#FFD700', fontSize: '18px', fontWeight: 'bold' }}>
            {trainingStats.episode || 0}
          </div>
        </div>
        
        <div style={{ textAlign: 'center' }}>
          <div style={{ color: '#D2B48C', fontSize: '12px' }}>Best Score</div>
          <div style={{ color: '#4CAF50', fontSize: '18px', fontWeight: 'bold' }}>
            {formatNumber(trainingStats.bestScore || 0)}
          </div>
        </div>
        
        {trainingStats.avgScore !== undefined && (
          <div style={{ textAlign: 'center' }}>
            <div style={{ color: '#D2B48C', fontSize: '12px' }}>Avg Score</div>
            <div style={{ color: '#2196F3', fontSize: '18px', fontWeight: 'bold' }}>
              {formatNumber(trainingStats.avgScore)}
            </div>
        </div>
      )}

        {trainingStats.avgDecisionTime !== undefined && (
          <div style={{ textAlign: 'center' }}>
            <div style={{ color: '#D2B48C', fontSize: '12px' }}>Decision Time</div>
            <div style={{ color: '#FF9800', fontSize: '18px', fontWeight: 'bold' }}>
              {trainingStats.avgDecisionTime.toFixed(1)}ms
      </div>
          </div>
        )}
        
        {trainingStats.avgLoss !== undefined && (
          <div style={{ textAlign: 'center' }}>
            <div style={{ color: '#D2B48C', fontSize: '12px' }}>Training Loss</div>
            <div style={{ color: '#F44336', fontSize: '18px', fontWeight: 'bold' }}>
              {formatNumber(trainingStats.avgLoss)}
          </div>
        </div>
        )}
        
        {trainingStats.memorySize !== undefined && trainingStats.memorySize > 0 && (
          <div style={{ textAlign: 'center' }}>
            <div style={{ color: '#D2B48C', fontSize: '12px' }}>Memory</div>
            <div style={{ color: '#9C27B0', fontSize: '18px', fontWeight: 'bold' }}>
              {trainingStats.memorySize}
            </div>
          </div>
        )}
        
        {trainingStats.epsilon !== undefined && (
          <div style={{ textAlign: 'center' }}>
            <div style={{ color: '#D2B48C', fontSize: '12px' }}>Exploration</div>
            <div style={{ color: '#00BCD4', fontSize: '18px', fontWeight: 'bold' }}>
              {(trainingStats.epsilon * 100).toFixed(1)}%
            </div>
          </div>
        )}
      </div>

      {/* Charts */}
      {!showOnlyNetwork && (
        <div style={{
          display: 'grid',
          gridTemplateColumns: compact ? '1fr' : 'repeat(auto-fit, minmax(300px, 1fr))',
          gap: '20px',
          marginBottom: '20px'
        }}>
          {trainingStats.scores && trainingStats.scores.length > 0 && (
            <div style={{
              background: 'rgba(0, 0, 0, 0.3)',
          borderRadius: '8px', 
              padding: '10px',
              border: '1px solid #8B4513'
            }}>
              <div ref={scoreChartRef} />
          </div>
          )}
          
          {trainingStats.rewards && trainingStats.rewards.length > 0 && (
            <div style={{
              background: 'rgba(0, 0, 0, 0.3)',
          borderRadius: '8px', 
              padding: '10px',
              border: '1px solid #8B4513'
        }}>
              <div ref={rewardChartRef} />
          </div>
          )}
          
          {trainingStats.losses && trainingStats.losses.length > 0 && trainingStats.supportsTraining && (
            <div style={{
              background: 'rgba(0, 0, 0, 0.3)',
            borderRadius: '8px', 
              padding: '10px',
              border: '1px solid #8B4513'
          }}>
              <div ref={lossChartRef} />
          </div>
        )}
      </div>
      )}

      {/* Network Architecture */}
      {(trainingStats.algorithm?.includes('DQN') || trainingStats.algorithm?.includes('Policy')) && (
        <div style={{
          background: 'rgba(0, 0, 0, 0.3)',
          borderRadius: '8px', 
          padding: '10px',
          border: '1px solid #8B4513',
          textAlign: 'center'
        }}>
          <div ref={networkRef} />
        </div>
      )}
      
      {/* Algorithm-specific info */}
        <div style={{ 
        marginTop: '15px',
        padding: '10px',
        background: 'rgba(255, 215, 0, 0.1)',
        borderRadius: '5px',
        border: '1px solid #FFD700'
      }}>
        <div style={{ color: '#FFD700', fontSize: '12px', fontWeight: 'bold', marginBottom: '5px' }}>
          Algorithm: {trainingStats.algorithm || 'Unknown'}
              </div>
        <div style={{ color: '#D2B48C', fontSize: '11px' }}>
          {trainingStats.supportsTraining ? '🎓 Supports Training' : '🚀 No Training Required'} | 
          {trainingStats.supportsVisualization ? ' 📊 Full Visualization' : ' 📈 Basic Stats'}
          {trainingStats.trainingSteps > 0 && ` | ${trainingStats.trainingSteps} Training Steps`}
              </div>
            </div>
    </div>
  );
}

export default AIVisualization; 