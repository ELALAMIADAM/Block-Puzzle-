import React, { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';

function AIVisualization({ trainingStats, isTraining }) {
  const chartRef = useRef(null);
  const lossChartRef = useRef(null);
  const networkVisualizationRef = useRef(null);
  const [lastUpdateTime, setLastUpdateTime] = useState(Date.now());

  // Always draw network on mount
  useEffect(() => {
    drawNetworkVisualization();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (!trainingStats || Object.keys(trainingStats).length === 0) {
      console.log('🎨 No training stats available for visualization');
      return;
    }
    
    console.log('🎨 Updating AI visualization with stats:', trainingStats);
    
    if (trainingStats.rewards && trainingStats.rewards.length > 0) {
      drawRewardChart();
    }
    
    if (trainingStats.losses && trainingStats.losses.length > 0) {
      drawLossChart();
    }
    
    drawNetworkVisualization();
    setLastUpdateTime(Date.now());
  }, [trainingStats]); // eslint-disable-line react-hooks/exhaustive-deps

  const drawRewardChart = () => {
    const container = chartRef.current;
    if (!container) return;

    // Clear previous chart
    d3.select(container).selectAll("*").remove();

    const { rewards, epsilonHistory } = trainingStats;
    if (!rewards || rewards.length === 0) return;

    // Set dimensions and margins
    const margin = { top: 20, right: 80, bottom: 40, left: 50 };
    const width = 500 - margin.left - margin.right;
    const height = 300 - margin.bottom - margin.top;

    // Create SVG
    const svg = d3.select(container)
      .append("svg")
      .attr("width", width + margin.left + margin.right)
      .attr("height", height + margin.top + margin.bottom)
      .style("background", "rgba(139, 69, 19, 0.1)")
      .style("border-radius", "8px");

    const g = svg.append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    // Create scales
    const xScale = d3.scaleLinear()
      .domain([0, rewards.length - 1])
      .range([0, width]);

    const yScaleReward = d3.scaleLinear()
      .domain(d3.extent(rewards))
      .range([height, 0]);

    const yScaleEpsilon = d3.scaleLinear()
      .domain([0, 1])
      .range([height, 0]);

    // Create line generators
    const rewardLine = d3.line()
      .x((d, i) => xScale(i))
      .y(d => yScaleReward(d))
      .curve(d3.curveMonotoneX);

    const epsilonLine = d3.line()
      .x((d, i) => xScale(i))
      .y(d => yScaleEpsilon(d))
      .curve(d3.curveMonotoneX);

    // Add axes
    g.append("g")
      .attr("transform", `translate(0,${height})`)
      .call(d3.axisBottom(xScale).tickFormat(d3.format("d")).ticks(6))
      .selectAll("text")
      .style("fill", "#FFD700");

    g.append("text")
      .attr("x", width / 2)
      .attr("y", height + 35)
      .attr("fill", "#FFD700")
      .style("text-anchor", "middle")
      .style("font-weight", "bold")
      .text("Episodes");

    g.append("g")
      .call(d3.axisLeft(yScaleReward).ticks(6))
      .selectAll("text")
      .style("fill", "#FFD700");

    g.append("text")
      .attr("transform", "rotate(-90)")
      .attr("y", -35)
      .attr("x", -height / 2)
      .attr("fill", "#FFD700")
      .style("text-anchor", "middle")
      .style("font-weight", "bold")
      .text("Reward");

    // Add right y-axis for epsilon
    g.append("g")
      .attr("transform", `translate(${width}, 0)`)
      .call(d3.axisRight(yScaleEpsilon).ticks(6))
      .selectAll("text")
      .style("fill", "#00CED1");

    g.append("text")
      .attr("transform", "rotate(-90)")
      .attr("y", width + 50)
      .attr("x", -height / 2)
      .attr("fill", "#00CED1")
      .style("text-anchor", "middle")
      .style("font-weight", "bold")
      .text("Epsilon");

    // Style axes
    g.selectAll(".domain")
      .style("stroke", "#8B4513");
    
    g.selectAll(".tick line")
      .style("stroke", "#8B4513");

    // Add grid lines
    g.selectAll(".grid-line")
      .data(yScaleReward.ticks())
      .enter()
      .append("line")
      .attr("class", "grid-line")
      .attr("x1", 0)
      .attr("x2", width)
      .attr("y1", d => yScaleReward(d))
      .attr("y2", d => yScaleReward(d))
      .attr("stroke", "#8B4513")
      .attr("stroke-opacity", 0.2)
      .attr("stroke-width", 1);

    // Add reward line with animation
    const rewardPath = g.append("path")
      .datum(rewards)
      .attr("fill", "none")
      .attr("stroke", "#FFD700")
      .attr("stroke-width", 3)
      .attr("d", rewardLine);

    // Animate the line drawing
    const totalLength = rewardPath.node().getTotalLength();
    rewardPath
      .attr("stroke-dasharray", totalLength + " " + totalLength)
      .attr("stroke-dashoffset", totalLength)
      .transition()
      .duration(2000)
      .attr("stroke-dashoffset", 0);

    // Add epsilon line if available
    if (epsilonHistory && epsilonHistory.length > 0) {
      const epsilonPath = g.append("path")
        .datum(epsilonHistory)
        .attr("fill", "none")
        .attr("stroke", "#00CED1")
        .attr("stroke-width", 2)
        .attr("stroke-dasharray", "5,5")
        .attr("d", epsilonLine);

      // Animate epsilon line
      const epsilonLength = epsilonPath.node().getTotalLength();
      epsilonPath
        .attr("stroke-dasharray", epsilonLength + " " + epsilonLength)
        .attr("stroke-dashoffset", epsilonLength)
        .transition()
        .delay(500)
        .duration(1500)
        .attr("stroke-dashoffset", 0);
    }

    // Add moving average line
    const movingAvg = getMovingAverage(rewards, 10);
    if (movingAvg.length > 0) {
      const avgLine = d3.line()
        .x((d, i) => xScale(i + 9))
        .y(d => yScaleReward(d))
        .curve(d3.curveMonotoneX);

      g.append("path")
        .datum(movingAvg)
        .attr("fill", "none")
        .attr("stroke", "#FF6B6B")
        .attr("stroke-width", 2)
        .attr("opacity", 0.8)
        .attr("d", avgLine);
    }

    // Add legend
    const legend = g.append("g")
      .attr("transform", `translate(${width - 100}, 20)`);

    const legendData = [
      { color: "#FFD700", text: "Reward", pattern: "none" },
      { color: "#00CED1", text: "Epsilon", pattern: "5,5" },
      { color: "#FF6B6B", text: "Avg (10)", pattern: "none" }
    ];

    legendData.forEach((item, i) => {
      const legendItem = legend.append("g")
        .attr("transform", `translate(0, ${i * 20})`);

      legendItem.append("line")
        .attr("x1", 0).attr("x2", 20)
        .attr("y1", 0).attr("y2", 0)
        .attr("stroke", item.color)
        .attr("stroke-width", 2)
        .attr("stroke-dasharray", item.pattern);

      legendItem.append("text")
        .attr("x", 25).attr("y", 5)
        .attr("fill", item.color)
        .style("font-size", "12px")
        .style("font-weight", "bold")
        .text(item.text);
    });
  };

  const drawLossChart = () => {
    const container = lossChartRef.current;
    if (!container || !trainingStats.losses || trainingStats.losses.length === 0) return;

    // Clear previous chart
    d3.select(container).selectAll("*").remove();

    const losses = trainingStats.losses;
    
    // Set dimensions and margins
    const margin = { top: 20, right: 30, bottom: 40, left: 60 };
    const width = 400 - margin.left - margin.right;
    const height = 200 - margin.bottom - margin.top;

    // Create SVG
    const svg = d3.select(container)
      .append("svg")
      .attr("width", width + margin.left + margin.right)
      .attr("height", height + margin.top + margin.bottom)
      .style("background", "rgba(139, 69, 19, 0.1)")
      .style("border-radius", "8px");

    const g = svg.append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    // Create scales
    const xScale = d3.scaleLinear()
      .domain([0, losses.length - 1])
      .range([0, width]);

    const yScale = d3.scaleLinear()
      .domain(d3.extent(losses))
      .range([height, 0]);

    // Create line generator
    const line = d3.line()
      .x((d, i) => xScale(i))
      .y(d => yScale(d))
      .curve(d3.curveMonotoneX);

    // Add axes
    g.append("g")
      .attr("transform", `translate(0,${height})`)
      .call(d3.axisBottom(xScale).tickFormat(d3.format("d")).ticks(5))
      .selectAll("text")
      .style("fill", "#FFD700");

    g.append("text")
      .attr("x", width / 2)
      .attr("y", height + 35)
      .attr("fill", "#FFD700")
      .style("text-anchor", "middle")
      .style("font-weight", "bold")
      .text("Training Steps");

    g.append("g")
      .call(d3.axisLeft(yScale).tickFormat(d3.format(".3f")).ticks(5))
      .selectAll("text")
      .style("fill", "#FFD700");

    g.append("text")
      .attr("transform", "rotate(-90)")
      .attr("y", -40)
      .attr("x", -height / 2)
      .attr("fill", "#FFD700")
      .style("text-anchor", "middle")
      .style("font-weight", "bold")
      .text("Loss");

    // Style axes
    g.selectAll(".domain")
      .style("stroke", "#8B4513");
    
    g.selectAll(".tick line")
      .style("stroke", "#8B4513");

    // Add gradient
    const gradient = svg.append("defs")
      .append("linearGradient")
      .attr("id", "loss-gradient")
      .attr("gradientUnits", "userSpaceOnUse")
      .attr("x1", 0).attr("y1", height)
      .attr("x2", 0).attr("y2", 0);

    gradient.append("stop")
      .attr("offset", "0%")
      .attr("stop-color", "#FF6B6B")
      .attr("stop-opacity", 0.1);

    gradient.append("stop")
      .attr("offset", "100%")
      .attr("stop-color", "#FF6B6B")
      .attr("stop-opacity", 0.8);

    // Add area under curve
    const area = d3.area()
      .x((d, i) => xScale(i))
      .y0(height)
      .y1(d => yScale(d))
      .curve(d3.curveMonotoneX);

    g.append("path")
      .datum(losses)
      .attr("fill", "url(#loss-gradient)")
      .attr("d", area);

    // Add loss line
    const lossPath = g.append("path")
      .datum(losses)
      .attr("fill", "none")
      .attr("stroke", "#FF6B6B")
      .attr("stroke-width", 2)
      .attr("d", line);

    // Animate loss line
    const totalLength = lossPath.node().getTotalLength();
    lossPath
      .attr("stroke-dasharray", totalLength + " " + totalLength)
      .attr("stroke-dashoffset", totalLength)
      .transition()
      .duration(1500)
      .attr("stroke-dashoffset", 0);
  };

  const drawNetworkVisualization = () => {
    const container = networkVisualizationRef.current;
    if (!container) return;

    // Clear previous visualization
    d3.select(container).selectAll("*").remove();

    // IMPROVED Network architecture: [112, 256, 256, 128, 64, 147]
    const layers = [112, 256, 256, 128, 64, 147];
    const layerNames = ['Input\n(State)', 'Hidden 1\n(BatchNorm)', 'Hidden 2\n(BatchNorm)', 'Hidden 3\n(ReLU)', 'Hidden 4\n(ReLU)', 'Output\n(Actions)'];
    
    const width = 800;
    const height = 400;
    const margin = { top: 60, right: 30, bottom: 100, left: 30 };
    
    const svg = d3.select(container)
      .append("svg")
      .attr("width", width)
      .attr("height", height)
      .style("background", "rgba(139, 69, 19, 0.1)")
      .style("border-radius", "8px");

    const layerWidth = (width - margin.left - margin.right) / (layers.length - 1);

    // Draw connections
    for (let i = 0; i < layers.length - 1; i++) {
      for (let j = 0; j < Math.min(3, layers[i]); j++) {
        for (let k = 0; k < Math.min(3, layers[i + 1]); k++) {
          const x1 = margin.left + i * layerWidth;
          const y1 = margin.top + (j + 1) * (height - margin.top - margin.bottom) / 4;
          const x2 = margin.left + (i + 1) * layerWidth;
          const y2 = margin.top + (k + 1) * (height - margin.top - margin.bottom) / 4;
          
          svg.append("line")
            .attr("x1", x1)
            .attr("y1", y1)
            .attr("x2", x2)
            .attr("y2", y2)
            .attr("stroke", "#8B4513")
            .attr("stroke-width", 1)
            .attr("opacity", 0.4);
        }
      }
    }

    // Draw nodes
    layers.forEach((nodeCount, layerIndex) => {
      const x = margin.left + layerIndex * layerWidth;
      
      let nodeColor;
      if (layerIndex === 0) nodeColor = "#FFD700";
      else if (layerIndex === layers.length - 1) nodeColor = "#FF6B6B";
      else nodeColor = "#00CED1";
      
      for (let nodeIndex = 0; nodeIndex < Math.min(3, nodeCount); nodeIndex++) {
        const y = margin.top + (nodeIndex + 1) * (height - margin.top - margin.bottom) / 4;
        
        const circle = svg.append("circle")
          .attr("cx", x)
          .attr("cy", y)
          .attr("r", 0)
          .attr("fill", nodeColor)
          .attr("stroke", "#8B4513")
          .attr("stroke-width", 2);

        // Animate node appearance
        circle.transition()
          .delay(layerIndex * 200 + nodeIndex * 50)
          .duration(500)
          .attr("r", 8);
      }

      // Add layer labels
      svg.append("text")
        .attr("x", x)
        .attr("y", height - 20)
        .attr("text-anchor", "middle")
        .attr("fill", "#FFD700")
        .style("font-size", "11px")
        .style("font-weight", "bold")
        .text(layerNames[layerIndex]);
      
      // Add node count
      svg.append("text")
        .attr("x", x)
        .attr("y", 25)
        .attr("text-anchor", "middle")
        .attr("fill", "#D2B48C")
        .style("font-size", "10px")
        .text(nodeCount);
    });

    // Add title
    svg.append("text")
      .attr("x", width / 2)
      .attr("y", 15)
      .attr("text-anchor", "middle")
      .attr("fill", "#FFD700")
      .style("font-size", "16px")
      .style("font-weight", "bold")
      .text("DQN Neural Network Architecture");
    
    // Add parameter information
    const parameterInfo = [
      { label: "Total Parameters", value: "~450k" },
      { label: "Input Features", value: "112" },
      { label: "Output Actions", value: "147" },
      { label: "Hidden Layers", value: "4" },
      { label: "Activation", value: "ReLU" },
      { label: "Loss Function", value: "Huber" }
    ];
    
    const paramY = height - 80;
    parameterInfo.forEach((param, index) => {
      const x = margin.left + (index * (width - margin.left - margin.right) / (parameterInfo.length - 1));
      
      // Parameter box
      svg.append("rect")
        .attr("x", x - 40)
        .attr("y", paramY)
        .attr("width", 80)
        .attr("height", 40)
        .attr("fill", "rgba(139, 69, 19, 0.3)")
        .attr("stroke", "#8B4513")
        .attr("stroke-width", 1)
        .attr("rx", 5);
      
      // Parameter label
      svg.append("text")
        .attr("x", x)
        .attr("y", paramY + 12)
        .attr("text-anchor", "middle")
        .attr("fill", "#FFD700")
        .style("font-size", "10px")
        .style("font-weight", "bold")
        .text(param.label);
      
      // Parameter value
      svg.append("text")
        .attr("x", x)
        .attr("y", paramY + 28)
        .attr("text-anchor", "middle")
        .attr("fill", "#D2B48C")
        .style("font-size", "12px")
        .style("font-weight", "bold")
        .text(param.value);
    });
    
    // Add model state indicator
    const modelState = trainingStats && trainingStats.episode > 0 ? "Trained" : "Untrained";
    const stateColor = modelState === "Trained" ? "#28a745" : "#ffc107";
    
    svg.append("circle")
      .attr("cx", width - 50)
      .attr("cy", 30)
      .attr("r", 8)
      .attr("fill", stateColor)
      .attr("stroke", "#8B4513")
      .attr("stroke-width", 2);
    
    svg.append("text")
      .attr("x", width - 50)
      .attr("y", 50)
      .attr("text-anchor", "middle")
      .attr("fill", stateColor)
      .style("font-size", "12px")
      .style("font-weight", "bold")
      .text(modelState);
  };

  const getMovingAverage = (data, windowSize = 10) => {
    if (!data || data.length < windowSize) return [];
    
    const result = [];
    for (let i = windowSize - 1; i < data.length; i++) {
      const window = data.slice(i - windowSize + 1, i + 1);
      const average = window.reduce((sum, val) => sum + val, 0) / windowSize;
      result.push(average);
    }
    return result;
  };

  const formatNumber = (num) => {
    if (typeof num !== 'number') return '0';
    if (num >= 1000) return (num / 1000).toFixed(1) + 'k';
    return num.toFixed(2);
  };

  const getPerformanceIndicator = () => {
    if (!trainingStats.rewards || trainingStats.rewards.length < 10) {
      return { color: '#808080', text: 'Collecting data...' };
    }
    
    const recentRewards = trainingStats.rewards.slice(-10);
    const avgReward = recentRewards.reduce((a, b) => a + b, 0) / recentRewards.length;
    
    if (avgReward > 500) {
      return { color: '#28a745', text: 'Excellent' };
    } else if (avgReward > 200) {
      return { color: '#ffc107', text: 'Good' };
    } else if (avgReward > 0) {
      return { color: '#fd7e14', text: 'Learning' };
    } else {
      return { color: '#dc3545', text: 'Poor' };
    }
  };

  const performance = getPerformanceIndicator();

  // Real-time learning indicator
  const getLearningIndicator = () => {
    if (!trainingStats.rewards || trainingStats.rewards.length < 5) {
      return { color: '#808080', text: 'Initializing...', trend: '⏳' };
    }
    
    const recentRewards = trainingStats.rewards.slice(-5);
    const trend = recentRewards[recentRewards.length - 1] - recentRewards[0];
    
    if (trend > 50) {
      return { color: '#28a745', text: 'Learning Fast!', trend: '📈' };
    } else if (trend > 10) {
      return { color: '#ffc107', text: 'Improving', trend: '📊' };
    } else if (trend > -10) {
      return { color: '#fd7e14', text: 'Stable', trend: '➡️' };
    } else {
      return { color: '#dc3545', text: 'Struggling', trend: '📉' };
    }
  };

  // Calculate learning speed
  const getLearningSpeed = () => {
    if (!trainingStats.rewards || trainingStats.rewards.length < 10) return 'N/A';
    
    const recent = trainingStats.rewards.slice(-10);
    const older = trainingStats.rewards.slice(-20, -10);
    
    if (older.length === 0) return 'N/A';
    
    const recentAvg = recent.reduce((a, b) => a + b, 0) / recent.length;
    const olderAvg = older.reduce((a, b) => a + b, 0) / older.length;
    
    const improvement = ((recentAvg - olderAvg) / Math.abs(olderAvg)) * 100;
    
    if (improvement > 10) return 'Fast';
    if (improvement > 3) return 'Moderate';
    if (improvement > -3) return 'Slow';
    return 'Negative';
  };

  const learningIndicator = getLearningIndicator();
  const learningSpeed = getLearningSpeed();

  return (
    <div className="ai-visualization">
      <h4>📊 Neural Network Training Progress</h4>
      
      {trainingStats && trainingStats.rewards && trainingStats.rewards.length > 0 ? (
        <div className="viz-container">
          <div className="chart-section">
            <h5>Reward & Exploration Progress</h5>
            <div ref={chartRef} className="chart-container"></div>
          </div>
          
          <div className="chart-section">
            <h5>Training Loss</h5>
            <div ref={lossChartRef} className="chart-container"></div>
          </div>
        </div>
      ) : (
        <div style={{ 
          textAlign: 'center', 
          padding: '40px',
          background: 'rgba(139, 69, 19, 0.2)',
          borderRadius: '10px',
          border: '1px solid #8B4513',
          margin: '20px 0'
        }}>
          <div style={{ fontSize: '48px', marginBottom: '20px' }}>📊</div>
          <h3 style={{ color: '#FFD700', marginBottom: '10px' }}>Ready to Visualize Training</h3>
          <p style={{ color: '#D2B48C', fontSize: '16px', marginBottom: '20px' }}>
            Start training to see real-time learning progress!
          </p>
          <div style={{ 
            display: 'grid', 
            gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
            gap: '15px',
            marginTop: '20px'
          }}>
            <div style={{ 
              background: 'rgba(139, 69, 19, 0.3)',
              padding: '15px',
              borderRadius: '8px',
              border: '1px solid #8B4513'
            }}>
              <div style={{ color: '#FFD700', fontWeight: 'bold', marginBottom: '5px' }}>📈 Reward Charts</div>
              <div style={{ color: '#D2B48C', fontSize: '12px' }}>Track AI performance over episodes</div>
            </div>
            <div style={{ 
              background: 'rgba(139, 69, 19, 0.3)',
              padding: '15px',
              borderRadius: '8px',
              border: '1px solid #8B4513'
            }}>
              <div style={{ color: '#FFD700', fontWeight: 'bold', marginBottom: '5px' }}>🧠 Learning Speed</div>
              <div style={{ color: '#D2B48C', fontSize: '12px' }}>Monitor real-time learning progress</div>
            </div>
            <div style={{ 
              background: 'rgba(139, 69, 19, 0.3)',
              padding: '15px',
              borderRadius: '8px',
              border: '1px solid #8B4513'
            }}>
              <div style={{ color: '#FFD700', fontWeight: 'bold', marginBottom: '5px' }}>🎮 Visual Training</div>
              <div style={{ color: '#D2B48C', fontSize: '12px' }}>Watch AI play on the game board</div>
            </div>
          </div>
        </div>
      )}

      <div className="network-section">
        <h5>Neural Network Architecture</h5>
        <div ref={networkVisualizationRef} className="network-container"></div>
      </div>
        
      <div className="performance-metrics">
        <div className="metric-card">
          <div className="metric-title">Performance</div>
          <div 
            className="metric-value"
            style={{ color: performance.color }}
          >
            {performance.text}
          </div>
        </div>
        
        {trainingStats.rewards && (
          <div className="metric-card">
            <div className="metric-title">Avg Reward (Last 10)</div>
            <div className="metric-value">
              {formatNumber(getMovingAverage(trainingStats.rewards, 10).slice(-1)[0] || 0)}
            </div>
          </div>
        )}
        
        <div className="metric-card">
          <div className="metric-title">Learning Speed</div>
          <div 
            className="metric-value"
            style={{ 
              color: learningSpeed === 'Fast' ? '#28a745' : 
                     learningSpeed === 'Moderate' ? '#ffc107' : 
                     learningSpeed === 'Slow' ? '#fd7e14' : '#dc3545' 
            }}
          >
            {learningSpeed}
          </div>
        </div>
        
        <div className="metric-card">
          <div className="metric-title">Exploration Rate</div>
          <div className="metric-value">
            {((trainingStats.epsilon || 0) * 100).toFixed(1)}%
          </div>
        </div>
        
        {trainingStats.avgLoss && (
          <div className="metric-card">
            <div className="metric-title">Training Loss</div>
            <div className="metric-value">
              {trainingStats.avgLoss.toFixed(4)}
            </div>
          </div>
        )}
      </div>
      
      {isTraining && (
        <div className="training-status">
          <div className="status-indicator pulsing"></div>
          <span>Neural network is actively learning...</span>
        </div>
      )}
      
      {/* Real-time Learning Status */}
      {isTraining && (
        <div style={{ 
          background: 'rgba(40, 167, 69, 0.2)', 
          padding: '15px', 
          borderRadius: '8px', 
          marginBottom: '20px',
          border: '1px solid #28a745',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <span style={{ fontSize: '24px' }}>{learningIndicator.trend}</span>
            <div>
              <div style={{ color: learningIndicator.color, fontWeight: 'bold', fontSize: '16px' }}>
                {learningIndicator.text}
              </div>
              <div style={{ fontSize: '12px', color: '#D2B48C' }}>
                Learning Speed: {learningSpeed}
              </div>
            </div>
          </div>
          <div style={{ textAlign: 'right' }}>
            <div style={{ fontSize: '12px', color: '#D2B48C' }}>Last Update</div>
            <div style={{ fontSize: '14px', color: '#28a745' }}>
              {new Date(lastUpdateTime).toLocaleTimeString()}
            </div>
          </div>
        </div>
      )}
      
      <div className="viz-tips">
        <h5>💡 What to Watch During Training:</h5>
        <ul>
          <li><strong>📈 Learning Fast! / 📊 Improving:</strong> The AI is successfully learning new strategies</li>
          <li><strong>➡️ Stable:</strong> AI has learned a strategy and is refining it</li>
          <li><strong>📉 Struggling:</strong> AI is having difficulty - consider more training episodes</li>
          <li><strong>Gold Line Rising:</strong> Episode rewards trending up = AI getting better at the game</li>
          <li><strong>Cyan Line Falling:</strong> Exploration rate decreasing = AI becoming more confident</li>
          <li><strong>Loss Chart Converging:</strong> Neural network training stabilizing = good learning</li>
          <li><strong>Learning Speed: Fast/Moderate:</strong> AI is actively improving its performance</li>
        </ul>
        
        <div style={{ 
          marginTop: '15px', 
          padding: '10px', 
          background: 'rgba(255, 215, 0, 0.1)', 
          borderRadius: '5px',
          border: '1px solid #FFD700'
        }}>
          <strong style={{ color: '#FFD700' }}>🎯 Training Tips:</strong>
          <ul style={{ marginTop: '8px', fontSize: '11px' }}>
            <li>Start with <strong>100-500 episodes</strong> for initial learning</li>
            <li>Look for <strong>upward trending rewards</strong> as a sign of progress</li>
            <li>If performance plateaus, try <strong>1000+ episodes</strong> for refinement</li>
            <li>Good performance: average rewards above <strong>200 points</strong></li>
            <li>The visualization updates <strong>every 5 episodes</strong> for real-time progress</li>
          </ul>
        </div>
      </div>
    </div>
  );
}

export default AIVisualization; 