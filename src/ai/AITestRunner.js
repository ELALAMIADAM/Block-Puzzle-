import { DQNAgent } from './DQNAgent';
import { DQNEnvironment } from './DQNEnvironment';
import { EliteDQNAgent } from './EliteDQNAgent';
import { EliteEnvironment } from './EliteEnvironment';
import { AlgorithmSelector } from './AdvancedAIAgents';
import { generateRandomBlocks } from '../utils/gameLogic';

export async function runAITests() {
  console.log('🧪 Starting Comprehensive AI System Tests...');
  
  const results = {
    dqnTest: false,
    eliteDqnTest: false,
    mctsTest: false,
    policyGradientTest: false,
    heuristicTest: false,
    stateEncodingTest: false,
    actionDecodingTest: false,
    trainingTest: false,
    performanceTest: false
  };

  try {
    // Test 1: Original DQN System
    console.log('📋 Test 1: Original DQN System');
    try {
      const dqnEnv = new DQNEnvironment();
      dqnEnv.reset();
      const dqnAgent = new DQNAgent(dqnEnv.getStateSize(), dqnEnv.getMaxActionSpace());
      
      if (dqnAgent.qNetwork && dqnAgent.targetNetwork) {
        console.log('✅ Original DQN system working');
        results.dqnTest = true;
      }
      
      // Cleanup
      dqnAgent.dispose();
    } catch (error) {
      console.log('❌ Original DQN test failed:', error.message);
    }

    // Test 2: Elite DQN System
    console.log('📋 Test 2: Elite DQN System');
    try {
      const eliteEnv = new EliteEnvironment();
      eliteEnv.reset();
      const eliteAgent = new EliteDQNAgent(eliteEnv.getStateSize(), eliteEnv.getMaxActionSpace(), {
        learningRate: 0.001,
        epsilon: 0.5,
        batchSize: 32
      });
      
      if (eliteAgent.qNetwork && eliteAgent.targetNetwork && eliteAgent.isElite) {
        console.log('✅ Elite DQN system working');
        results.eliteDqnTest = true;
      }
      
      // Cleanup
      eliteAgent.dispose();
    } catch (error) {
      console.log('❌ Elite DQN test failed:', error.message);
    }

    // Test 3: MCTS Algorithm
    console.log('📋 Test 3: MCTS Algorithm');
    try {
      const mctsEnv = new EliteEnvironment();
      mctsEnv.reset();
      const mctsAgent = AlgorithmSelector.createAgent('mcts', mctsEnv.getStateSize(), mctsEnv.getMaxActionSpace(), {
        maxSimulations: 100,
        explorationConstant: 1.414
      });
      
      // Test action selection
      const action = await mctsAgent.selectAction(mctsEnv);
      if (typeof action === 'number') {
        console.log('✅ MCTS algorithm working');
        results.mctsTest = true;
      }
      
      // Cleanup
      if (mctsAgent.dispose) mctsAgent.dispose();
    } catch (error) {
      console.log('❌ MCTS test failed:', error.message);
    }

    // Test 4: Policy Gradient Algorithm
    console.log('📋 Test 4: Policy Gradient Algorithm');
    try {
      const pgEnv = new EliteEnvironment();
      pgEnv.reset();
      const pgAgent = AlgorithmSelector.createAgent('policy-gradient', pgEnv.getStateSize(), pgEnv.getMaxActionSpace(), {
        learningRate: 0.001,
        gamma: 0.99
      });
      
      // Test action selection
      const state = pgEnv.getState();
      const validActions = pgEnv.getValidActions();
      const action = await pgAgent.selectAction(state, validActions);
      
      if (typeof action === 'number') {
        console.log('✅ Policy Gradient algorithm working');
        results.policyGradientTest = true;
      }
      
      // Cleanup
      state.dispose();
      if (pgAgent.dispose) pgAgent.dispose();
    } catch (error) {
      console.log('❌ Policy Gradient test failed:', error.message);
    }

    // Test 5: Heuristic Algorithm
    console.log('📋 Test 5: Heuristic Algorithm');
    try {
      const heuristicEnv = new EliteEnvironment();
      heuristicEnv.reset();
      const heuristicAgent = AlgorithmSelector.createAgent('heuristic', heuristicEnv.getStateSize(), heuristicEnv.getMaxActionSpace(), {
        lookaheadDepth: 2
      });
      
      // Test action selection
      const action = await heuristicAgent.selectAction(heuristicEnv);
      if (typeof action === 'number') {
        console.log('✅ Heuristic algorithm working');
        results.heuristicTest = true;
      }
      
      // Cleanup
      if (heuristicAgent.dispose) heuristicAgent.dispose();
    } catch (error) {
      console.log('❌ Heuristic test failed:', error.message);
    }

    // Test 6: State Encoding Consistency
    console.log('📋 Test 6: State Encoding Consistency');
    try {
      const testEnv = new EliteEnvironment();
      const testGrid = Array(9).fill(null).map(() => Array(9).fill(false));
      const testBlocks = generateRandomBlocks();
      testEnv.setState(testGrid, testBlocks, 0, 'normal');
      
      const state = testEnv.getState();
      if (state.shape[0] === testEnv.getStateSize()) {
        console.log('✅ State encoding consistent across algorithms');
        results.stateEncodingTest = true;
      }
      
      state.dispose();
    } catch (error) {
      console.log('❌ State encoding test failed:', error.message);
    }

    // Test 7: Action Decoding
    console.log('📋 Test 7: Action Decoding');
    try {
      const testEnv = new EliteEnvironment();
      const testAction = 1234;
      const decoded = testEnv.decodeAction(testAction);
      
      if (decoded.blockIndex >= 0 && decoded.row >= 0 && decoded.col >= 0) {
        console.log('✅ Action decoding works correctly');
        results.actionDecodingTest = true;
      }
    } catch (error) {
      console.log('❌ Action decoding test failed:', error.message);
    }

    // Test 8: Training Step Integration
    console.log('📋 Test 8: Training Step Integration');
    try {
      const testEnv = new EliteEnvironment();
      testEnv.reset();
      
      const validActions = testEnv.getValidActions();
      if (validActions.length > 0) {
        const action = validActions[0];
        const stepResult = testEnv.step(action);
        
        if (stepResult.state && typeof stepResult.reward === 'number') {
          console.log('✅ Training step integration working');
          results.trainingTest = true;
        }
        
        stepResult.state.dispose();
      }
    } catch (error) {
      console.log('❌ Training step test failed:', error.message);
    }

    // Test 9: Performance Benchmark
    console.log('📋 Test 9: Performance Benchmark');
    try {
      const benchmarkEnv = new EliteEnvironment();
      benchmarkEnv.reset();
      
      // Test state encoding speed
      const startTime = performance.now();
      for (let i = 0; i < 50; i++) {
        const state = benchmarkEnv.getState();
        state.dispose();
      }
      const encodingTime = performance.now() - startTime;
      
      // Test different algorithms' decision speed
      const algorithms = ['heuristic', 'mcts'];
      const decisionTimes = {};
      
      for (const algo of algorithms) {
        try {
          const agent = AlgorithmSelector.createAgent(algo, benchmarkEnv.getStateSize(), benchmarkEnv.getMaxActionSpace());
          const decisionStart = performance.now();
          
          for (let i = 0; i < 5; i++) {
            await agent.selectAction(benchmarkEnv);
          }
          
          decisionTimes[algo] = performance.now() - decisionStart;
          
          if (agent.dispose) agent.dispose();
        } catch (error) {
          console.log(`⚠️ ${algo} benchmark failed:`, error.message);
        }
      }
      
      console.log(`⏱️ Performance benchmarks:`);
      console.log(`  State encoding (50x): ${encodingTime.toFixed(2)}ms`);
      Object.entries(decisionTimes).forEach(([algo, time]) => {
        console.log(`  ${algo} decisions (5x): ${time.toFixed(2)}ms`);
      });
      
      if (encodingTime < 1000) {
        console.log('✅ Performance benchmarks acceptable');
        results.performanceTest = true;
      }
    } catch (error) {
      console.log('❌ Performance benchmark failed:', error.message);
    }

    // Summary
    const passedTests = Object.values(results).filter(Boolean).length;
    const totalTests = Object.keys(results).length;
    
    console.log(`\n🎯 Comprehensive Test Summary: ${passedTests}/${totalTests} tests passed`);
    console.log('📊 Algorithm Test Results:');
    console.log(`  🔷 Original DQN: ${results.dqnTest ? '✅' : '❌'}`);
    console.log(`  🔶 Elite DQN: ${results.eliteDqnTest ? '✅' : '❌'}`);
    console.log(`  🌳 MCTS: ${results.mctsTest ? '✅' : '❌'}`);
    console.log(`  📈 Policy Gradient: ${results.policyGradientTest ? '✅' : '❌'}`);
    console.log(`  🧠 Heuristic: ${results.heuristicTest ? '✅' : '❌'}`);
    console.log(`  🔧 System Integration: ${results.stateEncodingTest && results.actionDecodingTest && results.trainingTest ? '✅' : '❌'}`);
    console.log(`  ⚡ Performance: ${results.performanceTest ? '✅' : '❌'}`);
    
    if (passedTests === totalTests) {
      console.log('🎉 All AI algorithms are working perfectly!');
    } else if (passedTests >= totalTests * 0.8) {
      console.log('🔥 Most AI algorithms are working well!');
    } else {
      console.log('⚠️ Some AI algorithms need attention. Check the logs above.');
    }

  } catch (error) {
    console.error('🚨 Test suite error:', error);
  }

  return results;
}

export function validateGameState(grid, availableBlocks, score, difficulty) {
  const issues = [];
  
  // Check grid dimensions
  if (!grid || grid.length !== 9 || grid[0].length !== 9) {
    issues.push('Invalid grid dimensions');
  }
  
  // Check available blocks
  if (!availableBlocks || availableBlocks.length === 0) {
    issues.push('No available blocks');
  }
  
  // Check score
  if (typeof score !== 'number' || score < 0) {
    issues.push('Invalid score');
  }
  
  // Check difficulty
  if (!['normal', 'hard'].includes(difficulty)) {
    issues.push('Invalid difficulty setting');
  }
  
  return {
    isValid: issues.length === 0,
    issues: issues
  };
}

export function getAlgorithmCapabilities() {
  return {
    'dqn': {
      name: 'Original DQN',
      features: ['Neural Network', 'Experience Replay', 'Target Network'],
      supportsTraining: true,
      supportsVisualization: true,
      supportsStats: true
    },
    'elite-dqn': {
      name: 'Elite DQN',
      features: ['Double DQN', 'Dueling Architecture', 'Prioritized Replay', 'Multi-step Learning'],
      supportsTraining: true,
      supportsVisualization: true,
      supportsStats: true
    },
    'mcts': {
      name: 'Monte Carlo Tree Search',
      features: ['Tree Search', 'UCB1 Exploration', 'Rollout Policy', 'No Training Required'],
      supportsTraining: false,
      supportsVisualization: true,
      supportsStats: true
    },
    'policy-gradient': {
      name: 'Policy Gradient',
      features: ['Direct Policy Optimization', 'Action Masking', 'Entropy Regularization'],
      supportsTraining: true,
      supportsVisualization: true,
      supportsStats: true
    },
    'heuristic': {
      name: 'Hybrid Heuristic',
      features: ['Hand-crafted Rules', 'Lookahead Search', 'Instant Decisions', 'No Training Required'],
      supportsTraining: false,
      supportsVisualization: true,
      supportsStats: true
    }
  };
} 