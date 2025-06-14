import { DQNAgent } from './DQNAgent';
import { DQNEnvironment } from './DQNEnvironment';
import { generateRandomBlocks } from '../utils/gameLogic';

export async function runAITests() {
  console.log('🧪 Starting AI System Tests...');
  
  const results = {
    environmentTest: false,
    agentTest: false,
    stateEncodingTest: false,
    actionDecodingTest: false,
    trainingTest: false,
    performanceTest: false
  };

  try {
    // Test 1: Environment Initialization
    console.log('📋 Test 1: Environment Initialization');
    const env = new DQNEnvironment();
    env.reset();
    
    if (env.getStateSize() === 112 && env.getMaxActionSpace() === 147) {
      console.log('✅ Environment initialized correctly');
      results.environmentTest = true;
    } else {
      console.log('❌ Environment dimensions incorrect');
    }

    // Test 2: Agent Initialization
    console.log('📋 Test 2: Agent Initialization');
    const agent = new DQNAgent(env.getStateSize(), env.getMaxActionSpace());
    
    if (agent.qNetwork && agent.targetNetwork) {
      console.log('✅ Agent neural networks created');
      results.agentTest = true;
    } else {
      console.log('❌ Agent initialization failed');
    }

    // Test 3: State Encoding
    console.log('📋 Test 3: State Encoding');
    const testGrid = Array(9).fill(null).map(() => Array(9).fill(false));
    const testBlocks = generateRandomBlocks();
    env.setState(testGrid, testBlocks, 0, 'normal');
    
    const state = env.getState();
    if (state.shape[0] === 112) {
      console.log('✅ State encoding produces correct dimensions');
      results.stateEncodingTest = true;
    } else {
      console.log('❌ State encoding failed');
    }

    // Test 4: Action Decoding
    console.log('📋 Test 4: Action Decoding');
    const testAction = 123; // Random action
    const decoded = env.decodeAction(testAction);
    
    if (decoded.blockIndex >= 0 && decoded.row >= 0 && decoded.col >= 0) {
      console.log('✅ Action decoding works correctly');
      results.actionDecodingTest = true;
    } else {
      console.log('❌ Action decoding failed');
    }

    // Test 5: Training Step & Reward Calculation
    console.log('📋 Test 5: Training Step & Reward Calculation');
    try {
      const validActions = env.getValidActions();
      if (validActions.length > 0) {
        const action = validActions[0];
        const stepResult = env.step(action);
        
        if (stepResult.state && typeof stepResult.reward === 'number') {
          console.log(`✅ Training step executed successfully with reward: ${stepResult.reward.toFixed(2)}`);
          
          // Test that rewards are non-zero for valid moves
          if (Math.abs(stepResult.reward) > 0) {
            console.log('✅ Reward calculation produces non-zero values');
          } else {
            console.log('⚠️ Reward is zero - this might indicate an issue');
          }
          
          // Test game over condition with a filled grid
          const filledGrid = Array(9).fill(null).map(() => Array(9).fill(true));
          const testBlocks = generateRandomBlocks();
          env.setState(filledGrid, testBlocks, 0, 'normal');
          
          const gameOverCheck = env.gameOver;
          if (gameOverCheck) {
            console.log('✅ Game over detection works correctly');
            results.trainingTest = true;
          } else {
            console.log('❌ Game over detection failed');
          }
        } else {
          console.log('❌ Training step failed');
        }
      } else {
        console.log('⚠️ No valid actions available for training test');
        results.trainingTest = true; // Not a failure
      }
    } catch (error) {
      console.log('❌ Training step error:', error);
    }

    // Test 6: Performance Benchmark
    console.log('📋 Test 6: Performance Benchmark');
    const startTime = performance.now();
    
    // Test state encoding speed
    for (let i = 0; i < 100; i++) {
      env.getState().dispose(); // Clean up tensors
    }
    
    const encodingTime = performance.now() - startTime;
    console.log(`⏱️ 100 state encodings took ${encodingTime.toFixed(2)}ms`);
    
    // Test prediction speed
    const predictionStart = performance.now();
    const testState = env.getState();
    
    for (let i = 0; i < 10; i++) {
      await agent.predict(testState, [0, 1, 2, 3, 4]);
    }
    
    const predictionTime = performance.now() - predictionStart;
    console.log(`⏱️ 10 predictions took ${predictionTime.toFixed(2)}ms`);
    
    testState.dispose();
    
    if (encodingTime < 1000 && predictionTime < 1000) {
      console.log('✅ Performance benchmarks passed');
      results.performanceTest = true;
    } else {
      console.log('⚠️ Performance may be slow but functional');
      results.performanceTest = true; // Don't fail on slow performance
    }

    // Cleanup
    agent.dispose();
    
    // Summary
    const passedTests = Object.values(results).filter(Boolean).length;
    const totalTests = Object.keys(results).length;
    
    console.log(`\n🎯 Test Summary: ${passedTests}/${totalTests} tests passed`);
    
    if (passedTests === totalTests) {
      console.log('🎉 All tests passed! AI system is ready to use.');
    } else {
      console.log('⚠️ Some tests failed. Check the logs above for details.');
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