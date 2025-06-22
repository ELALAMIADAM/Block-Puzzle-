```mermaid
graph TD
    A[Start Policy Gradient Episode] --> B[Initialize Episode<br/>Clear episode data:<br/>• episodeStates = []<br/>• episodeActions = []<br/>• episodeRewards = []<br/>episode++]
    
    B --> C[Get Current State<br/>139 features from environment<br/>Same as DQN/Elite DQN<br/>Normalized feature vector]
    
    C --> D[POLICY NETWORK FORWARD PASS<br/>Neural network inference<br/>128 → 128 → actionSize<br/>Softmax output layer]
    
    D --> E[Get Action Probabilities<br/>stateInput = state.expandDims(0)<br/>actionProbs = policyNetwork.predict(stateInput)<br/>probArray = await actionProbs.data()]
    
    E --> F[ACTION MASKING<br/>Valid actions only<br/>Prevent illegal moves<br/>Renormalize probabilities]
    
    F --> G[Mask Invalid Actions<br/>For each valid action:<br/>• validProbs[actionIndex] = probArray[actionIndex]<br/>• totalValidProb += probArray[actionIndex]<br/>Zero out invalid action probabilities]
    
    G --> H{Valid Probabilities Exist?<br/>totalValidProb > 0<br/>Network outputs meaningful?}
    
    H -->|Yes| I[Normalize Probabilities<br/>For each valid action:<br/>validProbs[i] /= totalValidProb<br/>Ensure sum = 1.0]
    
    H -->|No| J[Fallback to Uniform<br/>uniformProb = 1.0 / validActions.length<br/>Equal probability for all valid actions<br/>Graceful degradation]
    
    I --> K[STOCHASTIC ACTION SAMPLING<br/>Sample from probability distribution<br/>Natural exploration mechanism]
    K --> L[Sample Action<br/>random = Math.random()<br/>cumulative probability method<br/>Return sampled action index]
    
    J --> L
    
    L --> M[Store Episode Data<br/>episodeStates.push(state.clone())<br/>episodeActions.push(selectedActionIndex)<br/>Prepare for end-episode training]
    
    M --> N[Execute Action<br/>environment.step(selectedAction)<br/>Get reward and next state<br/>Check if episode done]
    
    N --> O[Store Reward<br/>episodeRewards.push(stepResult.reward)<br/>remember(reward) method<br/>Build reward sequence]
    
    O --> P{Episode Done?<br/>stepResult.done<br/>Game over or max steps?}
    
    P -->|No| Q[Update State<br/>state = stepResult.state<br/>Continue episode<br/>Next action selection]
    Q --> D
    
    P -->|Yes| R[END EPISODE PROCESSING<br/>Complete episode collected<br/>Ready for training<br/>Final score available]
    
    R --> S{Episode Data Available?<br/>episodeRewards.length > 0<br/>episodeStates.length > 0<br/>Valid training data?}
    
    S -->|No| T[Skip Training<br/>No data to learn from<br/>Go to performance tracking]
    S -->|Yes| U[TRAINING PHASE<br/>Policy gradient learning<br/>REINFORCE algorithm<br/>End-of-episode update]
    
    U --> V[Calculate Discounted Returns<br/>Backward computation:<br/>runningAdd = reward + γ × runningAdd<br/>G_t = R_{t+1} + γ × G_{t+1}]
    
    V --> W[Discounted Rewards Calculation<br/>discounted = []<br/>runningAdd = 0<br/>for i = T-1 down to 0:<br/>  runningAdd = γ × runningAdd + rewards[i]<br/>  discounted.unshift(runningAdd)]
    
    W --> X[Reward Normalization<br/>Variance reduction technique<br/>mean = average(discountedRewards)<br/>std = standardDeviation(discountedRewards)<br/>normalized = (rewards - mean) / std]
    
    X --> Y[POLICY GRADIENT COMPUTATION<br/>REINFORCE loss function<br/>∇θ J(θ) = E[∇θ log π(a|s) × G_t]]
    
    Y --> Z[Prepare Training Tensors<br/>statesTensor = tf.stack(episodeStates)<br/>actionsTensor = tf.tensor1d(episodeActions)<br/>rewardsTensor = tf.tensor1d(normalizedRewards)]
    
    Z --> AA[Forward Pass for Training<br/>actionProbs = policyNetwork.apply(statesTensor)<br/>Get probability distribution<br/>For gradient computation]
    
    AA --> BB[Calculate Log Probabilities<br/>oneHotActions = tf.oneHot(actionsTensor)<br/>logProbs = tf.log(tf.sum(actionProbs × oneHotActions))<br/>Log π(a_t|s_t) for selected actions]
    
    BB --> CC[Policy Gradient Loss<br/>loss = -mean(logProbs × rewardsTensor)<br/>Maximize log probability × return<br/>Negative for minimization]
    
    CC --> DD[Entropy Regularization<br/>entropy = -sum(actionProbs × log(actionProbs))<br/>entropyBonus = entropyCoeff × mean(entropy)<br/>Encourage exploration]
    
    DD --> EE[Total Loss Function<br/>totalLoss = policyLoss - entropyBonus<br/>Balance exploitation vs exploration<br/>Prevent premature convergence]
    
    EE --> FF[Gradient Computation<br/>optimizer.minimize(lossFunction)<br/>Apply gradients to policy network<br/>Update θ parameters]
    
    FF --> GG[Training Statistics<br/>Record loss value<br/>losses.push(lossValue)<br/>Track learning progress]
    
    GG --> HH[Clean Up Episode Data<br/>Dispose TensorFlow tensors<br/>episodeStates.forEach(state => state.dispose())<br/>Clear arrays for next episode]
    
    T --> II[Performance Tracking<br/>Track REAL game score<br/>scores.push(finalScore)<br/>Update best score and averages]
    HH --> II
    
    II --> JJ[Update Statistics<br/>episode++<br/>Calculate running averages<br/>Performance history tracking]
    
    JJ --> KK[Log Episode Results<br/>Episode score and performance<br/>Training status indication<br/>Best score updates]
    
    KK --> LL{Continue Training?<br/>More episodes needed?<br/>Training not stopped?}
    
    LL -->|Yes| A
    LL -->|No| MM[Training Complete<br/>Policy gradient learning finished<br/>Final performance evaluation]
    
    %% Action Sampling Detail
    L --> L1[Sampling Detail<br/>cumulative = 0<br/>for i in range(probabilities):<br/>  cumulative += probabilities[i]<br/>  if random <= cumulative:<br/>    return i]
    L1 --> M
    
    %% Discounted Returns Detail
    V --> V1[Discounting Example<br/>γ = 0.99<br/>rewards = [r1, r2, r3]<br/>returns = [r1 + γr2 + γ²r3,<br/>          r2 + γr3,<br/>          r3]]
    V1 --> W
    
    %% Normalization Detail
    X --> X1[Normalization Benefits<br/>• Reduces gradient variance<br/>• Prevents gradient explosion<br/>• Faster convergence<br/>• Stable learning]
    X1 --> Y
    
    %% Policy Gradient Theory
    Y --> Y1[Policy Gradient Theorem<br/>∇θ E[R] = E[∇θ log π(a|s) × R]<br/>• π(a|s): Policy probability<br/>• R: Cumulative return<br/>• θ: Network parameters]
    Y1 --> Z
    
    %% Network Architecture
    D --> D1[Policy Network Architecture<br/>Input: 139 features<br/>Hidden: 128 → 128 (ReLU + Dropout)<br/>Output: actionSize (Softmax)<br/>Adam optimizer (lr=0.001)]
    D1 --> E
    
    %% Entropy Detail
    DD --> DD1[Entropy Benefits<br/>• Prevents deterministic policy<br/>• Maintains exploration<br/>• Avoids local optima<br/>• Coefficient: 0.01]
    DD1 --> EE
    
    %% Key Components Styling
    classDef policy fill:#e1f5fe,stroke:#01579b,stroke-width:2px,color:#01579b
    classDef sampling fill:#f3e5f5,stroke:#4a148c,stroke-width:2px,color:#4a148c
    classDef training fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px,color:#1b5e20
    classDef gradient fill:#fff3e0,stroke:#e65100,stroke-width:2px,color:#e65100
    classDef decision fill:#ffecb3,stroke:#f57f17,stroke-width:2px,color:#f57f17
    classDef performance fill:#fce4ec,stroke:#880e4f,stroke-width:2px,color:#880e4f
    classDef theory fill:#e8eaf6,stroke:#3f51b5,stroke-width:3px,color:#3f51b5
    
    class D,E,F,G,H,I,J,D1 policy
    class K,L,M,L1 sampling
    class U,V,W,X,Y,Z,AA,BB,CC,DD,EE,FF,GG,HH,V1,X1,DD1 training
    class Y,Y1,BB,CC,DD,EE gradient
    class P,S,LL decision
    class II,JJ,KK,T performance
    class Y1,D1 theory
``` 