```mermaid
graph TD
    A[Start MCTS Decision] --> B[Initialize Root Node<br/>Current Game State<br/>Empty children list<br/>Zero visits/value]
    B --> C[Begin Simulation Loop<br/>Target: maxSimulations times<br/>Current: 50 simulations]
    
    C --> D{Simulations < MaxLimit?<br/>Check: i < maxSimulations<br/>Early termination possible}
    D -->|Yes| E[Clone Environment<br/>Create fresh state copy<br/>Preserve original game]
    D -->|No| V[Select Best Action<br/>Highest visit count<br/>Most confident choice]
    
    E --> F[SELECTION PHASE<br/>Navigate using UCB1<br/>Start from root node]
    F --> G[Current Node = Root<br/>Begin tree traversal<br/>Track path for backprop]
    
    G --> H{Is Current Node Leaf?<br/>children.length === 0<br/>Or max depth reached?}
    H -->|No| I[Select Best Child<br/>UCB1 = exploitation + exploration<br/>UCB1 = Q + C×√(ln(N_parent)/N_child)]
    
    I --> J[Apply Child Action<br/>environment.step(action)<br/>Update game state]
    J --> K{Is Game Over?<br/>environment.gameOver<br/>Or stepResult.done?}
    K -->|Yes| P[Jump to Backpropagation<br/>Use current reward<br/>No further expansion]
    K -->|No| L[Update Current Node<br/>Move to selected child<br/>Continue tree traversal]
    L --> H
    
    H -->|Yes| M[EXPANSION PHASE<br/>Add new child nodes<br/>If not terminal state]
    M --> N{Game Over?<br/>Terminal state check<br/>No valid actions?}
    N -->|Yes| P
    N -->|No| O[Get Valid Actions<br/>environment.getValidActions()<br/>Legal moves only]
    
    O --> P1[Create Child Nodes<br/>For each valid action:<br/>• Clone game state<br/>• Create MCTSNode<br/>• Add to children list]
    P1 --> Q[Select Random Child<br/>Choose child for simulation<br/>Return for rollout phase]
    
    Q --> R[SIMULATION PHASE<br/>Monte Carlo Rollout<br/>Play to completion/limit]
    R --> S[Initialize Rollout<br/>totalReward = 0<br/>steps = 0<br/>maxSteps = 10]
    
    S --> T{Game Over OR<br/>Steps >= MaxSteps?<br/>Rollout termination}
    T -->|Yes| P
    T -->|No| U[Get Valid Actions<br/>Legal moves from current state<br/>Environment-specific generation]
    
    U --> U1{Action Selection Policy?<br/>rolloutPolicy setting<br/>Heuristic vs Random}
    U1 -->|Heuristic| U2[Heuristic Action Selection<br/>Domain knowledge:<br/>• Line completion priority<br/>• Spatial efficiency<br/>• Near-complete row bonus]
    U1 -->|Random| U3[Random Action Selection<br/>Uniform distribution<br/>Pure Monte Carlo]
    
    U2 --> W[Execute Action<br/>environment.step(action)<br/>Get reward and new state]
    U3 --> W
    W --> X[Update Rollout<br/>totalReward += stepResult.reward<br/>steps++]
    X --> Y{Rollout Done?<br/>stepResult.done<br/>Game ended?}
    Y -->|No| Z[Add New Blocks<br/>If needed:<br/>environment.generateBlocks()<br/>Continue rollout]
    Z --> T
    Y -->|Yes| P
    
    P[BACKPROPAGATION PHASE<br/>Update tree statistics<br/>Propagate rollout value]
    P --> P2[Start from Leaf Node<br/>currentNode = expanded node<br/>reward = totalReward]
    
    P2 --> P3{Current Node Exists?<br/>currentNode !== null<br/>Traverse to root}
    P3 -->|Yes| P4[Update Node Statistics<br/>visits++<br/>value += reward<br/>Q = value / visits]
    P4 --> P5[Move to Parent<br/>currentNode = parent<br/>Continue up tree]
    P5 --> P3
    
    P3 -->|No| AA[Yield Occasionally<br/>Every 10 simulations:<br/>setTimeout(resolve, 0)<br/>Prevent UI freezing]
    AA --> AB{Early Termination?<br/>Found clearly winning move?<br/>value/visits > 500}
    AB -->|Yes| V
    AB -->|No| AC[Increment Simulation<br/>i++<br/>Next iteration]
    AC --> D
    
    V --> V1[Analyze Root Children<br/>Compare visit counts<br/>Find most explored action]
    V1 --> V2[Select Best Action<br/>bestAction = child with max visits<br/>Robust selection method]
    
    V2 --> V3[Update Statistics<br/>Decision time tracking<br/>Simulation count<br/>Tree depth analysis]
    V3 --> V4[Log Decision<br/>Action selected<br/>Simulations run<br/>Decision time]
    V4 --> V5[Return Best Action<br/>Confident strategic choice<br/>Based on tree search]
    
    %% UCB1 Calculation Detail
    I --> I1[UCB1 Calculation Detail<br/>exploitation = value / visits<br/>exploration = C × √(ln(parent.visits) / visits)<br/>C = √2 ≈ 1.414]
    I1 --> I2[Balance Factors<br/>• High exploitation: Good average reward<br/>• High exploration: Under-visited nodes<br/>• Dynamic balance: Changes with tree growth]
    I2 --> J
    
    %% Heuristic Detail
    U2 --> U2A[Heuristic Scoring<br/>For each valid action:<br/>• Base random score<br/>• +50 for near-complete rows<br/>• Spatial positioning bonus<br/>• Select highest scoring]
    U2A --> W
    
    %% Performance Tracking
    V3 --> V3A[Performance Metrics<br/>• Total simulations run<br/>• Average decision time<br/>• Maximum tree depth<br/>• Node count statistics]
    V3A --> V4
    
    %% Optimization Features
    AB --> AB1[Optimization Features<br/>• Early termination for clear wins<br/>• Simulation count limits<br/>• UI yielding for responsiveness<br/>• Memory efficient state cloning]
    AB1 --> V
    
    %% Key Components Styling
    classDef selection fill:#e1f5fe,stroke:#01579b,stroke-width:2px,color:#01579b
    classDef expansion fill:#f3e5f5,stroke:#4a148c,stroke-width:2px,color:#4a148c
    classDef simulation fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px,color:#1b5e20
    classDef backprop fill:#fff3e0,stroke:#e65100,stroke-width:2px,color:#e65100
    classDef decision fill:#ffecb3,stroke:#f57f17,stroke-width:2px,color:#f57f17
    classDef optimization fill:#fce4ec,stroke:#880e4f,stroke-width:2px,color:#880e4f
    classDef ucb1 fill:#e8eaf6,stroke:#3f51b5,stroke-width:3px,color:#3f51b5
    
    class F,G,H,I,J,K,L,I1,I2 selection
    class M,N,O,P1,Q expansion
    class R,S,T,U,U1,U2,U3,W,X,Y,Z,U2A simulation
    class P,P2,P3,P4,P5 backprop
    class D,AB,AB1,V,V1,V2,V3,V4,V5 decision
    class AA,AC,V3A optimization
    class I,I1,I2 ucb1
``` 