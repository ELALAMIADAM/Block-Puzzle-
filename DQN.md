graph TD
    A[Start Episode] --> B[Environment Reset<br/>9x9 Grid Always<br/>Curriculum Block Complexity<br/>Generate 3 new blocks]
    B --> C[Get Current State<br/>139 Features Total:<br/>• 81 Grid: each cell 0/1<br/>• 27 Line Progress: row/col/square percent<br/>• 27 Block Info: 3 times 9 features each<br/>• 4 Meta: score/moves/level/timing]
    
    C --> D{Choose Action Strategy}
    
    %% Action Selection Branch
    D --> E[Adaptive Epsilon-Greedy<br/>ε decreases with curriculum level<br/>ε = base_ε times one_minus_level_times_0.2]
    D --> F[Guided Exploration 30 percent<br/>Simulate each action<br/>Count potential line clears<br/>Score = 1000 times lines_cleared]
    D --> G[Random Exploration<br/>Uniform selection from<br/>valid actions only]
    D --> H[DQN Exploitation<br/>Q-Network prediction<br/>argmax over valid actions]
    
    E --> I{Random < Adaptive ε?}
    I -->|Yes| J[Choose Exploration Type<br/>30 percent guided, 70 percent random]
    I -->|No| K[Predict Best Action<br/>Forward pass through Q-Network<br/>Architecture: 128 to 128 to 64 to action_space]
    
    J --> F
    J --> G
    
    F --> L[Evaluate Line Completion<br/>For each valid action:<br/>• Place block on test grid<br/>• Count completed rows/cols/squares<br/>• Score = lines times 1000<br/>• Return highest scoring action]
    G --> M[Random Valid Action<br/>Uniform random from<br/>getValidActions list]
    H --> K
    
    L --> N[Select Best Guided Action]
    M --> N
    K --> N
    
    %% Environment Interaction
    N --> O[Environment Step<br/>Execute selected action<br/>DQN game logic]
    O --> P[Decode Action<br/>actionId = block times 1000 + row times 10 + col<br/>Extract: blockIndex, row, col]
    P --> Q{Valid Placement?<br/>Check bounds and collisions<br/>canPlaceBlockAtPosition}
    
    Q -->|No| R[Invalid Action Penalty<br/>Reward: -100<br/>Game Over: True<br/>End episode immediately]
    Q -->|Yes| S[Place Block on Grid<br/>Set grid cells to true<br/>Remove block from available<br/>Generate new blocks if needed]
    
    S --> T[Clear Completed Lines<br/>Check all 9 rows, 9 columns<br/>Check all 9 three-by-three squares<br/>Clear = all cells filled<br/>Score += lines squared times 50 + lines times 100]
    
    T --> U[Spatial Analysis<br/>Creative Penalty System:<br/>• Count isolated cells with no neighbors<br/>• Detect dead spaces that are unfillable<br/>• Find wasted corners and edges<br/>• Measure fragmentation level]
    
    U --> V[Calculate Total Reward<br/>Constants defined once:<br/>LINE_BASE=10000, LINE_MULT=5000<br/>COMBO_MULT=15000, PLACE=10<br/>GAME_OVER=-5000, SURVIVAL=+1<br/><br/>Total = LINE_BASE + lines times LINE_MULT<br/>+ combo_bonus + spatial_penalties]
    
    V --> W[Check Game Over<br/>For each available block:<br/>Try all grid positions<br/>If none fit: game_over = true]
    
    W --> X[Generate New Blocks<br/>Based on Curriculum Level:<br/>• Level 0: 1-by-1, 2-by-1, 2-by-2 only<br/>• Level 1: plus 3-by-1, basic L-shapes<br/>• Level 2: plus T-shapes, complex L<br/>• Level 3: All 22 block types]
    
    %% Experience Storage
    R --> Y[Store Experience<br/>state, action, reward, next_state, done]
    X --> Y
    Y --> Z[Prioritized Memory Storage<br/>Line clearing experiences get<br/>MASSIVE priority 10 times normal<br/>Memory size: 5000 experiences<br/>Replace oldest non-line-clearing first]
    
    Z --> AA{Memory >= Batch Size?<br/>Batch size = 32<br/>Need sufficient experiences}
    AA -->|No| BB[Continue Episode<br/>Keep playing until<br/>enough experience collected]
    AA -->|Yes| CC[Sample Line Clearing Batch<br/>70 percent from line clearing pool<br/>30 percent from other experiences<br/>Ensures learning focus on rewards]
    
    %% Training Phase
    CC --> DD[Prepare Training Data<br/>Stack 32 states into tensor<br/>Stack 32 next_states<br/>Get Q values for states and next_states]
    
    DD --> EE[Calculate Targets<br/>For each experience:<br/>if done: target = reward<br/>else: target = reward + gamma times max Q_target<br/>γ gamma = 0.99 discount factor]
    
    EE --> FF[Train Q-Network<br/>Loss = MSE predicted_Q vs target_Q<br/>Optimizer = Adam with lr=0.001<br/>Validate tensor shapes match<br/>Architecture: 128 to 128 to 64 to action_space]
    
    FF --> GG[Update Training Metrics<br/>Store loss in history<br/>Adaptive ε decay with success bonus<br/>ε *= decay + line_clear_bonus<br/>decay=0.995, bonus=success times 0.002]
    
    %% Target Network Update
    GG --> HH{Training Step mod 50 == 0?<br/>Update frequency = 50 steps<br/>Prevents target drift}
    HH -->|Yes| II[Update Target Network<br/>Copy all weights:<br/>target_weights = main_weights<br/>Hard update and not soft]
    HH -->|No| JJ[Continue Training<br/>Keep current target weights]
    II --> JJ
    
    %% Curriculum Learning
    JJ --> KK{Episode Done?<br/>Check game_over flag}
    KK -->|No| BB
    KK -->|Yes| LL[Environment Curriculum Update<br/>Track stats at current level:<br/>episodes_count, lines_cleared<br/>avg_score = running_average]
    
    LL --> MM[Agent Curriculum Adaptation<br/>If curriculum advanced:<br/>ε *= 1.1 ; more exploration<br/>lr *= 0.95 ; slower learning<br/>Recompile network with new lr]
    
    MM --> NN[Performance Tracking<br/>Track separately:<br/>• AI internal rewards ; for training<br/>• Real game scores ; for evaluation<br/>• Line clearing success rate<br/>• Best score achieved]
    
    NN --> OO[Log Episode Stats<br/>Format: Episode X: AI_Reward=Y<br/>REAL_Score=Z, Lines=N, Level=L<br/>Success=P percent, Epsilon=E percent<br/>Clear distinction: AI vs Real metrics]
    
    OO --> PP[Next Episode<br/>Reset environment<br/>Generate new blocks]
    PP --> A
    
    %% Memory Management
    Z --> QQ[Memory Management<br/>Dispose old TensorFlow tensors<br/>Prevent memory leaks<br/>Replace by priority ; line clears last]
    QQ --> AA
    
    %% Curriculum Advancement Decision
    LL --> RR{Can Advance Curriculum?<br/>Requirements:<br/>• lines_cleared >= threshold<br/>• episodes >= 5 minimum<br/>• level < max_level which is 3}
    RR -->|Yes| SS[Advance Block Complexity<br/>Progression path:<br/>simple to medium to complex to full<br/>Reset: episodes=0, lines=0<br/>Increase threshold += 1]
    RR -->|No| MM
    SS --> TT[Generate New Curriculum Blocks<br/>Expand available block types<br/>More strategic variety<br/>Higher placement challenge]
    TT --> MM
    
    %% Environment State Components Detail
    C --> UU[Grid State Analysis<br/>81 features: 9-by-9 grid<br/>Each cell: 0=empty, 1=filled<br/>Flattened row-major order<br/>Index = row times 9 + col]
    C --> VV[Line Completion Progress<br/>27 features total:<br/>• 9 row percentages filled divided by 9<br/>• 9 column percentages filled divided by 9<br/>• 9 three-by-three square percentages filled divided by 9<br/>Values range 0.0 to 1.0]
    C --> WW[Available Blocks Encoding<br/>27 features: 3 blocks times 9 each<br/>Per block: size, width, height, density<br/>linear_potential, corner_fit, edge_fit<br/>flexibility_score, strategic_value<br/>Padded with 0s if block missing]
    C --> XX[Meta Information<br/>4 normalized features range 0.0 to 1.0:<br/>• score divided by 10000 normalized<br/>• moves_since_clear divided by 20<br/>• available_blocks divided by 3<br/>• curriculum_level divided by 3]
    
    %% Key Components Styling
    classDef actionSelection fill:#e1f5fe,stroke:#01579b,stroke-width:2px,color:#01579b
    classDef environment fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px,color:#1b5e20
    classDef training fill:#f3e5f5,stroke:#4a148c,stroke-width:2px,color:#4a148c
    classDef memory fill:#fff3e0,stroke:#e65100,stroke-width:2px,color:#e65100
    classDef curriculum fill:#f1f8e9,stroke:#33691e,stroke-width:2px,color:#33691e
    classDef decision fill:#ffecb3,stroke:#f57f17,stroke-width:2px,color:#f57f17
    classDef spatial fill:#fce4ec,stroke:#880e4f,stroke-width:2px,color:#880e4f
    
    class E,F,G,H,I,J,K,L,M,N actionSelection
    class B,O,P,Q,S,T,W,X,LL,RR,SS,TT environment
    class CC,DD,EE,FF,GG,HH,II training
    class Y,Z,QQ memory
    class MM,NN,OO curriculum
    class D,AA,KK decision
    class U,V spatial