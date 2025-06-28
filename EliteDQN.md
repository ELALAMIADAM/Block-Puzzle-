graph TD
    A[Start Episode] --> B[Elite Environment Reset<br/>9x9 Grid Always<br/>Elite Block Generation<br/>Advanced Performance Metrics<br/>Initialize spatial tracking]
    B --> C[Get Elite State<br/>139 Features Total:<br/>• 81 Grid: each cell 0/1<br/>• 27 Line Intelligence: advanced analysis<br/>• 27 Block Intelligence: 3 times 9 strategic<br/>• 4 Meta Intelligence: normalized metrics]
    
    C --> D{Choose Action Strategy}
    
    %% Action Selection Branch
    D --> E[Noisy Network 10 percent<br/>Built-in exploration noise<br/>No manual epsilon needed]
    D --> F[Adaptive Epsilon-Greedy<br/>ε based on curriculum + performance<br/>ε = base_ε times one_minus_level_times_0.2]
    D --> G[Strategic Action<br/>Line Completion Bias<br/>Evaluate spatial efficiency<br/>Chain reaction potential]
    D --> H[Curiosity-Driven Exploration<br/>Novelty score calculation<br/>Softmax action selection<br/>Encourages state diversity]
    
    E --> I[Select Noisy Action<br/>Network outputs include noise<br/>Automatic exploration]
    F --> J{Random < Adaptive ε?}
    J -->|Yes| K[Random Exploration<br/>Uniform from valid actions]
    J -->|No| L[Double DQN Exploitation<br/>Main net selects action<br/>Target net evaluates Q-value]
    
    G --> M[Evaluate Action Strategy<br/>For each valid action:<br/>• Line completion potential<br/>• Chain reaction opportunities<br/>• Spatial efficiency score<br/>• Strategic value up to 1000 times reward]
    H --> N[Calculate Novelty Score<br/>State visitation frequency<br/>Intrinsic motivation bonus<br/>Softmax selection probability]
    
    I --> O[Execute Action]
    K --> O
    L --> O
    M --> O
    N --> O
    
    %% Elite Environment Interaction
    O --> P[Elite Environment Step<br/>Execute selected action<br/>Advanced game logic]
    P --> Q[Decode Elite Action<br/>actionId to blockIndex, row, col<br/>Max 243 possible actions<br/>action = row times 81 + col times 3 + block]
    Q --> R{Valid Elite Placement?<br/>Advanced collision detection<br/>Spatial constraint checking}
    
    R -->|No| S[Invalid Action Penalty<br/>Reward: -1000<br/>Game Over: True<br/>Heavy discouragement]
    R -->|Yes| T[Place Elite Block<br/>Update grid state<br/>Track spatial changes<br/>Update performance metrics<br/>Remove used block]
    
    T --> U[Elite Line Clearing<br/>Check all 9 rows, 9 columns<br/>Advanced chain detection<br/>Track combo multipliers<br/>Sequential clear bonuses]
    
    U --> V[Advanced Spatial Analysis<br/>Elite Intelligence System:<br/>• Chain potential calculation<br/>• Spatial efficiency analysis<br/>• Connectivity measurement<br/>• Fragmentation detection<br/>• Dead space identification]
    
    V --> W[Calculate Elite Reward<br/>Constants defined once:<br/>LINE_BASE=15000, LINE_MULT=8000<br/>COMBO_MULT=25000, MAX_COMBO=40000<br/>NEAR_COMPLETE=1500, CHAIN_SETUP=300<br/><br/>Total = LINE_BASE + lines times LINE_MULT<br/>+ exponential_combo_bonus<br/>+ spatial_intelligence_bonuses]
    
    W --> X[Elite Game Over Check<br/>Comprehensive placement test<br/>All blocks vs all positions<br/>Advanced constraint checking]
    
    X --> Y[Generate Elite Blocks<br/>Sophisticated variety algorithm<br/>Strategic depth consideration<br/>Curriculum complexity scaling<br/>Ensure shape diversity]
    
    %% Experience Storage with Elite Priority
    S --> Z[Store Elite Experience<br/>state, action, reward, next_state, done<br/>Calculate initial priority]
    Y --> Z
    Z --> AA[Elite Prioritized Memory<br/>MASSIVE priority for line clearing<br/>Priority = TD_error + ε_priority<br/>High priority for terminal states<br/>Replace lowest priority experiences<br/>Memory size: 5000]
    
    AA --> BB{Memory >= Batch Size?<br/>Batch size = 32<br/>Sufficient for stable training}
    BB -->|No| CC[Continue Episode<br/>Accumulate more experience]
    BB -->|Yes| DD[Sample Elite Prioritized Batch<br/>Use priority^α for sampling<br/>α = 0.6 prioritization exponent<br/>Calculate importance weights<br/>β = 0.4 to 1.0 annealing schedule]
    
    %% Elite Training Phase
    DD --> EE[Prepare Elite Training Data<br/>Stack 32 states into tensors<br/>Stack 32 next_states<br/>Get Q-values from both networks<br/>Advanced tensor management]
    
    EE --> FF[Double DQN Target Calculation<br/>Two-step process:<br/>1. Main network: a* = argmax Q_main<br/>2. Target network: Q_target at s_prime and a*<br/>3. target = r + gamma times Q_target<br/>gamma = 0.99 discount factor]
    
    FF --> GG[Calculate TD Errors<br/>TD_error = absolute value of target minus predicted<br/>Absolute value for priority update<br/>Elite error measurement precision]
    GG --> HH[Update Experience Priorities<br/>new_priority = TD_error + ε_priority<br/>ε_priority = 0.01 minimum priority<br/>Maintain elite priority structure]
    HH --> II[Apply Importance Sample Weights<br/>weight = N times prob to power negative beta<br/>β increases 0.4 to 1.0 over training<br/>Corrects sampling bias]
    
    %% Elite Network Training
    II --> JJ[Train Elite Q-Network<br/>Loss = MSE with importance weights<br/>Optimizer = Adam with lr=0.001<br/>Progressive architecture:<br/>128 to 128 to 64 to action_space<br/>Advanced regularization: dropout=0.2]
    JJ --> KK[Update Elite Training Metrics<br/>Store loss in history<br/>Track training steps<br/>Performance analytics<br/>Convergence monitoring]
    
    %% Elite Target Network Update
    KK --> LL{Training Step mod 50 == 0?<br/>Update frequency = 50 steps<br/>Prevents target drift}
    LL -->|Yes| MM[Soft Update Target Network<br/>θ_target = tau times θ_main + one_minus_tau times θ_target<br/>τ = 0.001 soft update rate<br/>Elite precision update]
    LL -->|No| NN[Continue Training<br/>Maintain current targets]
    MM --> NN
    
    %% Elite Hyperparameter Updates
    NN --> OO[Update Elite Exploration<br/>Intelligent ε decay:<br/>ε *= decay + success_bonus<br/>decay=0.995, bonus=line_success times 0.002<br/>β annealing: β += 0.001<br/>Success-based adjustments]
    
    OO --> PP{Episode Done?<br/>Check game_over flag<br/>or max_steps reached}
    PP -->|No| CC
    PP -->|Yes| QQ[Elite Episode End Processing<br/>Comprehensive analysis phase]
    
    %% Elite Episode End Processing
    QQ --> RR[Track Elite Performance<br/>Dual metric tracking:<br/>• Real game score ; actual gameplay<br/>• AI reward ; training signal<br/>• Best score tracking<br/>• Performance history analysis<br/>• Spatial intelligence metrics]
    
    RR --> SS[Elite Curriculum Learning<br/>Performance-based advancement:<br/>• Line clearing mastery<br/>• Score improvement trends<br/>• Adjust learning rate dynamically<br/>• Modify exploration strategy<br/>• Update complexity levels]
    
    SS --> TT[Adaptive Elite Parameters<br/>Dynamic hyperparameter tuning:<br/>• Reduce ε if performance good<br/>• Increase ε if stuck/plateau<br/>• Learning rate adjustment<br/>• Fine-tune reward weights<br/>• Network architecture tweaks]
    
    TT --> UU[Log Elite Statistics<br/>Comprehensive episode summary:<br/>Episode X: AI_Reward=Y, REAL_Score=Z<br/>Lines=N, Chains=C, Level=L<br/>Spatial_Efficiency=E percent, Success=S percent<br/>Epsilon=ε percent, Chain_Length=CL]
    
    UU --> VV[Next Elite Episode<br/>Environment reset<br/>Generate new elite blocks]
    VV --> A
    
    %% Elite Memory Management
    AA --> WW[Elite Memory Management<br/>Advanced tensor lifecycle:<br/>Dispose old TensorFlow tensors<br/>Prevent memory leaks<br/>Replace lowest priority experiences<br/>Maintain line clearing priority]
    WW --> BB
    
    %% Elite Environment State Components Detail
    C --> XX[Elite Grid Analysis<br/>81 features: 9-by-9 full grid state<br/>Each cell: 0=empty, 1=filled<br/>Advanced spatial representation<br/>Row-major flattened indexing]
    C --> YY[Elite Line Intelligence<br/>27 features: completion analysis<br/>• 9 row percentages filled divided by 9<br/>• 9 column percentages filled divided by 9<br/>• 9 spatial patterns:<br/>  - near_complete_lines divided by 18<br/>  - chain_potential range 0 to 1<br/>  - spatial_efficiency range 0 to 1<br/>  - connectivity range 0 to 1<br/>  - fragmentation range 0 to 1<br/>  - corner_utilization divided by 4<br/>  - edge_utilization ratio<br/>  - dead_space_ratio range 0 to 1<br/>  - pattern_formation_score]
    C --> ZZ[Elite Block Intelligence<br/>27 features: 3 blocks times 9 each<br/>Per block detailed analysis:<br/>• size divided by 9 normalized<br/>• width divided by 3, height divided by 3<br/>• density range 0 to 1<br/>• linear_potential range 0 to 1<br/>• corner_fit range 0 to 1<br/>• edge_fit range 0 to 1<br/>• flexibility_score range 0 to 1<br/>• strategic_value range 0 to 1<br/>Padded with 0s if missing]
    C --> AAA[Elite Meta Intelligence<br/>4 normalized features range 0 to 1:<br/>• score divided by 10000 for performance<br/>• moves_since_clear divided by 20 for urgency<br/>• curriculum_level divided by 3 for difficulty<br/>• chain_length divided by 5 for combo state]
    
    %% Elite Spatial Analysis Details
    V --> BBB[Chain Potential Analysis<br/>Multi-line opportunity detection:<br/>• Identify potential chain reactions<br/>• Calculate sequential clear probability<br/>• Measure combo setup potential<br/>• Evaluate cascade effects<br/>Score range: 0 to 1 normalized]
    V --> CCC[Spatial Efficiency Metrics<br/>Advanced spatial intelligence:<br/>• Connectivity measurement<br/>• Fragmentation detection<br/>• Dead space identification<br/>• Corner & edge utilization<br/>• Compactness optimization<br/>• Pattern recognition]
    V --> DDD[Strategic Pattern Recognition<br/>Elite pattern analysis:<br/>• Pattern formation detection<br/>• Future opportunity creation<br/>• Isolation prevention strategies<br/>• Compactness optimization<br/>• Long-term spatial planning<br/>• Multi-move ahead thinking]
    
    %% Elite Curriculum Management
    SS --> EEE{Elite Curriculum Check<br/>Advancement criteria:<br/>• mastery_score >= threshold<br/>• consistency over episodes<br/>• line_clearing_rate >= target}
    EEE -->|Can Advance| FFF[Advance Elite Curriculum<br/>Increase complexity level<br/>Expand block variety pool<br/>Update performance thresholds<br/>Reset curriculum statistics<br/>Enhance challenge difficulty]
    EEE -->|Stay Current| TT
    FFF --> GGG[Generate Advanced Blocks<br/>Higher complexity shapes<br/>Strategic challenges<br/>Enhanced variety algorithms<br/>Curriculum-appropriate difficulty]
    GGG --> TT
    
    %% Key Components Styling
    classDef actionSelection fill:#e1f5fe,stroke:#01579b,stroke-width:2px,color:#01579b
    classDef training fill:#f3e5f5,stroke:#4a148c,stroke-width:2px,color:#4a148c
    classDef memory fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px,color:#1b5e20
    classDef metrics fill:#fff3e0,stroke:#e65100,stroke-width:2px,color:#e65100
    classDef decision fill:#ffecb3,stroke:#f57f17,stroke-width:2px,color:#f57f17
    classDef environment fill:#f1f8e9,stroke:#33691e,stroke-width:2px,color:#33691e
    classDef spatial fill:#fce4ec,stroke:#880e4f,stroke-width:2px,color:#880e4f
    classDef elite fill:#e8eaf6,stroke:#3f51b5,stroke-width:3px,color:#3f51b5
    
    class E,F,G,H,I,J,K,L,M,N actionSelection
    class DD,EE,FF,GG,HH,II,JJ,KK,LL,MM training
    class AA,WW memory
    class QQ,RR,SS,TT,UU metrics
    class D,BB,PP decision
    class B,P,Q,R,T,U,X,Y,EEE,FFF,GGG environment
    class V,BBB,CCC,DDD spatial
    class A,C,O,Z,VV elite