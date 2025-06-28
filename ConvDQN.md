```mermaid
graph TD
    A[Start Episode] --> B[Visual Environment Reset<br/>12×12 Grid Always<br/>Curriculum Block Generation<br/>4-Channel Visual State<br/>Initialize visual metrics]
    B --> C[Get Visual State<br/>4-Channel Tensor: 12×12×4<br/>• Channel 1: Grid state 0/1<br/>• Channel 2: Block placement overlay<br/>• Channel 3: Line completion potentials<br/>• Channel 4: Strategic importance map]
    
    C --> D{Choose Visual Strategy}
    
    %% Action Selection Branch
    D --> E[Adaptive Epsilon-Greedy<br/>ε based on curriculum + visual progress<br/>ε = base_ε × one_minus_level_times_0.2]
    D --> F[Pattern-Guided Exploration 30%<br/>Evaluate line completion potential<br/>Simulate placement on 12×12 grid<br/>Score = 1000 × lines_completed]
    D --> G[Random Exploration<br/>Uniform selection from<br/>valid actions for CNN diversity]
    D --> H[CNN Exploitation<br/>Convolutional network prediction<br/>Spatial pattern recognition]
    
    E --> I{Random < Adaptive ε?}
    I -->|Yes| J[Choose Exploration Type<br/>30% pattern-guided, 70% random]
    I -->|No| K[CNN Forward Pass<br/>4-channel 12×12×4 input<br/>→ Conv layers → Q-values]
    
    J --> F
    J --> G
    
    F --> L[Evaluate Line Potential<br/>For each valid action:<br/>• Place block on test grid<br/>• Count completed rows/columns<br/>• Score = lines × 1000<br/>• Return highest scoring action]
    G --> M[Random Valid Action<br/>Uniform random from<br/>getValidActions list]
    H --> K
    
    L --> N[Select Best Pattern Action]
    M --> N
    K --> N
    
    %% Environment Interaction
    N --> O[Visual Environment Step<br/>Execute selected action<br/>12×12 CNN game logic]
    O --> P[Decode Action<br/>actionId → blockIndex, row, col<br/>Max 432 possible actions<br/>row × 36 + col × 3 + blockIndex]
    P --> Q{Valid Placement?<br/>Check bounds and collisions<br/>12×12 spatial constraints}
    
    Q -->|No| R[Invalid Action Penalty<br/>Reward: -1000<br/>Game Over: True<br/>End episode immediately]
    Q -->|Yes| S[Place Block on 12×12 Grid<br/>Set grid cells to true<br/>Remove block from available<br/>Generate new curriculum blocks]
    
    S --> T[Clear Completed Lines<br/>Check all 12 rows, 12 columns<br/>Clear = all cells filled<br/>Score += lines × 100 × one_plus_lines]
    
    T --> U[Visual Intelligence Analysis<br/>Calculate spatial metrics:<br/>• Spatial efficiency (filled/total)<br/>• Visual harmony (edge/center balance)<br/>• Symmetry score (horizontal symmetry)<br/>• Territory control (strategic positions)]
    
    U --> V[Calculate Visual Reward<br/>PATTERN_BASE=20000, SPATIAL_EFFICIENCY=12000<br/>VISUAL_HARMONY=8000, SYMMETRY=3000<br/>TERRITORY_CONTROL=2500<br/><br/>Total = pattern_completion + visual_bonuses<br/>+ spatial_intelligence + curriculum_multiplier]
    
    V --> W[Check Game Over<br/>For each available block:<br/>Try all 12×12 positions<br/>If none fit: game_over = true]
    
    W --> X[Generate Curriculum Blocks<br/>Based on Visual Level:<br/>• Level 0: 1×1, 2×1, 2×2 only<br/>• Level 1: plus 3×1, basic L-shapes<br/>• Level 2: plus T-shapes, complex L<br/>• Level 3: plus advanced spatial shapes]
    
    %% Experience Storage
    R --> Y[Store Visual Experience<br/>state, action, reward, next_state, done<br/>4-channel tensor format]
    X --> Y
    Y --> Z[Visual Pattern Memory<br/>Line clearing gets MASSIVE priority<br/>Priority = 15× normal for patterns<br/>Memory size: 5000 experiences<br/>Replace lowest priority first]
    
    Z --> AA{Memory >= Batch Size?<br/>Batch size = 32<br/>Sufficient for CNN training}
    AA -->|No| BB[Continue Episode<br/>Accumulate visual experiences]
    AA -->|Yes| CC[Sample Visual Pattern Batch<br/>70% from line clearing pool<br/>30% from other experiences<br/>Ensures CNN learns high-reward patterns]
    
    %% CNN Training Phase
    CC --> DD[Prepare CNN Training Data<br/>Stack 32 visual states: [32, 12, 12, 4]<br/>Stack 32 next_states<br/>4-channel tensor management]
    
    DD --> EE[CNN Forward Pass<br/>Conv2D(32,3×3) → Conv2D(64,3×3)<br/>→ MaxPool(2×2) → Conv2D(128,3×3)<br/>→ GlobalAvgPool → Dense(256) → Dense(128)<br/>→ Output(action_space)]
    
    EE --> FF[Calculate CNN Targets<br/>For each experience:<br/>if done: target = reward<br/>else: target = reward + gamma × max Q_target<br/>γ = 0.99 discount factor]
    
    FF --> GG[Train CNN Network<br/>Loss = MSE predicted_Q vs target_Q<br/>Optimizer = Adam with lr=0.0005<br/>CNN-optimized learning rate<br/>20% dropout for regularization]
    
    GG --> HH[Update Visual Metrics<br/>Store loss in history<br/>Visual pattern success tracking<br/>ε *= decay + line_clear_bonus<br/>decay=0.996, bonus=success × 0.002]
    
    %% Target Network Update
    HH --> II{Training Step mod 100 == 0?<br/>CNN update frequency = 100 steps<br/>Extended for convergence stability}
    II -->|Yes| JJ[Update CNN Target Network<br/>Copy all weights:<br/>target_weights = main_weights<br/>Hard update for stability]
    II -->|No| KK[Continue CNN Training<br/>Keep current target weights]
    JJ --> KK
    
    %% Visual Curriculum Learning
    KK --> LL{Episode Done?<br/>Check game_over flag<br/>or max_steps reached}
    LL -->|No| BB
    LL -->|Yes| MM[Visual Curriculum Update<br/>Track patterns completed<br/>episodes at current level<br/>avg score analysis]
    
    MM --> NN[CNN Parameter Adaptation<br/>If curriculum advanced:<br/>ε *= 1.1 ; more visual exploration<br/>lr *= 0.98 ; slower CNN learning<br/>Recompile network with new lr]
    
    NN --> OO[Visual Performance Tracking<br/>Track separately:<br/>• AI internal rewards ; CNN training<br/>• Real game scores ; performance eval<br/>• Pattern recognition success rate<br/>• Spatial intelligence metrics]
    
    OO --> PP[Log Visual Episode Stats<br/>Format: Episode X: AI_Reward=Y<br/>REAL_Score=Z, Patterns=N, Level=L<br/>12×12 grid, Success=P%, Epsilon=E%<br/>Clear CNN vs game performance]
    
    PP --> QQ[Next Visual Episode<br/>Reset 12×12 environment<br/>Generate new curriculum blocks]
    QQ --> A
    
    %% Memory Management
    Z --> RR[Visual Memory Management<br/>Advanced tensor lifecycle:<br/>Dispose CNN tensors properly<br/>Prevent memory leaks<br/>Replace by visual pattern priority]
    RR --> AA
    
    %% Visual Curriculum Check
    MM --> SS{Visual Curriculum Check<br/>Advancement criteria:<br/>• patterns_completed >= threshold<br/>• episodes >= 5 minimum<br/>• level < max_level = 3}
    SS -->|Can Advance| TT[Advance Visual Curriculum<br/>Increase complexity level<br/>Expand 12×12 block variety<br/>Update pattern thresholds<br/>Reset curriculum statistics]
    SS -->|Stay Current| NN
    TT --> UU[Generate Advanced Blocks<br/>Higher complexity shapes<br/>12×12 strategic challenges<br/>Enhanced spatial variety<br/>Curriculum-appropriate difficulty]
    UU --> NN
    
    %% Visual State Components Detail
    C --> VV[Channel 1: Grid State<br/>144 features: 12×12 grid<br/>Each cell: 0=empty, 1=filled<br/>Spatial pattern representation<br/>CNN-optimized format]
    C --> WW[Channel 2: Block Overlay<br/>144 features: placement potentials<br/>Shows where blocks can go<br/>Intensity = placement probability<br/>Helps CNN learn spatial patterns]
    C --> XX[Channel 3: Line Potentials<br/>144 features: completion chances<br/>Row/column completion mapping<br/>Guides toward high-reward moves<br/>Strategic spatial guidance]
    C --> YY[Channel 4: Strategic Map<br/>144 features: importance values<br/>• Corners: 0.9 strategic value<br/>• Edges: 0.6 strategic value<br/>• Center: 0.7 strategic value<br/>• Territory control guidance]
    
    %% CNN Architecture Details
    EE --> ZZ[CNN Layer 1<br/>Conv2D: 32 filters, 3×3 kernel<br/>ReLU activation, He initialization<br/>Input: [batch, 12, 12, 4]<br/>Output: [batch, 12, 12, 32]]
    EE --> AAA[CNN Layer 2<br/>Conv2D: 64 filters, 3×3 kernel<br/>ReLU activation, He initialization<br/>Input: [batch, 12, 12, 32]<br/>Output: [batch, 12, 12, 64]]
    EE --> BBB[CNN Layer 3<br/>MaxPool2D: 2×2 pooling<br/>Spatial dimension reduction<br/>Input: [batch, 12, 12, 64]<br/>Output: [batch, 6, 6, 64]]
    EE --> CCC[CNN Layer 4<br/>Conv2D: 128 filters, 3×3 kernel<br/>ReLU activation, spatial patterns<br/>Input: [batch, 6, 6, 64]<br/>Output: [batch, 6, 6, 128]]
    EE --> DDD[CNN Layer 5<br/>GlobalAveragePooling2D<br/>Spatial intelligence summary<br/>Input: [batch, 6, 6, 128]<br/>Output: [batch, 128]]
    
    %% Dense Layers Detail
    EE --> EEE[Dense Layer 1<br/>256 units, ReLU activation<br/>20% dropout regularization<br/>Spatial feature processing<br/>Input: 128 → Output: 256]
    EE --> FFF[Dense Layer 2<br/>128 units, ReLU activation<br/>20% dropout regularization<br/>Decision layer processing<br/>Input: 256 → Output: 128]
    EE --> GGG[Output Layer<br/>action_space units, linear<br/>Q-value predictions<br/>Input: 128 → Output: actions<br/>CNN spatial intelligence]
    
    %% Visual Intelligence Analysis
    U --> HHH[Spatial Efficiency<br/>Calculate filled_cells / total_cells<br/>Measure space utilization<br/>Range: 0.0 to 1.0<br/>Optimal: > 0.7]
    U --> III[Visual Harmony<br/>Balance edge vs center occupation<br/>min(edge, center) / max(edge, center+1)<br/>Range: 0.0 to 1.0<br/>Good: > 0.6]
    U --> JJJ[Symmetry Score<br/>Horizontal symmetry measurement<br/>Compare left vs right halves<br/>Range: 0.0 to 1.0<br/>Bonus: > 0.5]
    U --> KKK[Territory Control<br/>Strategic position control<br/>corners + edges + center / filled<br/>Range: 0.0 to 1.0<br/>Strong: > 0.4]
    
    %% Reward Calculation Details
    V --> LLL[Pattern Completion<br/>Base: 20000 for any line clear<br/>Line bonus: lines × 5000<br/>Combo bonus: lines² × 8000<br/>Massive rewards for patterns]
    V --> MMM[Visual Intelligence Bonuses<br/>Spatial efficiency × 12000<br/>Visual harmony × 8000<br/>Symmetry × 3000<br/>Territory control × 2500]
    V --> NNN[Standard Rewards<br/>Placement: cells × 15 points<br/>Survival: +2 per step<br/>Game over: -8000 penalty<br/>Curriculum: × (1 + level × 0.2)]
    
    %% Key Components Styling
    classDef actionSelection fill:#e1f5fe,stroke:#01579b,stroke-width:2px,color:#01579b
    classDef training fill:#f3e5f5,stroke:#4a148c,stroke-width:2px,color:#4a148c
    classDef memory fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px,color:#1b5e20
    classDef metrics fill:#fff3e0,stroke:#e65100,stroke-width:2px,color:#e65100
    classDef decision fill:#ffecb3,stroke:#f57f17,stroke-width:2px,color:#f57f17
    classDef environment fill:#f1f8e9,stroke:#33691e,stroke-width:2px,color:#33691e
    classDef visual fill:#fce4ec,stroke:#880e4f,stroke-width:2px,color:#880e4f
    classDef cnn fill:#e8eaf6,stroke:#3f51b5,stroke-width:3px,color:#3f51b5
    
    class E,F,G,H,I,J,K,L,M,N actionSelection
    class CC,DD,EE,FF,GG,HH,II,JJ training
    class Z,RR memory
    class MM,NN,OO,PP metrics
    class D,AA,LL decision
    class B,O,P,Q,S,T,W,X,SS,TT,UU environment
    class U,V,HHH,III,JJJ,KKK,LLL,MMM,NNN visual
    class A,C,Y,QQ,ZZ,AAA,BBB,CCC,DDD,EEE,FFF,GGG cnn
``` 