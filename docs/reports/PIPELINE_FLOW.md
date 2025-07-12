# TRUST MCNet Pipeline Flow Diagram

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           TRUST MCNet Federated Learning                       │
│                              with IoT Optimization                             │
└─────────────────────────────────────────────────────────────────────────────────┘

    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
    │    Client   │    │    Client   │    │    Client   │    │    Client   │
    │      1      │    │      2      │    │      3      │    │     ...     │
    │   📱 IoT    │    │   📱 IoT    │    │   📱 IoT    │    │   📱 IoT    │
    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
           │                   │                   │                   │
           └───────────────────┼───────────────────┼───────────────────┘
                               │                   │
                               ▼                   ▼
                         ┌─────────────────────────────────┐
                         │         🌸 Flwr Server         │
                         │      (FedAdam + Trust)         │
                         └─────────────────────────────────┘
                                        │
                                        ▼
                              ┌─────────────────┐
                              │   📊 Results    │
                              │ & Visualization │
                              └─────────────────┘
```

## Detailed Pipeline Flow

### Phase 1: Initialization
```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                               INITIALIZATION                                   │
└─────────────────────────────────────────────────────────────────────────────────┘

1. Configuration Loading
   ┌─────────────┐
   │ config.yaml │ ──────► ┌─────────────────┐
   └─────────────┘         │ ConfigManager   │
                           └─────────────────┘
                                    │
                                    ▼
2. Data Preparation                 
   ┌─────────────┐         ┌─────────────────┐         ┌─────────────────┐
   │ MNIST Data  │ ──────► │ MNISTDataLoader │ ──────► │ Binary Labels   │
   └─────────────┘         └─────────────────┘         │ (Normal/Anomaly)│
                                    │                  └─────────────────┘
                                    ▼                           │
   ┌─────────────┐         ┌─────────────────┐                 │
   │Synthetic IoT│ ──────► │   Data Split    │ ◄───────────────┘
   │    Data     │         │  (IID/non-IID)  │
   └─────────────┘         └─────────────────┘
                                    │
                                    ▼
3. Model Creation
   ┌─────────────┐         ┌─────────────────┐
   │   Model     │ ──────► │ MLP/LSTM Model  │
   │ Parameters  │         │   Creation      │
   └─────────────┘         └─────────────────┘
                                    │
                                    ▼
4. Client Distribution
   ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
   │ Client Dataset  │    │ Client Dataset  │    │ Client Dataset  │
   │       1         │    │       2         │    │      ...        │
   └─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Phase 2: Federated Training Loop
```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           FEDERATED TRAINING LOOP                              │
│                              (For Each Round)                                  │
└─────────────────────────────────────────────────────────────────────────────────┘

Round 1: Random Client Selection
┌─────────────────┐
│ All Available   │ ──────► ┌─────────────────┐ ──────► ┌─────────────────┐
│    Clients      │         │ Random Selection │         │  Selected       │
└─────────────────┘         └─────────────────┘         │  Clients        │
                                                        └─────────────────┘

Round 2+: Trust-Based Selection
┌─────────────────┐         ┌─────────────────┐         ┌─────────────────┐
│ Trust Scores    │ ──────► │ Trust-Based     │ ──────► │  Selected       │
│   History       │         │   Selection     │         │  Clients        │
└─────────────────┘         └─────────────────┘         └─────────────────┘
                                                                │
                                                                ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              LOCAL TRAINING                                    │
└─────────────────────────────────────────────────────────────────────────────────┘

For Each Selected Client:
┌─────────────────┐         ┌─────────────────┐         ┌─────────────────┐
│ Local Dataset   │ ──────► │ IoT Resource    │ ──────► │ Adaptive Batch  │
└─────────────────┘         │   Monitoring    │         │     Sizing      │
                           └─────────────────┘         └─────────────────┘
                                    │                           │
                                    ▼                           ▼
┌─────────────────┐         ┌─────────────────┐         ┌─────────────────┐
│ Memory & CPU    │ ──────► │ Training Loop   │ ◄────── │ Model Updates   │
│   Management    │         │  (Local Epochs) │         └─────────────────┘
└─────────────────┘         └─────────────────┘                 │
                                    │                           │
                                    ▼                           ▼
                           ┌─────────────────┐         ┌─────────────────┐
                           │ Performance     │         │ Model Weights   │
                           │   Metrics       │         │   Upload        │
                           └─────────────────┘         └─────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                             TRUST EVALUATION                                   │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────┐         ┌─────────────────┐         ┌─────────────────┐
│ Client Model    │ ──────► │ Cosine          │ ──────► │   Cosine        │
│   Updates       │         │ Similarity      │         │    Score        │
└─────────────────┘         └─────────────────┘         └─────────────────┘
        │                                                       │
        │                   ┌─────────────────┐                 │
        └─────────────────► │ Entropy         │ ──────► ┌─────────────────┐
        │                   │ Analysis        │         │  Entropy Score  │
        │                   └─────────────────┘         └─────────────────┘
        │                                                       │
        │                   ┌─────────────────┐                 │
        └─────────────────► │ Reputation      │ ──────► ┌─────────────────┐
                           │   Scoring       │         │ Reputation      │
                           └─────────────────┘         │    Score        │
                                                      └─────────────────┘
                                                              │
                                                              ▼
                                              ┌─────────────────────────────┐
                                              │       Hybrid Trust          │
                                              │  Score = 0.4*Cosine +       │
                                              │  0.3*Entropy + 0.3*Reputation│
                                              └─────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                            MODEL AGGREGATION                                   │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────┐         ┌─────────────────┐         ┌─────────────────┐
│ Client Models   │ ──────► │ Trust-Weighted  │ ──────► │  Global Model   │
│ + Trust Scores  │         │   FedAdam       │         │    Update       │
└─────────────────┘         │  Aggregation    │         └─────────────────┘
                           └─────────────────┘                 │
                                                              ▼
                                              ┌─────────────────────────────┐
                                              │     Broadcast Updated       │
                                              │     Global Model to         │
                                              │      All Clients            │
                                              └─────────────────────────────┘
```

### Phase 3: Evaluation & Results
```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           EVALUATION & RESULTS                                 │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────┐         ┌─────────────────┐         ┌─────────────────┐
│ Global Model    │ ──────► │ Test Dataset    │ ──────► │ Performance     │
└─────────────────┘         │   Evaluation    │         │   Metrics       │
                           └─────────────────┘         └─────────────────┘
                                                               │
                                                               ▼
┌─────────────────┐         ┌─────────────────┐         ┌─────────────────┐
│ Training Logs   │ ──────► │ Results         │ ──────► │ Visualization   │
└─────────────────┘         │ Processing      │         │    Plots        │
                           └─────────────────┘         └─────────────────┘
                                    │                           │
                                    ▼                           ▼
┌─────────────────┐         ┌─────────────────┐         ┌─────────────────┐
│ Trust History   │         │ Experiment      │         │     Report      │
└─────────────────┘         │   Summary       │         │   Generation    │
                           └─────────────────┘         └─────────────────┘
```

## IoT Resource Monitoring Flow

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           IoT RESOURCE MONITORING                              │
└─────────────────────────────────────────────────────────────────────────────────┘

Continuous Monitoring (Background Thread):
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   CPU Usage     │    │ Memory Usage    │    │ Battery Level   │
│  Monitoring     │    │   Monitoring    │    │  (Optional)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
        │                       │                       │
        └───────────────────────┼───────────────────────┘
                                │
                                ▼
                    ┌─────────────────────────────┐
                    │    Resource Constraint      │
                    │       Detection            │
                    └─────────────────────────────┘
                                │
                                ▼
        ┌─────────────────────────────────────────────────────────┐
        │                Adaptive Actions                         │
        ├─────────────────┬─────────────────┬─────────────────────┤
        │ Reduce Batch    │ Skip Epochs     │ Memory Cleanup      │
        │     Size        │                 │                     │
        └─────────────────┴─────────────────┴─────────────────────┘
```

## Trust Score Evolution

```
Round 1:  All clients start with default trust (0.5)
         [Client1: 0.5] [Client2: 0.5] [Client3: 0.5] [Client4: 0.5]

Round 2:  Trust scores based on Round 1 performance
         [Client1: 0.8] [Client2: 0.3] [Client3: 0.7] [Client4: 0.6]

Round 3:  Updated trust with reputation decay
         [Client1: 0.85] [Client2: 0.2] [Client3: 0.75] [Client4: 0.65]

Round N:  Long-term trust evolution
         [High Trust Clients] ←→ [Medium Trust] ←→ [Low Trust/Excluded]
```

## Configuration-to-Execution Mapping

```
config.yaml Parameters ──────► Execution Components

model.type: "MLP"         ──────► MLP(input_dim=784, hidden_dims=[256,128,64])
federated.num_clients: 5  ──────► 5 Client instances created
federated.strategy: "FedAdam" ──► TrustAwareFedAdam strategy
iot_config.max_memory_mb  ──────► IoTResourceMonitor(max_memory=512)
trust.enabled: true       ──────► TrustEvaluator initialization
data.anomaly_digits: [1,7] ─────► MNISTDatasetWrapper(anomaly=[1,7])
```

This flow diagram provides a comprehensive view of how each component in the TRUST MCNet pipeline interacts and executes during the federated learning process.
