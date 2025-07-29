# Dynamic Trust-Weighted Aggregation (DTWA)

This document describes the simplified Dynamic Trust-Weighted Aggregation (DTWA) algorithm for federated IoT security implemented in TRUST-MCNet.

## Algorithm Overview

DTWA solves key challenges in federated learning for IoT systems:
- Non-IID data distribution across devices
- Potential for poisoned or unreliable clients
- Need for continuous progress even with heterogeneous clients

## Key Components

1. **Hybrid Trust Score**
   
   Tᵢᵗ = αₜ·cosineᵢᵗ + βₜ·entropyᵢᵗ + γₜ·reputationᵢᵗ
   
   Each component measures a distinct aspect of client reliability:
   - **Cosine**: similarity between client's update Δwᵢ and population average Δw̄
   - **Entropy**: prediction uncertainty on a small public probe set
   - **Reputation**: EMA of historical accuracy deltas

2. **Bayesian-Mirror Dynamic Weights**
   
   - Computes Spearman correlations ρ between trust components and accuracy
   - Converts to evidence s=(ρ+1)/2, smooths via forgetting factor λ
   - Applies mirror-descent tilt: g=exp(ηₜ·ρ), θₜ∝θ̄⊙g
   - Self-adapts weights to emphasize the most predictive signals

3. **Dynamic Thresholding**
   
   - Sets trust cutoff τₜ to the p-th percentile of trust scores
   - Guarantees at least a minimum number of trusted clients
   - Raises selectivity as the model matures

4. **Robust Aggregation**
   
   - Computes temperature-controlled softmax weights:
     wᵢ = exp(Tᵢ/tₜ)/∑ⱼexp(Tⱼ/tₜ), tₜ=max(t₀–δ·t, t_min)
   - Applies trimmed-mean: drops extreme updates, then computes weighted mean

## Algorithm Flow

```text
Input: initial global model w⁰, probe set Dₚ, config {λ, η₀, t₀, δ, p, k}

for t = 1…T:
  1. Server broadcasts wᵗ⁻¹ to all selected clients.
  2. Each client i computes Δwᵢᵗ ← local_update(wᵗ⁻¹).
  3. Compute population average Δw̄ᵗ = (1/N)∑ᵢΔwᵢᵗ.
  4. For each client i:
       cosᵢ = cosine(Δwᵢᵗ, Δw̄ᵗ)
       entᵢ = entropy_on_probe(Dₚ; model_i)
       repᵢ = EMA_accuracy_delta(client_i)
  5. Compute dynamic weights [α,β,γ] via Bayesian-Mirror updates.
  6. Trust scores: Tᵢᵗ = α·cosᵢ + β·entᵢ + γ·repᵢ.
  7. Dynamic threshold τₜ = percentile({Tᵢᵗ}, p).
     Keep all i with Tᵢᵗ ≥ τₜ (or at least ⌈min_clients⌉).
  8. Softmax weights wᵢ ∝ exp(Tᵢᵗ / tₜ), temperature tₜ = max(t₀–δ·t, tₘᵢₙ).
  9. Aggregate by trimmed-mean:
       – Let U = set of surviving clients.
       – Drop k = ⌊trim_ratio·|U|⌋ highest & lowest updates.
       – Renormalize softmax weights, compute
         wᵗ = ∑_{i∈U\trim} wᵢ·Δwᵢᵗ
 10. Update global model wᵗ = wᵗ⁻¹ + wᵗ
 11. Update histories (model, trust, weights).
```

## Implementation Benefits

- **Continuous Progress**: Dynamic thresholds avoid "no-update" stagnation in early rounds
- **Robustness**: Combines multiple metrics and robust aggregation to resist data heterogeneity and poisoning
- **Adaptivity**: Automatically emphasizes the most predictive trust signals via Bayesian-Mirror
- **Scalability**: Linear complexity in number of clients × parameters
