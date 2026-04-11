# Dissertation Outline
## "Numerical Analysis of Mean-Field BSDEs: Deep Learning Solutions for Interacting Agents in Limit Order Books"

### Chapter 1: Introduction
- Market making in dealer markets
- The tacit collusion problem (CX motivation)
- BSDEs as the natural framework for stochastic control
- Contribution: deep BSDE solver validated on the exact CX model
- Outline of chapters

### Chapter 2: Mathematical Framework
- 2.1 BSDEs: Pardoux-Peng (1990), existence and uniqueness
- 2.2 BSDEs with jumps (BSDEJ): Tang-Li (1994), execution events as Poisson jumps
- 2.3 Nonlinear Feynman-Kac: connection between BSDEJ and HJB equations
- 2.4 Mean-field games: McKean-Vlasov formulation, N→∞ limit
- 2.5 The Cont-Xiong dealer market model
  - Execution probabilities (eq 6)
  - Nash equilibrium characterisation (eq 28)
  - Pareto optimum (collusion, eq 49)
  - Fictitious play (Algorithm 1)
- 2.6 The CX model as a BSDEJ (Section from bsde_connection.tex)

### Chapter 3: Deep BSDE Methods
- 3.1 The Deep BSDE paradigm (Han, Jentzen, E 2017/2018)
  - NN approximates Z-process
  - Terminal condition mismatch as loss
  - Batch normalisation discovery (undocumented in paper, critical for convergence)
- 3.2 Extension to MV-FBSDEs (Han, Hu, Long 2022)
  - Fictitious play decomposition
  - Wasserstein convergence guarantees
  - Monotonicity requirements
- 3.3 Extension to jump processes
  - NN approximates U^a, U^b (jump coefficients)
  - Optimal quotes from FOC
  - Our finite-horizon BSDEJ solver
- 3.4 Architecture considerations
  - BatchNorm erasure of broadcast features (from preprint 2)
  - FiLM conditioning vs additive (interaction test)
  - These are valid regardless of underlying model

### Chapter 4: The Neural Bellman Solver
- 4.1 Stationary formulation (infinite horizon)
  - ValueNet learns V(q) on discrete inventory grid
  - Quotes from FOC at each q
  - Connection to stationary BSDEJ
- 4.2 Validation against Algorithm 1
  - N=1 (monopolist): 0.0% error
  - N=2: 0.6% error
  - N=5,10: <1% error
  - Quote profiles match at every inventory level
- 4.3 Neural fictitious play
  - Outer loop with fixed population averages
  - Converges in 10 iterations
  - W2 distance as convergence metric
- 4.4 Q-scaling: larger inventory limits
  - Q=5 (validated), Q=10 (validated)
  - Q=20, Q=50 (from Hamilton cluster)
  - Demonstrates NN scales where grid methods slow down
- 4.5 Continuous inventory
  - NN interpolates between grid points
  - No grid discretisation needed

### Chapter 5: Nash Equilibrium and Collusion
- 5.1 Nash equilibrium structure
  - Spread narrowing with competition (N=2 vs monopolist)
  - Spread widening at large N (1/N market share effect)
  - Inventory skewing: dealers hedge by adjusting ask/bid asymmetry
- 5.2 Pareto optimum (explicit collusion)
  - Cartel uses monopolist execution probability
  - Spreads wider than Nash and monopolist
  - Inventory skewing more aggressive under collusion
- 5.3 Tacit collusion via decentralised learning
  - MADDPG implementation (CX Section 6)
  - Pre-training on monopolist solution
  - 20 independent rounds (Hamilton cluster)
  - Statistical test: 95% CI for learned spread vs Nash
  - Result: spreads above Nash → tacit collusion confirmed
- 5.4 Information structure and market design implications
  - Full information → Nash
  - No information → monopolist
  - Partial information → between (tacit collusion)
  - Regulatory implications

### Chapter 6: Conclusion
- Summary of contributions
- Limitations
  - Stylised model (not realistic LOB)
  - Discrete inventory
  - No common noise
  - MADDPG training instability
- Future work
  - Continuous-time BSDEJ solver with jump paths
  - Common noise via deep signatures (Hu et al. 2025)
  - Multi-asset extension
  - Real LOB data calibration

### Appendices
- A: Proofs (Feynman-Kac for BSDEJ)
- B: Implementation details (network architectures, hyperparameters)
- C: Hamilton cluster configuration
- D: Additional figures and animations
