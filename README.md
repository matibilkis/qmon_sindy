# Quantum SINDy: Discovering Quantum Master Equations from Measurement Data

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Latest-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-pytest-blue.svg)](tests/)

> **A quantum extension of the Sparse Identification of Nonlinear Dynamics (SINDy) algorithm** for discovering quantum master equations from continuous measurement records and time-series quantum sensing data.

> **A quantum extension of the Sparse Identification of Nonlinear Dynamics (SINDy) algorithm** for discovering quantum master equations from continuous measurement records and time-series quantum sensing data.

---

## 🎯 Overview

This repository implements a **quantum version of the SINDy algorithm** that can derive quantum master equations directly from quantum sensing data and measurement records. Unlike classical SINDy, which discovers ordinary differential equations from state trajectories, this work tackles the fundamentally more challenging problem of **inferring quantum dynamics from noisy, indirect measurements**—a critical capability for quantum metrology and quantum system identification.

### Key Innovation

The core challenge: **How can we discover the underlying quantum dynamics (master equations) using solely the measurement trace `dy_t`?** This is a quantum filtering problem where we must simultaneously:
- Track the hidden quantum state via a **quantum Kalman filter**
- Learn the **unknown external force dynamics** using trainable integration kernels
- Infer the **quantum master equation structure** from sparse measurement data

---

## 🔬 Problem Statement

### The Quantum Sensing Setup

We consider a continuously-monitored quantum harmonic oscillator subject to an unknown external force. The system evolves according to:

$$
\begin{align*}
d\vec{x} &= (A \vec{x} - \Sigma_t C^T) dt + C \Sigma_t C^T d\vec{y}_t + (0, f^{(a)}_t)^T dt \\
\frac{d\Sigma_t}{dt} &= A \Sigma_t + \Sigma_t A^T + D - \Sigma_t C C^T \Sigma_t \\
d\vec{y} &= C(\vec{x} + C^{-1} d\vec{W}_t)
\end{align*}
$$

where:
- $\vec{x} = (\langle q \rangle, \langle p \rangle)$ are the quadrature expectations
- $\Sigma_t$ is the covariance matrix of the conditional quantum state
- $d\vec{y}_t$ is the measurement record (homodyne detection)
- $f_t$ is the **unknown external force** we aim to discover

### The SINDy Challenge

Given only the noisy measurement record `dy_t`, we want to:
1. **Reconstruct the hidden state** $\vec{x}_t$ via quantum filtering
2. **Discover the force dynamics** $df/dt = \mathcal{F}(f)$ using sparse regression
3. **Infer the quantum master equation** structure from the data

This is fundamentally harder than classical SINDy because:
- Measurements are **indirect** (we observe `dy`, not `x` or `f`)
- Quantum noise is **non-commutative** and fundamentally stochastic
- The filtering and discovery problems are **coupled** and must be solved jointly

---

## 🏗️ Architecture & Technical Highlights

### Custom PyTorch Recurrent Neural Network

The core implementation uses a **physics-informed RNN** that combines:

1. **Quantum Kalman Filter Cell** (`GRNN`): Implements the stochastic master equation update
2. **Trainable Integration Kernel**: Learns the force dynamics via sparse dictionary of functions
3. **Maximum Likelihood Training**: Optimizes the likelihood of observed measurements

```python
class GRNN(torch.nn.Module):
    """
    Custom recurrent cell implementing continuous quantum measurement updates.
    
    The state update follows the stochastic master equation:
        dx = (A - ΞC)x·dt + Ξ·dy
        dΣ = (AΣ + ΣAᵀ + D - ΞΞᵀ)·dt
    
    where Ξ = ΣCᵀ is the Kalman gain coupling measurements to state updates.
    """
    
    def kernel(self, f):
        """Trainable integration kernel for force dynamics"""
        term1 = torch.squeeze(self.K1).matmul(f)      # Linear terms
        term2 = torch.squeeze(self.K2).matmul(f**2)    # Quadratic terms  
        term3 = torch.squeeze(self.K3).matmul(f*torch.flip(f, [-1]))  # Cross terms
        return term1 + term2 + term3
    
    def forward(self, dy, state, f):
        # Quantum Kalman filter update
        xicov = cov.matmul(self.C.T)
        dx = (self.A - xicov.matmul(self.C)).matmul(x)*self.dt + xicov.matmul(dy)
        
        # Learn force dynamics via trainable RK4 integration
        fnew = f + self.rk_step(f)
        
        # Update covariance matrix
        dcov = self.dt*(cov.matmul(self.A.T) + self.A.matmul(cov) + 
                       self.D - (xicov.matmul(xicov.T)))
        
        return nstate, dy_hat, fnew
```

### High-Performance Numerical Integration

- **Numba-accelerated SDE solvers**: JIT-compiled stochastic differential equation integration
- **RK4 integration**: Fourth-order Runge-Kutta for deterministic force dynamics
- **Monte Carlo trajectory generation**: Parallel simulation of quantum trajectories

### HPC & Parallel Computing

- **HTCondor integration**: Massively parallel job submission for hyperparameter sweeps
- **Multiprocessing support**: Local parallel execution for trajectory generation
- **Scalable architecture**: Designed for cluster computing with automatic resource management

---

## 📊 Results & Capabilities

### What This Code Demonstrates

✅ **Custom Deep Learning Architectures**: Physics-informed RNNs with trainable integration kernels  
✅ **Complex Optimization**: Maximum likelihood estimation with L1 regularization for sparsity  
✅ **HPC Workflows**: HTCondor job scheduling, parallel Monte Carlo simulations  
✅ **Time-Series Forecasting**: Quantum state prediction from measurement records  
✅ **Statistical Inference**: Bayesian-style parameter estimation from noisy quantum data  
✅ **Automated Experimentation**: Hyperparameter sweeps, automated result saving/loading  
✅ **Complex Modeling**: Quantum filtering, stochastic master equations, SDE integration  
✅ **Comprehensive Testing**: Full test suite with pytest covering all major components  

### Example Results

The model successfully:
- Tracks hidden quantum states from noisy measurements
- Learns sinusoidal force dynamics from measurement traces
- Handles multiple force ansatzes (linear, quadratic, cross-terms)
- Generalizes to different force types (exponential decay, oscillatory, FitzHugh-Nagumo)

---

## 🚀 Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/matibilkis/qmon_sindy.git
cd qmon_sindy

# Create virtual environment
python3 -m venv qenv
source qenv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Generate Quantum Trajectories

```bash
# Single trajectory simulation
python numerics/integration/external_forces/sin.py --itraj 1

# This generates:
# - hidden_state.npy: True quantum state trajectory
# - external_signal.npy: True external force
# - dys.npy: Noisy measurement record
```

### Train the Quantum SINDy Model

```bash
# Train model to discover force dynamics
python numerics/NN/modes/sin/in1_3.py \
    --itraj 1 \
    --printing 1 \
    --alpha 1e-16 \
    --lr 1e-3 \
    --tmp_net 0
```

### Run on HPC Cluster

```bash
# Submit parallel jobs via HTCondor
cd HPC/sinin_1_15
condor_submit condor_traj_NN.sub
```

### Running Tests

The repository includes a comprehensive test suite using pytest:

```bash
# Install test dependencies (if not already installed)
pip install -r requirements.txt

# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_models.py -v

# Run with coverage report
pytest tests/ --cov=numerics --cov-report=html
```

**Test Coverage:**
- ✅ Utility functions (parameter generation, data loading)
- ✅ Loss functions (maximum likelihood, regularization)
- ✅ Neural network models (GRNN, RecurrentNetwork)
- ✅ Quantum trajectory integration
- ✅ SDE solvers (RK4, Euler-Maruyama, Rossler)

All tests are designed to be fast and can run without requiring pre-generated data files (using temporary directories for integration tests).

---

## 📁 Repository Structure

```
qmon_sindy/
├── numerics/                  # Core Python implementation (main codebase)
│   ├── integration/          # Quantum trajectory simulation
│   │   ├── external_forces/  # Different force types (sin, exp-dec, FHN, etc.)
│   │   └── steps.py          # Numba-accelerated SDE solvers
│   ├── NN/                    # Machine learning models
│   │   ├── models/           # Custom PyTorch RNN architectures
│   │   ├── modes/            # Training scripts for different force types
│   │   └── losses.py         # Maximum likelihood loss functions
│   └── utilities/            # Data loading, plotting, parameter management
├── tests/                     # Comprehensive test suite (pytest)
│   ├── test_utilities.py     # Tests for utility functions
│   ├── test_losses.py        # Tests for loss functions
│   ├── test_models.py        # Tests for neural network models
│   ├── test_integration.py   # Tests for quantum trajectory integration
│   └── test_sde_solvers.py   # Tests for SDE solvers
├── analysis/                 # Jupyter notebooks for exploration/analysis
├── HPC/                      # HTCondor job submission scripts
├── mp_runs/                  # Multiprocessing scripts for parallel execution
├── setup.py                  # Package installation script
├── requirements.txt          # Python dependencies
├── .gitattributes            # Git language detection configuration
└── README.md                 # This file
```

**Note**: The repository is primarily Python code. Jupyter notebooks in `analysis/` are for exploration and visualization, not part of the core codebase.

---

## 🔗 Related Repositories

This project is part of a broader research program on quantum sensing and parameter estimation:

- **[qsense-continuos](https://github.com/matibilkis/qsense-continuos)**: Parameter estimation with Fisher information tracking for continuously-monitored quantum systems. Focuses on **estimating known parameters** (e.g., oscillator frequency) from measurement records.

- **[qmonsprt](https://github.com/matibilkis/qmonsprt)**: Sequential hypothesis testing for continuously-monitored quantum systems. Implements the methods from [Gasbarri et al. (2024)](https://quantum-journal.org/papers/q-2024-03-20-1289/).

### Key Differences

| Repository | Purpose | Focus |
|------------|---------|-------|
| **qmon_sindy** | **Discover quantum dynamics** | Learn unknown master equations from data (this repo) |
| **qsense-continuos** | **Estimate known parameters** | Fisher information, Cramér-Rao bounds, parameter estimation |
| **qmonsprt** | **Hypothesis testing** | Sequential testing, discrimination between quantum models |

---

## ⚠️ Current Limitations & Challenges

This is an **active research project** that addresses a fundamentally difficult problem. Current limitations include:

### Technical Challenges

1. **Non-convex Optimization**: The joint filtering + discovery problem has many local minima
2. **Sparse Dictionary Selection**: Choosing the right ansatz for force dynamics is non-trivial
3. **Measurement Noise**: Quantum noise fundamentally limits information extraction
4. **Computational Complexity**: Full quantum state tracking is expensive for long trajectories
5. **Generalization**: Models trained on one force type may not generalize to others

### Known Issues

- **Limited Force Types**: Currently tested on sinusoidal, exponential decay, and FitzHugh-Nagumo forces
- **Hyperparameter Sensitivity**: Training requires careful tuning of learning rates and regularization
- **Initialization Dependence**: Results can depend on initial parameter guesses
- **Scalability**: Full quantum SINDy for multi-mode systems is computationally intensive

### Future Directions

- [ ] Automatic dictionary selection via sparsity-promoting regularization
- [ ] Multi-mode quantum systems (coupled oscillators)
- [ ] Non-Markovian dynamics discovery
- [ ] Real experimental data integration
- [ ] Uncertainty quantification for discovered models

---

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@misc{qmon_sindy2023,
  author = {Bilkis, Matias and Gasbarri, Giulio},
  title = {Quantum SINDy: Discovering Quantum Master Equations from Measurement Data},
  year = {2023},
  howpublished = {\url{https://github.com/matibilkis/qmon_sindy}},
  note = {Presented at QTML 2023, CERN}
}
```

**Related Publications:**
- [Gasbarri et al. (2024)](https://quantum-journal.org/papers/q-2024-03-20-1289/): "Sequential hypothesis testing for continuously-monitored quantum systems" — See [qmonsprt](https://github.com/matibilkis/qmonsprt) for implementation

---

## 👥 Collaboration

This project is a collaboration with **Giulio Gasbarri** (Universitat Autònoma de Barcelona).

**Principal Investigator**: Matias Bilkis  
**Institution**: Universitat Autònoma de Barcelona  
**Research Group**: Quantum Information Theory Group

---

## 🛠️ Technologies & Skills

This repository demonstrates expertise in:

- **Deep Learning**: Custom PyTorch architectures, RNNs, physics-informed neural networks
- **Scientific Computing**: NumPy, SciPy, Numba JIT compilation
- **HPC**: HTCondor job scheduling, parallel computing, cluster management
- **Quantum Physics**: Stochastic master equations, quantum filtering, continuous measurements
- **Optimization**: Maximum likelihood estimation, gradient-based optimization, regularization
- **Time-Series Analysis**: State-space models, Kalman filtering, forecasting
- **Software Engineering**: Modular design, CLI interfaces, automated workflows

---

## 📄 License

MIT License — See [LICENSE](LICENSE) file for details.

**Academic Use**: If you use this code in research, please cite appropriately and acknowledge the authors.

---

## 🙏 Acknowledgments

- **Giulio Gasbarri** for collaboration on quantum sensing and filtering
- **John Calsamiglia** and **Elisabet Roda-Salichs** for guidance and discussions
- **QTML 2023** organizers at CERN for the opportunity to present this work

---

## 📧 Contact

For questions, collaborations, or feedback, please open an issue or contact:
- **Matias Bilkis**: [GitHub](https://github.com/matibilkis)

---

<p align="center">
  <em>Building quantum system identification tools for the next generation of quantum sensors</em>
</p>
