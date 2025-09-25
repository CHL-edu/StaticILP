# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python project for implementing the Integer Linear Programming (ILP) model for Routing and Spectrum Assignment (RSA) in elastic optical networks, based on the paper "A novel integer linear programming model for routing and spectrum assignment in optical networks". The project implements a complete edge-node ILP formulation using PuLP with CPLEX solver.

## Key Files and Structure

```
ILP_RSA_Solver/
├── core/
│   ├── network_model.py          # Network topology and demand management
│   ├── ilp_formulation.py        # ILP model construction with PuLP
│   ├── constraint_generator.py   # Constraint generation for routing and spectrum
│   └── solution_analyzer.py      # Solution analysis and validation
├── algorithms/
│   ├── branch_and_cut.py         # Branch-and-Cut implementation (future)
│   ├── separation_oracles.py     # Separation oracles for min-cut (future)
│   └── heuristics.py             # Heuristic algorithms (future)
├── utils/
│   ├── data_loader.py            # Data loading from JSON/topology files
│   ├── result_visualizer.py      # Result visualization with English labels
│   └── performance_metrics.py   # Performance evaluation metrics
├── tests/
│   ├── test_small_networks.py    # Small network validation tests
│   └── validation_cases.py       # Validation test cases
├── main.py                       # Main program entry point
├── topology_6nodes.py            # Original 6-node topology generator
└── log/topology_6nodes.json      # Generated topology data
```

## Architecture and Components

### ILP Model Implementation
The project implements the complete ILP formulation from the paper:

**Decision Variables:**
- `x_e_k`: Binary routing variables (1 if demand k uses edge e)
- `z_s_k`: Binary spectrum assignment variables (1 if slot s is the last slot for demand k)
- `t_e_s_k`: Binary edge-slot assignment variables
- `a_e`: Binary edge activation variables (for O2 objective)
- `p_e`, `p`: Max slot position variables (for O3 objective)

**Constraints:**
- Routing constraints (origin, destination, path continuity, degree, cycle elimination)
- Spectrum assignment constraints (continuity, contiguity, non-overlapping)
- Transmission reach constraints
- Edge activation constraints (for O2)
- Max slot position constraints (for O3)

**Objective Functions (Priority Order):**
1. **O3**: Minimize maximum slot position (primary objective)
2. **O1**: Minimize total hop count (secondary objective, if O3 tied)

### Network Topology System
- **Network Creation**: 6-node test topology with specific edge distances
- **Spectrum Management**: Each edge has 10 spectrum slots for allocation
- **Demand Management**: Source-destination pairs with slot requirements and reach constraints
- **Validation**: Network connectivity and solution feasibility checking

## Common Commands

### Running the ILP Solver
```bash
python main.py
```

### Running Tests
```bash
python tests/test_small_networks.py
python tests/validation_cases.py
```

### Dependencies
The project requires:
- pulp (ILP modeling)
- networkx (graph operations)
- matplotlib (visualization)
- json, time, os (built-in)

## Development Notes

### Solver Configuration
- Primary solver: CBC (included with PuLP, no additional installation required)
- For small-scale validation (6-node network), complete constraint enumeration is used
- Branch-and-Cut method planned for larger networks

### Implementation Strategy
1. **Phase 1**: Basic ILP model with complete constraint enumeration
2. **Phase 2**: Branch-and-Cut implementation with separation oracles
3. **Phase 3**: Performance optimization and larger network testing

### Key Features
- Modular design with high cohesion and low coupling
- Console-based interface (preference over GUI)
- English labels for charts and visualizations
- Comprehensive testing and validation
- Clean separation of concerns between network modeling, ILP formulation, and result analysis

## Paper Implementation Details

### Problem Definition
- **Input**: Network G=(V,E), spectrum slots S, demand set K with source/destination, slot requirements, and reach constraints
- **Output**: Routing paths and spectrum assignments satisfying continuity, contiguity, and non-overlapping constraints
- **Objective**: Multi-objective optimization with O3 > O1 > O2 priority

### Technical Approach
- Uses NetworkX for graph operations and path calculations
- Implements complete ILP formulation using PuLP
- For 6-node network: uses complete constraint enumeration
- Supports multiple optimization objectives with priority ordering
- Includes solution validation and performance metrics
- Target.md,#claude.md
- Target.md#claude.md
- Target.md#claude.md
- Target.md,#claude.md