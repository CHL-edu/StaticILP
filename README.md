# README.md

## Project Overview

This is a Python project for implementing the Integer Linear Programming (ILP) model for Routing and Spectrum Assignment (RSA) in elastic optical networks, based on the paper "A novel integer linear programming model for routing and spectrum assignment in optical networks". The project implements a complete edge-node ILP formulation using PuLP with CPLEX solver.

## Key Files and Structure

```
ILP_RSA_Solver/
├── core/
│   ├── network_model.py          # Network topology and demand management
│   ├── ilp_formulation.py        # ILP model construction with PuLP
│   └── solution_analyzer.py      # Solution analysis and validation
├── main.py                       # Main program entry point
├── log/
└── Target.md                     # Project documentation
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

###
A simple recurrent of the article"Y. Hadhbi, H. Kerivin and A. Wagler, "A novel integer linear programming model for routing and spectrum assignment in optical networks," 2019 Federated Conference on Computer Science and Information Systems (FedCSIS), Leipzig, Germany, 2019, pp. 127-134, doi: 10.15439/2019F188. keywords: {Routing;Linear programming;Computational modeling;Integer linear programming;Wavelength division multiplexing;Optical network units}"

