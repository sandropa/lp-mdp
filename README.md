# Linear Programming Assignment

This repository contains a university assignment that solves a Markov Decision Process (MDP) problem using linear programming. The problem involves optimizing the actions of a robot collecting soda cans while managing its battery levels efficiently.

## Problem Overview

The robot operates in two battery states:
- **High battery**: The robot can either search for cans, wait for cans, or take other actions.
- **Low battery**: The robot can wait for cans or charge its battery.

### Objectives
Maximize the cumulative discounted reward with a discount factor \( \gamma = 0.9 \). The rewards and transitions are defined as:
- **Search for cans**: Reward = 2, but risks transitioning to the low state with probability \(1 - \alpha\).
- **Wait for cans**: Reward = 1, keeps the battery state unchanged.
- **Charge the battery**: Only available in the low state, transitions to the high state without a direct reward.

## Mathematical Formulation

The problem is formulated as a linear programming (LP) problem using Bellman optimality equations:
- Define \(v(h)\) and \(v(l)\) as the value functions for high and low battery states.
- Derive constraints based on rewards and transitions.
- Solve the LP to minimize \(v(h)\) subject to these constraints.

## Implementation

The Python implementation uses the `cvxpy` library to solve the LP formulation. It calculates:
1. Optimal value functions (\(v(h)\) and \(v(l)\)).
2. Optimal policies for each state (\(\pi_*(h)\) and \(\pi_*(l)\)).

### Key Features
- Decision variables for high and low battery states.
- Constraints derived from the Bellman equations.
- Minimization objective function.
- Policy determination based on the solution.

### Code Example
Here’s a snippet of the implementation:
```python
import cvxpy as cp

def recycling_robot(alpha, beta, r_s=2, r_w=1, gamma=0.9):
    # Decision variables
    v_h = cp.Variable(name="v_h")
    v_l = cp.Variable(name="v_l")

    # Objective
    objective = cp.Minimize(v_h)

    # Constraints
    constraints = [
        v_h >= r_w + gamma*v_h,
        v_h >= r_s + gamma*(alpha*v_h + (1 - alpha)*v_l),
        v_l >= r_w + gamma*v_l,
        v_l >= gamma*v_h,
        v_l >= beta*r_s - 3*(1 - beta) + gamma*((1 - beta)*v_h + beta*v_l)
    ]

    # Solve
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.GLPK)

    return {
        "v_h": float(v_h.value),
        "v_l": float(v_l.value),
    }

