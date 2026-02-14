# M3DDPG-MLA: Explainable Agentic Anti-Jamming for Social IoT

A PyTorch implementation of the Minimax Multi-Agent Deep Deterministic Policy Gradient with Multi-Head Latent Attention (M3DDPG-MLA) framework for explainable anti-jamming in Social Internet of Things (SIoT) networks.

This repository contains the code for the paper:
**"Explainable Agentic Anti-Jamming for Social IoT: A Hybrid Game-Theoretic DRL Framework"**

## Overview

M3DDPG-MLA extends the standard M3DDPG algorithm by integrating:
- **Multi-Head Latent Attention (MLA)** mechanism for cognitive perception of jamming threats
- **Action-grounded Grad-CAM** visualization for explainable decision-making
- **Hybrid game-theoretic formulation** combining Colonel Blotto Game with continuous 2D physical environment
- **Standards-based propagation model** using 3GPP TR 38.901 for realistic channel modeling

The framework enables autonomous defenders to learn robust anti-jamming strategies while providing visual explanations of their tactical decisions through attention heatmaps.

## Key Features

- **Hybrid Action Space**: Combines continuous physical maneuvering with discrete strategic resource allocation using Gumbel-Softmax relaxation
- **Explainable AI**: Grad-CAM visualizations integrated with MLA to expose agent's internal cognitive focus
- **Physical Layer Realism**: 
  - 3GPP TR 38.901 path loss model
  - Rayleigh/Rician small-scale fading
  - Doppler-induced temporal correlation
  - SINR-threshold capture rule based on standard receiver sensitivity
- **Self-play Training**: Defender and jammer co-evolve through adversarial learning
- **Modular Design**: Extensible to multi-agent and heterogeneous SIoT scenarios

## Architecture
