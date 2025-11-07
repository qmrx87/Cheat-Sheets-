# Reinforcement Learning Cheat Sheet  
**Hands-on Guide for RL in 2025**  
*Higher School Of Computer Science - Sidi Bel Abbès*  
 **DAIT DEHANE Yacine** – *November 2025*

---

## Outline

| Section | Topics |
|--------|-------|
| 1. Core Concepts | MDP, Policies, Value Functions |
| 2. Key Algorithms | Q-Learning, SARSA, DQN, PPO, SAC |
| 3. RLHF (Reinforcement Learning from Human Feedback) | Reward Modeling, PPO, DPO |
| 4. Hands-on Tools & Frameworks | Gymnasium, Stable-Baselines3, Ray RLlib, TRL |
| 5. Evaluation & Debugging | Metrics, Logging, Safety |
| 6. Real-World Applications | LLMs, Robotics, Games, Finance |
| 7. Limitations & Mitigations | Sample Inefficiency, Reward Hacking |

---

## 1. Core Concepts

| Term | Definition | Formula |
|------|----------|--------|
| **MDP** | Markov Decision Process: `(S, A, P, R, γ)` | — |
| **State (s)** | Agent’s observation of environment | — |
| **Action (a)** | Agent’s decision | — |
| **Reward (r)** | Immediate feedback | `r(s,a)` |
| **Policy π** | Strategy: `π(a|s)` | — |
| **Value Function V(s)** | Expected return from state `s` | `V^π(s) = E[Σ γ^t r_t \| s_0=s]` |
| **Q-Function Q(s,a)** | Expected return from `(s,a)` | `Q^π(s,a) = E[Σ γ^t r_t \| s_0=s, a_0=a]` |
| **Discount Factor γ** | Future reward importance | `0 ≤ γ < 1` |
| **Return G_t** | Discounted sum of rewards | `G_t = r_{t+1} + γ r_{t+2} + ...` |

> **Bellman Equation**  
> `V(s) = max_a [R(s,a) + γ Σ P(s'|s,a) V(s')]`  
> `Q(s,a) = R(s,a) + γ Σ P(s'|s,a) max_a' Q(s',a')`

---

## 2. Key Algorithms

| Algorithm | Type | Update Rule | Pros | Cons |
|---------|------|-----------|------|------|
| **Q-Learning** | Off-policy TD | `Q(s,a) ← Q(s,a) + α [r + γ max Q(s',·) - Q(s,a)]` | Simple, converges | Tabular only |
| **SARSA** | On-policy TD | `Q(s,a) ← Q(s,a) + α [r + γ Q(s',a') - Q(s,a)]` | Safe exploration | Slower |
| **DQN** | Deep Q-Network | Target: `y = r + γ max Q_target(s',·)` | Scalable | Overestimation |
| **Double DQN** | Fix overestimation | `y = r + γ Q_target(s', argmax Q(s',·))` | Stable | — |
| **PPO** | Policy Gradient | Clipped surrogate objective | Stable, sample-efficient | Hyperparameter-sensitive |
| **SAC** | Actor-Critic | Entropy-regularized | Max entropy, robust | Complex |
| **A2C/A3C** | Synchronous/Async | `∇log π(a|s) · A(s,a)` | Parallel | Variance |

### PPO Objective (Clipped)
```math
L^{CLIP} = \mathbb{E} \left[ \min(r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t) \right]
```
> `r_t(θ) = π_θ(a|s) / π_old(a|s)`

---

## 3. RLHF – Aligning LLMs with Human Values

| Stage | Goal | Method |
|------|------|--------|
| **1. Supervised Fine-Tuning (SFT)** | Base instruction following | `L = -log P(y|x)` |
| **2. Reward Modeling** | Learn human preference | Bradley-Terry: `P(y1 > y2) = σ(r(y1) - r(y2))` |
| **3. RL Fine-Tuning** | Optimize policy w/ reward | **PPO**, **DPO**, **KTO** |

### DPO (Direct Preference Optimization) – No Reward Model!
```math
L_{DPO} = - \mathbb{E}_{(x,y_w,y_l)} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} \right) \right]
```

---

## 4. Hands-on Tools & Frameworks

| Tool | Use Case | Install |
|------|--------|--------|
| **Gymnasium** | Environments | `pip install gymnasium` |
| **Stable-Baselines3** | Classic RL | `pip install stable-baselines3[extra]` |
| **Ray RLlib** | Distributed RL | `pip install "ray[rllib]"` |
| **Hugging Face TRL** | RLHF for LLMs | `pip install trl` |
| **PettingZoo** | Multi-agent | `pip install pettingzoo` |

### Train PPO with Stable-Baselines3
```python
from stable_baselines3 import PPO
import gymnasium as gym

env = gym.make("CartPole-v1")
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100_000)
model.save("ppo_cartpole")
```

### RLHF with TRL (PPO on Llama)
```python
from trl import PPOTrainer, PPOConfig
from transformers import AutoTokenizer

model_name = "meta-llama/Llama-3-8b"
tokenizer = AutoTokenizer.from_pretrained(model_name)

ppo_trainer = PPOTrainer(
    model=model,
    config=PPOConfig(batch_size=16, mini_batch_size=4),
    tokenizer=tokenizer
)

# Generate → Get reward → Update
for batch in dataloader:
    response = model.generate(batch["input_ids"])
    reward = reward_model(response)
    ppo_trainer.step(batch["input_ids"], response, reward)
```

---

## 5. Evaluation & Debugging

| Metric | Formula | Use |
|-------|--------|-----|
| **Episode Return** | `Σ r_t` | Final performance |
| **Success Rate** | `% solved` | Goal-reaching |
| **Sample Efficiency** | `Return vs. Timesteps` | Learning speed |
| **KL Divergence** | `D_KL(π_old || π_new)` | Policy drift (RLHF) |
| **Reward Model Accuracy** | `% correct rankings` | Preference alignment |

### Logging with TensorBoard
```python
from stable_baselines3.common.callbacks import BaseCallback
import tensorboard

# Auto-logged: episodic_return, episodic_length
model.learn(total_timesteps=1e5, callback=TensorBoardCallback())
```

---

## 6. Real-World Applications

| Domain | Example | Algorithm |
|-------|--------|----------|
| **LLM Alignment** | ChatGPT, Claude | **RLHF (PPO/DPO)** |
| **Robotics** | Dexterous manipulation | **SAC, PPO** |
| **Games** | AlphaStar, Dota 2 | **PPO + Self-play** |
| **Finance** | Algorithmic trading | **DQN, PPO** |
| **Healthcare** | Drug dosing | **Safe RL (CQL)** |
| **Recommendation** | YouTube, Netflix | **Contextual Bandits** |

---

## 7. Limitations & Mitigations

| Limitation | Impact | Mitigation |
|----------|--------|----------|
| **Sample Inefficiency** | Needs millions of steps | **Offline RL**, **Model-based RL**, **Replay Buffers** |
| **Reward Hacking** | Exploits loopholes | **Reward Shaping**, **Adversarial Testing** |
| **Exploration Collapse** | Gets stuck in local optima | **Entropy Bonus**, **Noisy Networks**, **Curiosity** |
| **Catastrophic Forgetting** | Loses old skills | **EWC**, **PackNet**, **Progressive Networks** |
| **Safety Risks** | Dangerous actions | **Constrained MDP**, **Shielding**, **Human-in-the-loop** |
| **Credit Assignment** | Hard in long horizons | **TD(λ)**, **GAE**, **Hindsight Experience Replay (HER)** |

---

## Quick Start Commands

```bash
# Classic RL
pip install stable-baselines3 gymnasium
python train_ppo.py

# RLHF
pip install trl transformers datasets accelerate
python rlhf_ppo.py --model llama-3-8b

# Multi-Agent
pip install pettingzoo ray[rllib]
python multi_agent_competition.py
```

---

## Resources

| Type | Link |
|------|------|
| **Book** | *Reinforcement Learning: An Introduction* – Sutton & Barto |
| **Course** | [CS294 Deep RL – UC Berkeley](http://rail.eecs.berkeley.edu/deeprlcourse/) |
| **Docs** | [Stable-Baselines3](https://stable-baselines3.readthedocs.io/), [TRL](https://huggingface.co/docs/trl) |
| **Papers** | PPO, DQN, DPO, RLHF |
| **Leaderboard** | [Open RL Benchmark](https://github.com/openrlbenchmark/openrlbenchmark) |

---

## Pro Tips

- **Start small**: `CartPole-v1` → `LunarLander` → `Custom Env`
- **Log everything**: Use Weights & Biases or TensorBoard
- **Use vectorized envs**: `DummyVecEnv`, `SubprocVecEnv`
- **For RLHF**: Use `trl` + `peft` + `bitsandbytes` for 8-bit training
- **Debug reward**: Normalize, clip, visualize

---

**Ready to align your LLM or train an agent?**  
Start with **Gymnasium + PPO** → scale to **TRL + DPO**.

---
 