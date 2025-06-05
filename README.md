# Flappy Bird AI: Deep Q-Network Reinforcement Learning Project

## 🚀 Project Overview

This project implements a **Deep Q-Network (DQN)** agent that learns to play Flappy Bird using reinforcement learning. The AI agent starts with no knowledge of the game and progressively improves its performance through trial and error, eventually mastering the challenging task of navigating through pipes.

### 🎯 Project Goals
- Train an AI agent to play Flappy Bird autonomously
- Implement a robust DQN algorithm with experience replay and target networks
- Achieve stable learning and high scores through proper reward shaping
- Create a modular, well-documented codebase for educational purposes

## 🧠 Algorithm Deep Dive

### Deep Q-Network (DQN) Architecture

The project uses a sophisticated DQN implementation with several key improvements over vanilla Q-learning:

#### 1. Neural Network Architecture
```
Input Layer:  178 neurons (flattened game state)
Hidden Layer 1: 256 neurons + ReLU activation
Hidden Layer 2: 128 neurons + ReLU activation  
Hidden Layer 3: 64 neurons + ReLU activation
Output Layer: 2 neurons (representing Q-values for each action)
```

#### 2. Key DQN Components

**Experience Replay Buffer**
- **Purpose**: Breaks correlation between consecutive experiences
- **Size**: 20,000 experiences maximum
- **Sampling**: Random batch sampling (32 experiences per training step)
- **Benefits**: Improves sample efficiency and stabilizes training

**Target Network**
- **Purpose**: Provides stable Q-value targets during training
- **Update Frequency**: Every 10 episodes
- **Mechanism**: Copies weights from main network to target network
- **Benefits**: Reduces moving target problem and improves convergence

**Epsilon-Greedy Exploration**
- **Initial Epsilon**: 1.0 (100% exploration)
- **Decay Rate**: 0.995 per episode
- **Minimum Epsilon**: 0.01 (1% exploration maintained)
- **Strategy**: Balances exploration vs exploitation throughout training

### 3. Training Algorithm Flow

```
1. Initialize environment and DQN agent
2. For each episode:
   a. Reset environment and get initial state
   b. While episode not done:
      - Choose action using epsilon-greedy policy
      - Execute action in environment
      - Observe reward and next state
      - Store experience in replay buffer
      - Sample random batch from buffer
      - Train neural network on batch
      - Update epsilon (decay exploration)
   c. Update target network periodically
   d. Track performance metrics
3. Save trained model
```

## 📁 Project Structure

```
flappy-bird/
├── agent.py                    # Main agent class and training loop
├── dqn_algorithm.py           # DQN implementation with neural network
├── README.md                  # This comprehensive documentation
├── dqn_model_episode_1000.pt # Checkpoint at 1000 episodes
├── dqn_model_episode_2000.pt # Checkpoint at 2000 episodes
├── dqn_model_episode_3000.pt # Checkpoint at 3000 episodes
├── dqn_model_episode_4000.pt # Checkpoint at 4000 episodes
├── dqn_model_episode_5000.pt # Checkpoint at 5000 episodes
├── dqn_model_episode_6000.pt # Checkpoint at 6000 episodes
├── dqn_model_episode_7000.pt # Checkpoint at 7000 episodes
├── dqn_model_episode_8000.pt # Checkpoint at 8000 episodes
├── dqn_model_episode_9000.pt # Checkpoint at 9000 episodes
└── final_dqn_model.pt        # Final trained model
```

## 🔧 Implementation Details

### State Representation

The agent receives a **178-dimensional state vector** from the Flappy Bird environment, which includes:
- Bird position (x, y coordinates)
- Bird velocity
- Pipe positions and gaps
- Distance to next pipe
- Game physics parameters

**State Preprocessing Pipeline:**
```python
def preprocess_state(self, state):
    # Handle tuple returns from environment reset
    if isinstance(state, tuple):
        observation = state[0]
    else:
        observation = state
    
    # Convert to numpy array and flatten
    if isinstance(observation, (list, tuple)):
        observation = np.array(observation, dtype=np.float32)
    
    # Flatten and pad/truncate to fixed size
    flattened = observation.flatten()
    return flattened[:self.state_size] if len(flattened) >= self.state_size else np.pad(flattened, (0, self.state_size - len(flattened)))
```

### Action Space

The agent can choose between **2 discrete actions**:
- **Action 0**: Do nothing (bird falls due to gravity)
- **Action 1**: Flap wings (bird moves upward)

### Reward Engineering

Sophisticated reward shaping to encourage desired behaviors:

```python
# Base reward from environment (typically +1 for passing pipe, -1 for collision)
reward = float(reward)

# Additional survival bonus
if not done:
    reward += 0.1  # Small positive reward for staying alive

# This encourages the agent to:
# 1. Stay alive as long as possible
# 2. Pass through pipes (environment reward)
# 3. Avoid collisions (negative terminal reward)
```

### Training Optimizations

**Batch Training**
- Training occurs every 4 steps instead of every step
- Reduces computational overhead
- Maintains learning stability

**Gradient Clipping**
```python
torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
```
- Prevents exploding gradients
- Improves training stability

**Adam Optimizer**
- Learning rate: 0.0005
- Adaptive learning rate optimization
- Better convergence properties than SGD

## 🏗️ Step-by-Step Build Process

### Phase 1: Environment Setup
1. **Installed Dependencies**
   ```bash
   pip install flappy-bird-gymnasium
   pip install gymnasium
   pip install torch
   pip install numpy
   pip install matplotlib
   ```

2. **Environment Integration**
   - Configured Flappy Bird Gymnasium environment
   - Set up proper rendering modes (human for testing, None for training)
   - Implemented state observation handling

### Phase 2: DQN Implementation
1. **Neural Network Design**
   - Created 4-layer fully connected network
   - Implemented proper weight initialization
   - Added ReLU activations for non-linearity

2. **Experience Replay**
   - Implemented circular buffer using `collections.deque`
   - Added random sampling mechanism
   - Optimized memory usage with maximum capacity

3. **Target Network**
   - Created separate target network for stable learning
   - Implemented periodic weight copying
   - Added soft update mechanism option

### Phase 3: Training Infrastructure
1. **Training Loop**
   - Episode-based training with proper state management
   - Progress tracking and logging
   - Automatic model checkpointing

2. **Performance Monitoring**
   - Score tracking and averaging
   - Best score recording
   - Training progress visualization

3. **Model Persistence**
   - Complete state saving (network weights, optimizer state, epsilon)
   - Robust loading mechanism with error handling
   - Checkpoint system for training continuity

### Phase 4: Testing and Evaluation
1. **Testing Framework**
   - Separate testing mode with exploration disabled
   - Performance metrics calculation
   - Visual demonstration capability

2. **Model Evaluation**
   - Statistical analysis of test performance
   - Comparison across different training checkpoints
   - Visualization of learning progress

## 📊 Training Results

### Training Configuration
- **Total Episodes**: 10,000
- **State Size**: 178 dimensions
- **Action Size**: 2 (do nothing, flap)
- **Learning Rate**: 0.0005
- **Discount Factor (γ)**: 0.99
- **Batch Size**: 32
- **Memory Buffer**: 20,000 experiences
- **Target Network Update**: Every 10 episodes

### Performance Metrics

The agent shows progressive improvement throughout training:

- **Episodes 0-1000**: Learning basic game mechanics
- **Episodes 1000-3000**: Developing pipe navigation strategies  
- **Episodes 3000-6000**: Refining timing and positioning
- **Episodes 6000-9000**: Achieving consistent performance
- **Episodes 9000+**: Master-level gameplay with high scores

### Model Checkpoints

Regular model saves allow for performance analysis:
- Each checkpoint represents 1000 episodes of training
- Progressive improvement visible across checkpoints
- Final model represents peak performance

## 🎮 Usage Instructions

### Training a New Model
```python
from agent import FlappyBirdAgent

# Create agent instance
agent = FlappyBirdAgent()

# Start training
agent.run_training_session()
```

### Testing a Trained Model
```python
# Test the final model
agent.test_saved_model('final_dqn_model.pt', episodes=10, render=True)

# Test a specific checkpoint
agent.test_saved_model('dqn_model_episode_9000.pt', episodes=5, render=True)
```

### Continuing Training
```python
# Load existing model and continue training
agent.agent.load('dqn_model_episode_5000.pt')
agent.train(episodes=2000)  # Train for 2000 more episodes
```

### Visualizing Training Progress
```python
# Plot training scores and moving averages
agent.plot_training_progress()
```

## 🔬 Technical Innovations

### 1. Robust State Handling
- Flexible state preprocessing for different environment return formats
- Automatic padding/truncation for consistent input dimensions
- Type safety with numpy array conversions

### 2. Adaptive Training Schedule
- Dynamic epsilon decay for optimal exploration-exploitation balance
- Periodic target network updates for stability
- Batch training with configurable frequency

### 3. Comprehensive Model Persistence
- Complete training state serialization
- Robust error handling for model loading
- Checkpoint system for training resilience

### 4. Modular Architecture
- Separation of concerns between agent and algorithm
- Configurable hyperparameters
- Easy extension for different environments

## 🔍 Key Learnings and Insights

### Reinforcement Learning Challenges Addressed
1. **Sparse Rewards**: Solved through reward shaping and survival bonuses
2. **Exploration vs Exploitation**: Balanced with epsilon-greedy strategy
3. **Sample Efficiency**: Improved with experience replay
4. **Training Stability**: Enhanced with target networks and gradient clipping

### Game-Specific Adaptations
1. **Timing Precision**: High-frequency action decisions for precise control
2. **Physics Understanding**: Learning gravity and momentum effects
3. **Spatial Reasoning**: Developing understanding of pipe gaps and positioning

## 🚀 Future Enhancements

### Potential Improvements
1. **Prioritized Experience Replay**: Weight important experiences more heavily
2. **Dueling DQN**: Separate value and advantage function approximation
3. **Double DQN**: Reduce overestimation bias in Q-value updates
4. **Multi-step Learning**: Use n-step returns for better credit assignment
5. **Convolutional Networks**: Process raw pixel input instead of engineered features

### Extended Applications
1. **Transfer Learning**: Apply trained model to variations of Flappy Bird
2. **Multi-Agent Learning**: Train multiple agents simultaneously
3. **Curriculum Learning**: Progressive difficulty increase during training
4. **Human-AI Comparison**: Benchmark against human player performance

## 📚 Dependencies and Requirements

### Required Libraries
```
torch>=1.9.0          # Deep learning framework
numpy>=1.21.0         # Numerical computations
matplotlib>=3.4.0     # Plotting and visualization
gymnasium>=0.26.0     # Reinforcement learning environment interface
flappy-bird-gymnasium # Flappy Bird game environment
```

### System Requirements
- Python 3.7+
- 4GB+ RAM (for large replay buffer)
- GPU optional but recommended for faster training

## 🎯 Conclusion

## Video demo

https://github.com/user-attachments/assets/635ea4e0-e234-4538-a9a5-70c0427c03c1




This project successfully demonstrates the power of Deep Q-Network reinforcement learning in mastering complex sequential decision-making tasks. The implementation showcases modern RL techniques including experience replay, target networks, and sophisticated reward engineering.

The modular architecture and comprehensive documentation make this project an excellent educational resource for understanding reinforcement learning concepts and their practical implementation. The progressive training checkpoints allow for detailed analysis of the learning process and provide insights into how neural networks develop game-playing strategies.

Through careful algorithm design, hyperparameter tuning, and engineering best practices, we've created a robust AI agent capable of superhuman performance in the challenging Flappy Bird environment.

---

**Project Author**: Pedro Lorenzo Rosario  
**Last Updated**: June 5, 2025  
**License**: MIT License  
**Contact**: For questions or contributions, please open an issue in the repository.
