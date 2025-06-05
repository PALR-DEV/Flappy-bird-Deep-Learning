import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque


class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.0005, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.memory = deque(maxlen=20000)  # Increased memory size
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        #Neural network 
        self.q_network = self.build_model()
        self.target_network = self.build_model()
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)

    def build_model(self):
        model = nn.Sequential(
            nn.Linear(self.state_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_size)
        )
        return model
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self,state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state_tensor)
        return np.argmax(q_values.cpu().data.numpy())
    
    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        
        # Convert to tensors more efficiently
        states = torch.FloatTensor(np.array([e[0] for e in batch]))
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor(np.array([e[3] for e in batch]))
        dones = torch.BoolTensor([e[4] for e in batch])

        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)

        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        # Add gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    

    def save(self, filepath='dqn_model.pt'):
        """Save the model and training state"""
        checkpoint = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'state_size': self.state_size,
            'action_size': self.action_size,
            'learning_rate': self.learning_rate,
            'gamma': self.gamma
        }
        torch.save(checkpoint, filepath)
        print(f"Model saved to {filepath}")

    def load(self, filepath='dqn_model.pt'):
        """Load the model and training state"""
        try:
            checkpoint = torch.load(filepath)
            self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
            self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            print(f"Model loaded from {filepath}")
            return True
        except FileNotFoundError:
            print(f"No saved model found at {filepath}")
            return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    
    def test_model(self, env, episodes = 10, filepath='dqn_model.pt', render=True):
        """Test the trained model"""
        if not self.load(filepath):
            print("Failed to load model.")
            return []
        
        original_epsilon = self.epsilon
        self.epsilon = 0.0  # No exploration during testing

        test_scores = []
        print("Testing the model...")

        for episode in range(episodes):
            state_info = env.reset()
            if isinstance(state_info, tuple):
                state = state_info[0]
            else:
                state = state_info

            if not isinstance(state, np.ndarray):
                state = np.array(state)

            total_reward = 0
            done = False
            step_count = 0

            while not done:
                if render:
                    env.render()

                action = self.act(state)

                step_result = env.step(action)
                next_state, reward, done = step_result[:3]

                if not isinstance(next_state, np.ndarray):
                    next_state = np.array(next_state)

                state = next_state
                total_reward += reward
                step_count += 1

            test_scores.append(total_reward)
            print(f"Test Episode {episode + 1}: Score = {total_reward}, Steps = {step_count}")
        
        self.epsilon = original_epsilon  # Restore epsilon after testing
        avg_test_score = np.mean(test_scores)
        max_score = np.max(test_scores)
        min_score = np.min(test_scores)

        print(f"\n=== Test Results ===")
        print(f"Episodes: {episodes}")
        print(f"Average Score: {avg_test_score:.2f}")
        print(f"Max Score: {max_score}")
        print(f"Min Score: {min_score}")
        print(f"Model: {filepath}")

        return test_scores
    
