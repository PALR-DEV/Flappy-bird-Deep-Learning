import flappy_bird_gymnasium
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from dqn_algorithm import DQNAgent

class FlappyBirdAgent:
    def __init__(self, state_size=178, action_size=2):  # Use full observation space
        self.env = gym.make("FlappyBird-v0", render_mode=None)
        self.state_size = state_size
        self.action_size = action_size
        self.agent = DQNAgent(state_size, action_size)
        self.scores = []
        self.best_score = -float('inf')
        
    def preprocess_state(self, state):
        """Convert state to feature vector"""
        # Handle both single observation and tuple from reset()
        if isinstance(state, tuple):
            observation = state[0]
        else:
            observation = state
            
        # Convert to numpy array and flatten
        if isinstance(observation, (list, tuple)):
            observation = np.array(observation, dtype=np.float32)
        elif not isinstance(observation, np.ndarray):
            observation = np.array([observation], dtype=np.float32)
            
        # Flatten and limit to state_size
        flattened = observation.flatten()
        return flattened[:self.state_size] if len(flattened) >= self.state_size else np.pad(flattened, (0, self.state_size - len(flattened)))
    
    def train(self, episodes=10000, target_update_freq=10):
        """Train the agent"""
        print("Starting training...")
        step_count = 0
        
        for episode in range(episodes):
            # Handle tuple return from reset()
            state_info = self.env.reset()
            state = self.preprocess_state(state_info)
            total_reward = 0.0
            done = False
            
            while not done:
                # Agent takes action
                action = self.agent.act(state)
                
                # Environment step
                step_result = self.env.step(action)
                next_state, reward, done = step_result[:3]
                info = step_result[3] if len(step_result) > 3 else {}
                
                # Reward shaping: small reward for staying alive
                reward = float(reward)
                if not done:
                    reward += 0.1
                
                next_state = self.preprocess_state(next_state)
                
                # Store experience
                self.agent.remember(state, action, reward, next_state, done)
                
                # Train the network every 4 steps instead of every step
                step_count += 1
                if step_count % 4 == 0:
                    self.agent.replay()
                
                state = next_state
                total_reward += reward
            
            # Update target network periodically
            if episode % target_update_freq == 0:
                self.agent.update_target_network()
            
            # Track scores
            self.scores.append(total_reward)
            
            # Update best score if improved
            if total_reward > self.best_score:
                self.best_score = total_reward
                print(f"New best score: {total_reward}!")
            
            # Save model every 100 episodes
            if episode % 1000 == 0 and episode > 0:
                self.agent.save(f'dqn_model_episode_{episode}.pt')
                print(f"Model saved at episode {episode}")

            # Print progress
            if episode % 50 == 0:
                avg_score = np.mean(self.scores[-100:]) if len(self.scores) >= 100 else np.mean(self.scores)
                print(f"Episode {episode}, Avg Score: {avg_score:.2f}, Epsilon: {self.agent.epsilon:.3f}, Best: {self.best_score}")
        
        # Save final model
        self.agent.save('final_dqn_model.pt')
        print("Training completed!")
    
    def test(self, episodes=10, render=True, model_path='final_dqn_model.pt'):
        """Test the trained agent"""
        # Create environment with render mode if needed
        if render and self.env.render_mode != 'human':
            self.env.close()
            self.env = gym.make("FlappyBird-v0", render_mode='human', use_lidar=True)
        elif not render and self.env.render_mode == 'human':
            self.env.close()
            self.env = gym.make("FlappyBird-v0", render_mode=None)
        
        # Load the model and disable exploration
        if self.agent.load(model_path):
            original_epsilon = self.agent.epsilon
            self.agent.epsilon = 0  # No exploration during testing
        
        test_scores = []
        
        for episode in range(episodes):
            state_info = self.env.reset()
            state = self.preprocess_state(state_info)
            total_reward = 0.0
            done = False
            steps = 0
            
            while not done:
                if render and steps % 2 ==0:
                    self.env.render()
                
                # Agent takes action (no exploration)
                action = self.agent.act(state)
                
                # Environment step
                step_result = self.env.step(action)
                next_state, reward, done = step_result[:3]
                
                state = self.preprocess_state(next_state)
                total_reward += float(reward)
                steps += 1
            
            test_scores.append(total_reward)
            print(f"Test Episode {episode + 1}: Score = {total_reward}, Steps = {steps}")
        
        # Restore original epsilon if model was loaded
        if 'original_epsilon' in locals():
            self.agent.epsilon = original_epsilon
        
        avg_test_score = np.mean(test_scores)
        max_score = max(test_scores)
        min_score = min(test_scores)
        
        print(f"\n=== Test Results ===")
        print(f"Episodes: {episodes}")
        print(f"Average Score: {avg_test_score:.2f}")
        print(f"Max Score: {max_score}")
        print(f"Min Score: {min_score}")
        
        return test_scores
    
    def plot_training_progress(self):
        """Plot training scores"""
        if not self.scores:
            print("No training data to plot")
            return
            
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.scores)
        plt.title('Training Scores')
        plt.xlabel('Episode')
        plt.ylabel('Score')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        # Moving average
        window = min(100, len(self.scores))
        if len(self.scores) >= window:
            moving_avg = [np.mean(self.scores[i:i+window]) for i in range(len(self.scores)-window+1)]
            plt.plot(moving_avg)
            plt.title(f'Moving Average ({window} episodes)')
            plt.xlabel('Episode')
            plt.ylabel('Average Score')
            plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def run_training_session(self):
        """Complete training session with progress tracking"""
        try:
            self.train(episodes=10000)
            self.plot_training_progress()
            print("\nTesting the trained model...")
            # self.test(episodes=5, model_path='final_dqn_model.pt')
        except KeyboardInterrupt:
            print("\nTraining interrupted. Saving current model...")
            # self.agent.save('interrupted_dqn_model.pt')
            self.plot_training_progress()

    def test_saved_model(self, model_path='final_dqn_model.pt', episodes=10, render=True):
        """Test a specific saved model"""
        return self.test(episodes=episodes, render=render, model_path=model_path)

# Usage example
if __name__ == "__main__":
    # Create agent
    agent = FlappyBirdAgent()
    
    # Option 1: Full training session
    # agent.run_training_session()
    
    # Option 2: Just test a pre-trained model
    agent.test_saved_model('dqn_model_episode_9000.pt', episodes=10, render=True)
    # agent.test_saved_model('final_dqn_mode.pt', episodes=10, render=True)
    
    # Option 3: Continue training from saved model
    # agent.agent.load('final_dqn_model.pt')
    # agent.train(episodes=1000)