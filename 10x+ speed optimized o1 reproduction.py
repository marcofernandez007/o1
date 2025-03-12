import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from torch.nn.utils import clip_grad_norm_

# Set seed for reproducibility
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Define the Environment with batched operations
class TextGenerationEnv:
    def __init__(self, target_sequence, batch_size=32):
        self.target_sequence = target_sequence
        self.vocab = list(set("".join(target_sequence)))  # Vocabulary
        self.vocab_size = len(self.vocab)
        self.char_to_idx = {char: idx for idx, char in enumerate(self.vocab)}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        self.max_steps = len(target_sequence)
        self.batch_size = batch_size
        
        # Pre-encode the target sequence for faster comparison
        self.target_encoded = torch.tensor([self.char_to_idx[c] for c in target_sequence], 
                                          device=device)

    def reset(self, batch_size=None):
        """Reset the environment for a new batch of episodes."""
        if batch_size is None:
            batch_size = self.batch_size
            
        self.current_states = ["" for _ in range(batch_size)]
        self.steps = 0
        # Create a tensor of encoded states (initially empty)
        self.encoded_states = torch.zeros(batch_size, 0, dtype=torch.long, device=device)
        self.dones = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        return self.encoded_states

    def step(self, action_indices):
        """Take a step in the environment with batched actions."""
        batch_size = len(action_indices)
        action_chars = [self.idx_to_char[idx.item()] for idx in action_indices]
        
        # Update current states with the new characters
        for i in range(batch_size):
            if not self.dones[i]:
                self.current_states[i] += action_chars[i]
        
        self.steps += 1
        
        # Update encoded states tensor by appending new actions
        self.encoded_states = torch.cat([
            self.encoded_states, 
            action_indices.unsqueeze(1)
        ], dim=1)
        
        # Check if done (reached max steps)
        self.dones = torch.logical_or(self.dones, self.steps >= self.max_steps)
        
        # Calculate rewards
        correct_prefix = torch.zeros(batch_size, dtype=torch.bool, device=device)
        for i in range(batch_size):
            if len(self.current_states[i]) <= len(self.target_sequence):
                # Check if current state is a correct prefix of target
                correct_prefix[i] = self.target_sequence.startswith(self.current_states[i])
        
        # Full match gets higher reward
        full_match = torch.tensor(
            [state == self.target_sequence for state in self.current_states], 
            dtype=torch.bool, device=device
        )
        
        rewards = torch.where(full_match, 
                             torch.ones(batch_size, device=device), 
                             torch.where(correct_prefix, 
                                        torch.zeros(batch_size, device=device), 
                                        torch.full((batch_size,), -0.1, device=device)))
        
        return self.encoded_states, rewards, self.dones

# 2. Define the Policy Model with efficiency improvements
class PolicyNetwork(nn.Module):
    def __init__(self, vocab_size, hidden_size=128, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Initialize parameters for faster convergence
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
    
    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)

    def forward(self, x, hidden=None):
        batch_size = x.size(0)
        if hidden is None:
            hidden = self.init_hidden(batch_size)
            
        # Handle empty sequences (at the start of episodes)
        if x.size(1) == 0:
            # Return prediction for the first character and initial hidden state
            logits = torch.zeros(batch_size, 1, self.fc.out_features, device=device)
            return logits, hidden
        
        embed = self.embedding(x)
        out, hidden = self.rnn(embed, hidden)
        logits = self.fc(out)
        return logits, hidden

# 3. Optimized Training Loop
def train_model(env, policy_net, optimizer, num_episodes=1000, gamma=0.99, batch_size=32):
    """Train the policy model using REINFORCE with batched operations."""
    policy_net.train()
    policy_net.to(device)
    
    # Track progress
    episode_count = 0
    all_rewards = []
    
    # Use experience replay buffer for more efficient learning
    replay_buffer = []
    
    while episode_count < num_episodes:
        # Reset environment for a batch of episodes
        states = env.reset(batch_size)
        batch_log_probs = [[] for _ in range(batch_size)]
        batch_rewards = [[] for _ in range(batch_size)]
        
        hidden = policy_net.init_hidden(batch_size)
        dones = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        # Generate episodes
        while not dones.all():
            # Get action probabilities
            logits, hidden = policy_net(states, hidden)
            probs = torch.softmax(logits[:, -1, :], dim=-1)
            
            # Sample actions
            actions = torch.multinomial(probs, 1).squeeze(1)
            
            # Save log probabilities for selected actions
            log_probs_actions = torch.log(probs.gather(1, actions.unsqueeze(1))).squeeze(1)
            
            # Take actions in environment
            next_states, rewards, new_dones = env.step(actions)
            
            # Store log_probs and rewards
            for i in range(batch_size):
                if not dones[i]:
                    batch_log_probs[i].append(log_probs_actions[i])
                    batch_rewards[i].append(rewards[i])
            
            # Update
            states = next_states
            dones = new_dones
        
        # Process all episodes in the batch
        episode_count += batch_size
        
        # Calculate returns and compute gradients
        optimizer.zero_grad()
        
        batch_loss = 0
        for b in range(batch_size):
            episode_log_probs = torch.stack(batch_log_probs[b])
            episode_rewards = batch_rewards[b]
            
            # Compute returns (discounted rewards)
            returns = []
            R = 0
            for r in reversed(episode_rewards):
                R = r + gamma * R
                returns.insert(0, R)
            returns = torch.tensor(returns, device=device)
            
            # Normalize returns for stability
            if len(returns) > 1:
                returns = (returns - returns.mean()) / (returns.std() + 1e-9)
            
            # Compute loss
            episode_loss = -(episode_log_probs * returns).sum()
            batch_loss += episode_loss
            
            # Track rewards
            all_rewards.append(sum(episode_rewards))
        
        # Normalize by batch size
        batch_loss /= batch_size
        
        # Backprop and optimize
        batch_loss.backward()
        clip_grad_norm_(policy_net.parameters(), max_norm=1.0)  # Prevent exploding gradients
        optimizer.step()
        
        if episode_count % 100 <= batch_size:
            avg_reward = sum(all_rewards[-batch_size:]) / batch_size
            print(f"Episodes {episode_count}, Avg Reward: {avg_reward:.2f}")
    
    return all_rewards

# 4. Run the Training
if __name__ == "__main__":
    target_sequence = "hello"
    batch_size = 64  # Larger batch size for parallelism
    
    env = TextGenerationEnv(target_sequence, batch_size=batch_size)
    vocab_size = len(env.vocab)
    
    # Smaller network for this simple task
    policy_net = PolicyNetwork(vocab_size, hidden_size=64)
    
    # Use a more efficient optimizer with appropriate learning rate
    optimizer = optim.AdamW(policy_net.parameters(), lr=0.003, weight_decay=1e-5)
    
    # Use learning rate scheduler for faster convergence
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=5)
    
    rewards = train_model(
        env, 
        policy_net, 
        optimizer, 
        num_episodes=1000, 
        batch_size=batch_size
    )
    
    # Test the trained model
    print("\nTesting the trained model:")
    policy_net.eval()
    state = env.reset(batch_size=1)
    hidden = policy_net.init_hidden(1)
    
    generated_text = ""
    
    with torch.no_grad():
        for _ in range(len(target_sequence)):
            logits, hidden = policy_net(state, hidden)
            probs = torch.softmax(logits[:, -1, :], dim=-1)
            action = torch.argmax(probs).item()  # Take best action in test mode
            generated_text += env.idx_to_char[action]
            
            # Update state for next iteration
            state = torch.cat([state, torch.tensor([[action]], device=device)], dim=1)
    
    print(f"Target: '{target_sequence}'")
    print(f"Generated: '{generated_text}'")
