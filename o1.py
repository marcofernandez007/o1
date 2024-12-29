import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

# Set seed for reproducibility
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# 1. Define the Environment
class TextGenerationEnv:
    def __init__(self, target_sequence):
        self.target_sequence = target_sequence
        self.vocab = list(set("".join(target_sequence)))  # Vocabulary
        self.vocab_size = len(self.vocab)
        self.char_to_idx = {char: idx for idx, char in enumerate(self.vocab)}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        self.max_steps = len(target_sequence)

    def reset(self):
        """Reset the environment for a new episode."""
        self.current_state = ""
        self.steps = 0
        return self.current_state

    def step(self, action_idx):
        """Take a step in the environment."""
        action_char = self.idx_to_char[action_idx]
        self.current_state += action_char
        self.steps += 1

        done = self.steps >= self.max_steps
        reward = 1.0 if self.current_state == self.target_sequence else -0.1

        return self.current_state, reward, done

    def encode_state(self, state):
        """Encode state as a one-hot vector."""
        return [self.char_to_idx[char] for char in state] if state else []

# 2. Define the Policy Model
class PolicyNetwork(nn.Module):
    def __init__(self, vocab_size, hidden_size=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        embed = self.embedding(x)
        out, hidden = self.rnn(embed, hidden)
        logits = self.fc(out)
        return logits, hidden

# 3. Training Loop
def train_model(env, policy_net, optimizer, num_episodes=1000, gamma=0.99):
    """Train the policy model using REINFORCE."""
    policy_net.train()
    all_rewards = []

    for episode in range(num_episodes):
        state = env.reset()
        log_probs = []
        rewards = []

        hidden = None
        done = False
        while not done:
            # Encode state and choose action
            state_encoded = torch.tensor(env.encode_state(state)).unsqueeze(0)
            logits, hidden = policy_net(state_encoded, hidden)
            probs = torch.softmax(logits[:, -1, :], dim=-1)
            action = torch.multinomial(probs, 1).item()

            # Take action in the environment
            next_state, reward, done = env.step(action)
            log_prob = torch.log(probs[0, action])

            log_probs.append(log_prob)
            rewards.append(reward)

            state = next_state

        # Compute cumulative rewards (discounted)
        cumulative_rewards = []
        total_reward = 0
        for r in reversed(rewards):
            total_reward = r + gamma * total_reward
            cumulative_rewards.insert(0, total_reward)

        cumulative_rewards = torch.tensor(cumulative_rewards)
        all_rewards.append(sum(rewards))

        # Policy gradient update
        loss = -torch.sum(torch.stack(log_probs) * cumulative_rewards)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if episode % 100 == 0:
            print(f"Episode {episode}, Total Reward: {sum(rewards):.2f}")

    return all_rewards

# 4. Run the Training
if __name__ == "__main__":
    target_sequence = "hello"
    env = TextGenerationEnv(target_sequence)

    vocab_size = len(env.vocab)
    policy_net = PolicyNetwork(vocab_size)
    optimizer = optim.Adam(policy_net.parameters(), lr=0.01)

    rewards = train_model(env, policy_net, optimizer, num_episodes=1000)
