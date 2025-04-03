import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from collections import deque
import gymnasium as gym

# Configuration de l'environnement
env = gym.make('Blackjack-v1', sab=True)

# Hyperparamètres
BUFFER_SIZE = 20000
BATCH_SIZE = 128
GAMMA = 0.95
LR = 0.0005
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.995
TARGET_UPDATE = 50

# Réseau de neurones avec Keras
def create_model(input_dim, output_dim):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dense(output_dim, activation='linear')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR), loss='mse')
    return model

# Mémoire de replay
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, next_state, reward, done):
        self.buffer.append((state, action, next_state, reward, done))

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, next_states, rewards, dones = zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), np.array(next_states), np.array(rewards), np.array(dones)

    def __len__(self):
        return len(self.buffer)

# Conversion des états pour le réseau
def preprocess_state(state):
    return np.array(state, dtype=np.float32)

# Initialisation des réseaux
input_dim = 3  # (player_sum, dealer_card, usable_ace)
output_dim = env.action_space.n

policy_net = create_model(input_dim, output_dim)
target_net = create_model(input_dim, output_dim)
target_net.set_weights(policy_net.get_weights())  # Synchronisation initiale

memory = ReplayBuffer(BUFFER_SIZE)

epsilon = EPS_START

# Boucle d'entraînement
episode_rewards = []
for episode in range(10000):
    state, _ = env.reset()
    state = preprocess_state(state)
    total_reward = 0
    done = False

    while not done:
        # Sélection d'action epsilon-greedy
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            q_values = policy_net.predict(state[np.newaxis], verbose=0)
            action = np.argmax(q_values[0])

        # Exécution de l'action
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_state = preprocess_state(next_state)

        # Stockage dans la mémoire de replay
        memory.push(state, action, next_state, reward, done)

        # Mise à jour de l'état et cumul des récompenses
        state = next_state
        total_reward += reward

        # Apprentissage par lots si la mémoire est suffisante
        if len(memory) >= BATCH_SIZE:
            states, actions, next_states, rewards, dones = memory.sample(BATCH_SIZE)

            # Q-values cibles avec le réseau cible (target_net)
            next_q_values = target_net.predict(next_states, verbose=0)
            max_next_q_values = np.max(next_q_values, axis=1)
            target_q_values = rewards + GAMMA * max_next_q_values * (1 - dones)

            # Mise à jour des Q-values courantes avec le réseau policy_net
            q_values = policy_net.predict(states, verbose=0)
            for i in range(BATCH_SIZE):
                q_values[i][actions[i]] = target_q_values[i]

            # Entraînement du modèle sur les Q-values mises à jour
            policy_net.fit(states, q_values, epochs=1, verbose=0)

    # Mise à jour du réseau cible périodiquement
    if episode % TARGET_UPDATE == 0:
        target_net.set_weights(policy_net.get_weights())

    # Réduction progressive de epsilon (exploration/exploitation)
    epsilon = max(EPS_END, epsilon * EPS_DECAY)

    episode_rewards.append(total_reward)

# Évaluation du modèle entraîné sur des parties de test
def play_episode(env, policy_net):
    state, _ = env.reset()
    state = preprocess_state(state)
    done = False

    while not done:
        q_values = policy_net.predict(state[np.newaxis], verbose=0)
        action = np.argmax(q_values[0])

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        state = preprocess_state(next_state)

    return reward

# Test sur 1000 parties pour évaluer les performances du modèle entraîné
results = [play_episode(env, policy_net) for _ in range(1000)]
win_rate = np.mean([r > 0 for r in results])
print(f"Taux de victoire : {win_rate:.2%}")
