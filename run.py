import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Input, Lambda
from collections import deque
from minigrid.envs import EmptyEnv

# Paramètres d'entraînement
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995  # Décroissance plus douce pour une meilleure exploration
gamma = 0.99
batch_size = 32
memory_size = 100000
episodes = 500
target_update_freq = 5  # Fréquence de mise à jour du modèle cible (en nombre d'épisodes)

# Création de l'environnement MiniGrid
env = gym.make('MiniGrid-Empty-5x5-v0', render_mode='human')

state_shape = env.observation_space["image"].shape
action_shape = env.action_space.n

def create_q_model():
    inputs = Input(shape=state_shape)
    # Normalisation des pixels (de 0 à 1)
    x = Lambda(lambda img: img / 255.0)(inputs)
    # Extraction de caractéristiques via des couches convolutionnelles
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    outputs = Dense(action_shape)(x)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')
    return model

# Création du modèle et du modèle cible
q_model = create_q_model()
target_model = create_q_model()
target_model.set_weights(q_model.get_weights())

# Mémoire pour l'expérience replay
memory = deque(maxlen=memory_size)

def store_experience(state, action, reward, next_state, done):
    memory.append((state, action, reward, next_state, done))

def sample_batch():
    indices = np.random.choice(len(memory), batch_size, replace=False)
    states, actions, rewards, next_states, dones = zip(*[memory[i] for i in indices])
    return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

# Politique epsilon-greedy
def epsilon_greedy_action(state):
    if np.random.rand() < epsilon:
        return int(env.action_space.sample())
    q_values = q_model.predict(state[np.newaxis, ...], verbose=0)
    return int(np.argmax(q_values[0]))

# Processus d'entraînement : mise à jour du modèle
def train_step():
    if len(memory) < batch_size:
        return

    states, actions, rewards, next_states, dones = sample_batch()

    # Calcul des Q-values cibles
    target_q = q_model.predict(states, verbose=0)
    next_q = target_model.predict(next_states, verbose=0)
    max_next_q = np.max(next_q, axis=1)

    for i in range(batch_size):
        # Correction : mise à jour de la Q-value pour l'action réellement choisie
        target = rewards[i] + (gamma * max_next_q[i] * (1 - int(dones[i])))
        target_q[i][actions[i]] = target

    q_model.fit(states, target_q, epochs=1, verbose=0)

# Historique des récompenses pour le suivi
reward_history = []

# Boucle d'entraînement principale
for episode in range(episodes):
    obs, info = env.reset()
    state = obs["image"]
    total_reward = 0
    done = False

    while not done:
        action = epsilon_greedy_action(state)
        next_obs, reward, terminated, truncated, info = env.step(action)
        next_state = next_obs["image"]
        done = terminated or truncated
        store_experience(state, action, reward, next_state, done)
        total_reward += reward
        state = next_state
        train_step()

    # Mise à jour d'epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    # Mise à jour du modèle cible à la fréquence définie
    if episode % target_update_freq == 0:
        target_model.set_weights(q_model.get_weights())
        target_model.save_weights('minigrid_model.weights.h5')  # Correction ici

    reward_history.append(total_reward)
    print(f"Episode: {episode}, Reward: {total_reward}")
