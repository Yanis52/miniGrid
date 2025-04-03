import os

import gymnasium as gym
import numpy as np
from keras.layers import Dense, Conv2D
from keras.saving.save import load_model
from tensorflow.keras import Model, layers, optimizers
from collections import deque
from minigrid.wrappers import ImgObsWrapper
import tensorflow as tf

# Hyperparamètres
epsilon = 1
epsilon_min = 0.01
epsilon_decay = 0.995
gamma = 0.99
batch_size = 64
memory_size = 100000
episodes = 500
learning_rate = 1e-6

# Configuration environnement
env = gym.make('MiniGrid-Empty-5x5-v0', render_mode='human')
env = ImgObsWrapper(env)
state_shape = env.observation_space.shape
action_shape = env.action_space.n

print(f"State shape: {state_shape}, Actions: {action_shape}")


def create_q_model():
    inputs = tf.keras.Input(shape=state_shape)

    x = Conv2D(32, kernel_size=3, strides=2, activation='relu')(inputs)
    x = layers.Flatten()(x)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(action_shape)(x)

    model = Model(inputs, outputs)
    model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate),
                  loss='huber')
    return model

if os.path.exists('q_model.h5'):
    print("Les modèles existent déjà, chargement des poids...")
    q_model = load_model('q_model.h5')
    target_model = load_model('q_model.h5')
else:
    q_model = create_q_model()
    target_model = create_q_model()
    target_model.set_weights(q_model.get_weights())

memory = deque(maxlen=memory_size)


def preprocess_state(state):
    return state.astype(np.float32) / 255.0


def store_experience(state, action, reward, next_state, done):
    state = preprocess_state(state)
    next_state = preprocess_state(next_state)
    memory.append((state, action, reward, next_state, done))


def sample_batch():
    indices = np.random.choice(len(memory), batch_size, replace=False)
    states, actions, rewards, next_states, dones = zip(*[memory[i] for i in indices])
    return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)


def epsilon_greedy_action(state):
    if np.random.rand() < epsilon:
        return env.action_space.sample()

    state = preprocess_state(state)[np.newaxis, ...]
    q_values = q_model.predict(state, verbose=0)
    return np.argmax(q_values[0])


def train_step():
    if len(memory) < batch_size:
        return 0.0

    states, actions, rewards, next_states, dones = sample_batch()

    target_q = q_model.predict(next_states, verbose=0)
    next_q = target_model.predict(next_states, verbose=0)
    max_target_q = np.max(next_q, axis=1)

    for i in range(batch_size):
        target_q[i][actions[i]] = rewards[i] if dones[i] else rewards[i] + gamma * max_target_q[i]

    # Mise à jour du modèle
    q_model.fit(states, target_q, epochs=1, verbose=0)


# Boucle d'entraînement
# Boucle d'entraînement
for episode in range(episodes):
    obs, _ = env.reset()
    state = obs
    total_reward = 0
    done = False
    previous_pos = None
    same_pos_count = 0
    max_same_pos_count = 5  # Nombre maximum de fois que l'agent peut rester sur la même case

    while not done:
        action = epsilon_greedy_action(state)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Accéder à l'environnement sous-jacent pour obtenir la position de l'agent
        current_pos = env.unwrapped.agent_pos
        if current_pos == previous_pos:
            same_pos_count += 1
            if same_pos_count >= max_same_pos_count:
                reward -= 5  # Pénalité pour rester sur la même case
        else:
            same_pos_count = 0

        previous_pos = current_pos

        if done and terminated:
            reward += 100

        store_experience(state, action, reward, next_obs, done)
        loss = train_step()

        total_reward += reward
        state = next_obs

    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    if episode % 10 == 0:
        target_model.set_weights(q_model.get_weights())
        target_model.save('q_model.h5')

    if episode % 50 == 0:
        env.render()

    print(f"Episode: {episode:4d} | Reward: {total_reward:4.1f} | Epsilon: {epsilon:4.2f}")

env.close()