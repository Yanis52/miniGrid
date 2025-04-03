import gymnasium as gym
import numpy as np
from minigrid.wrappers import ImgObsWrapper
from tensorflow.keras.models import load_model

# Configuration environnement
env = gym.make('MiniGrid-Empty-5x5-v0', render_mode='human')
env = ImgObsWrapper(env)
state_shape = (7, 7, 3)
action_shape = env.action_space.n

# Charger le modèle entraîné
q_model = load_model('q_model.h5')

# Prétraitement des états
def preprocess_state(state):
    return state.reshape(state_shape).astype(np.float32) / 10.0

# Fonction pour sélectionner une action (greedy)
def greedy_action(state):
    state = preprocess_state(state)[np.newaxis, ...]
    q_values = q_model.predict(state, verbose=0)
    return np.argmax(q_values[0])

# Boucle de test
episodes = 10  # Nombre d'épisodes à tester
for episode in range(episodes):
    obs, _ = env.reset()
    state = preprocess_state(obs)
    total_reward = 0
    done = False

    while not done:
        action = greedy_action(state)  # Utilisation de la politique greedy (sans exploration)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        total_reward += reward
        state = preprocess_state(next_obs)

    print(f"Test Episode: {episode:4d} | Reward: {total_reward:4.1f}")

env.close()