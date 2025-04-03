import pickle

import gymnasium as gym
import numpy as np
import neat
from gym.vector.utils import spaces
from neat import DefaultSpeciesSet, DefaultReproduction, DefaultStagnation
from neat.config import Config
from neat.genome import DefaultGenome
from neat.population import Population
from neat.reporting import StdOutReporter
from configparser import ConfigParser
import minigrid
from minigrid.core.mission import MissionSpace

# Configuration de l'environnement
env = gym.make('MiniGrid-Empty-5x5-v0', render_mode='rgb_array')
env.reset()

# Chargement de la configuration
config = Config(
    genome_type=DefaultGenome,
    reproduction_type=DefaultReproduction,
    species_set_type=DefaultSpeciesSet,
    stagnation_type=DefaultStagnation,
    filename='neat_config.txt'
)

# Initialisation correcte des sous-espaces
agent_pos_space = spaces.Box(
    low=np.array([0, 0]),
    high=np.array([15, 15]),  # Pour 16x16 grid
    dtype=int
)

goal_pos_space = spaces.Box(
    low=np.array([0, 0]),
    high=np.array([15, 15]),
    dtype=int
)

observation_space = spaces.Dict({
    'direction': spaces.Discrete(4),
    'mission': goal_pos_space,
    'image': spaces.Box(0, 255, (7,7,3), np.uint8)
})

actions = {
    0: 'turn_left',  # Rotation gauche
    1: 'turn_right',  # Rotation droite
    2: 'move_forward',  # Avancer
    3: 'pickup',  # Prendre objet
    4: 'drop',  # Déposer objet
    5: 'toggle',  # Interagir
    6: 'done'  # Terminer l'épisode
}


# Préprocessing des observations
def preprocess_obs(obs):
    # Conversion de l'image et normalisation
    img = obs['image'].astype(np.float32) / 255.0
    img_flat = img.flatten()

    # Encodage direction (0-3)
    direction = obs['direction']
    direction_oh = np.zeros(4)
    direction_oh[direction] = 1.0

    return np.concatenate([img_flat, direction_oh])

# Initialisation corrigée de l'environnement
env = gym.make('MiniGrid-Empty-8x8-v0', render_mode='rgb_array')
env = env.unwrapped


# Évaluation des génomes
def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        total_reward = 0

        for _ in range(3):  # Réduire le nombre d'essais
            obs, _ = env.reset()
            episode_reward = 0
            previous_pos = None

            for step in range(50):  # Limite de pas réduite
                inputs = preprocess_obs(obs)
                action = np.argmax(net.activate(inputs))

                obs, reward, done, _, _ = env.step(action)

                # Récompense de progression
                current_pos = env.agent_pos
                if current_pos != previous_pos:
                    episode_reward += 0.1
                    previous_pos = current_pos

                if done:
                    episode_reward += 10  # Bonus pour réussite
                    break

            total_reward += episode_reward

        genome.fitness = total_reward / 3



# Entraînement
def run_neat():
    population = Population(config)
    population.add_reporter(StdOutReporter(True))

    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    winner = population.run(eval_genomes, 100)

    return winner


# Visualisation de l'agent entraîné
def visualize_best_agent(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    obs, _ = env.reset()
    env.render()

    while True:
        action = np.argmax(net.activate(preprocess_obs(obs)))
        obs, reward, done, _, _ = env.step(action)
        env.render()

        if done:
            break


if __name__ == "__main__":
    best_genome = run_neat()
    visualize_best_agent(best_genome, config)
