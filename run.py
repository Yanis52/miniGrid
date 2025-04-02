import gymnasium as gym
from minigrid.envs import EmptyEnv

def main():
    # Creation de l'environnement MiniGrid
    env = gym.make('MiniGrid-Empty-16x16-v0',render_mode='human')

    # Réinitialiser l'environnement
    obs, info = env.reset()
    done = False

    print("Observation Space:", env.observation_space)
    print("Action Space:", env.action_space)

    # Boucle principale pour interagir avec l'environnement
    while not done:
        # Prendre une action aléatoire
        
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        # Vérifier si l'épisode est terminé
        done = terminated or truncated

        # Afficher l'environnement
        env.render()

        
        print(f"Action: {action}, Reward: {reward}, Done: {done}")

    # Fermer l'environnement
    env.close()

if __name__ == "__main__":
    main()