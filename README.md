# PROJET MiniGrid
Yanis B.-Enzo D..-Kenuhn R.

## Documentation du Code NEAT pour MiniGrid

### Description Générale

Ce code implémente un agent d'apprentissage par évolution (NEAT - NeuroEvolution of Augmenting Topologies) pour résoudre des environnements MiniGrid. L'agent apprend à naviguer dans une grille 6x6 en utilisant des réseaux de neurones évolutifs.

### Prérequis

Les bibliothèques Python suivantes doivent être installées :

- gymnasium
- numpy
- neat-python
- minigrid

### Structure du Code

1. Configuration de l'Environnement
   - L'environnement utilisé est 'MiniGrid-Empty-Random-6x6-v0', un espace de grille 6x6 vide avec des positions aléatoires.
2. Espaces d'Observation et Actions
   - Position de l'agent : grille 16x16
   - Position de l'objectif : grille 16x16
   - Actions possibles : tourner à gauche/droite, avancer, ramasser, déposer, interagir, terminer

3. Comportement du Programme
   - Si un fichier 'winner.pkl' existe, le meilleur agent sera chargé et visualisé
   - Sinon, un nouvel entraînement sera lancé
   - L'entraînement s'exécute sur 200 générations
   - Le meilleur génome est sauvegardé dans 'winner.pkl'

### Paramètres Ajustables

Vous pouvez modifier les paramètres suivants pour optimiser l'apprentissage :

- Nombre de générations (actuellement 200)
- Nombre d'essais par évaluation (actuellement 3)
- Récompenses et pénalités dans eval_genomes
- Taille de la grille et type d'environnement

### Résultats Attendus

Après l'entraînement, l'agent devrait être capable de :

- Naviguer efficacement dans la grille
- Éviter les obstacles
- Atteindre l'objectif de manière optimale