import random
import torch
from model import pongAgent
import ponggame
from typing import List
from pickle import dump, load

def create_population(population_size: int, hidden_size: int) -> List[pongAgent]:
    return [pongAgent(hidden_size=hidden_size) for _ in range(population_size)]

def evaluate_fitness(agent: pongAgent) -> int:
    final_state = ponggame.runLogicAgent(agent)
    return final_state.score

def select_parents(population: List[pongAgent], fitness_scores: List[int], num_parents: int) -> List[pongAgent]:
    parents = []
    for _ in range(num_parents):
        max_fitness_idx = fitness_scores.index(max(fitness_scores))
        parents.append(population[max_fitness_idx])
        fitness_scores[max_fitness_idx] = -1  # Ensure this parent isn't selected again
    return parents

def crossover(parent1: pongAgent, parent2: pongAgent) -> pongAgent:
    child = pongAgent(hidden_size=parent1.model.hidden_size)
    for child_param, parent1_param, parent2_param in zip(child.model.parameters(), parent1.model.parameters(), parent2.model.parameters()):
        child_param.data.copy_(torch.where(torch.rand_like(child_param) > 0.5, parent1_param, parent2_param))
    return child

def mutate(agent: pongAgent, mutation_rate: float, mutation_strength: float):
    for param in agent.model.parameters():
        if random.random() < mutation_rate:
            param.data += torch.randn_like(param) * mutation_strength

def genetic_algorithm(population_size: int, hidden_size: int, num_generations: int, num_parents: int, mutation_rate: float, mutation_strength: float):
    population = create_population(population_size, hidden_size)
    
    for generation in range(num_generations):
        fitness_scores = [evaluate_fitness(agent) for agent in population]
        best_score = max(fitness_scores)
        avg_score = sum(fitness_scores) / len(fitness_scores)
        print(f"Generation {generation + 1}: Best Score = {best_score}, Average Score = {avg_score:.2f}")
        
        parents = select_parents(population, fitness_scores, num_parents)
        next_generation = parents.copy()
        
        while len(next_generation) < population_size:
            parent1, parent2 = random.sample(parents, 2)
            child = crossover(parent1, parent2)
            mutate(child, mutation_rate, mutation_strength)
            next_generation.append(child)
        
        population = next_generation
    
    return population[fitness_scores.index(max(fitness_scores))]

if __name__ == '__main__':
    # Hyperparameters
    POPULATION_SIZE = 50
    HIDDEN_SIZE = 16
    NUM_GENERATIONS = 20
    NUM_PARENTS = 10
    MUTATION_RATE = 0.1
    MUTATION_STRENGTH = 0.1
    
    best_agent = genetic_algorithm(POPULATION_SIZE, HIDDEN_SIZE, NUM_GENERATIONS, NUM_PARENTS, MUTATION_RATE, MUTATION_STRENGTH)
    final_score = evaluate_fitness(best_agent)
    print(f"Final best agent score: {final_score}")
    dump(best_agent, open("best_model.pk","wb"))