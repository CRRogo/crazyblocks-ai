#!/usr/bin/env python3
"""
Tune DirectAgent weights using genetic algorithm
This evolves the DirectAgent's sophisticated heuristics
"""

import random
import json
from dataclasses import dataclass, asdict
from typing import List, Dict
from direct_agent import DirectAgent
from train_ai import GameEngine

@dataclass
class DirectAgentStrategy:
    """Strategy/weights for DirectAgent"""
    score_priority: float
    min_blocks: float
    column_balance: float
    large_groups: float
    group_merging: float
    cascade_potential: float
    cascade_depth: float
    danger_reduction: float
    top_row_clearance: float
    survival_bonus: float
    space_creation: float
    color_balance: float
    isolated_penalty: float
    look_ahead_bonus: float
    
    @classmethod
    def random(cls):
        """Create random strategy"""
        return cls(
            score_priority=random.uniform(50, 150),
            min_blocks=random.uniform(20, 60),
            column_balance=random.uniform(30, 90),
            large_groups=random.uniform(10, 40),
            group_merging=random.uniform(15, 45),
            cascade_potential=random.uniform(10, 30),
            cascade_depth=random.uniform(5, 25),
            danger_reduction=random.uniform(10, 40),
            top_row_clearance=random.uniform(10, 30),
            survival_bonus=random.uniform(5, 15),
            space_creation=random.uniform(5, 25),
            color_balance=random.uniform(5, 20),
            isolated_penalty=random.uniform(-20, 0),
            look_ahead_bonus=0.0,  # Disabled - random blocks make it useless
        )
    
    def mutate(self, mutation_rate: float = 0.1):
        """Create mutated copy"""
        strategy_dict = asdict(self)
        for key in strategy_dict:
            if random.random() < mutation_rate:
                # Add noise proportional to current value
                current = strategy_dict[key]
                if key == 'isolated_penalty':
                    strategy_dict[key] += random.uniform(-5, 5)
                    strategy_dict[key] = max(-50, min(0, strategy_dict[key]))
                elif key == 'look_ahead_bonus':
                    strategy_dict[key] += random.uniform(-0.1, 0.1)
                    strategy_dict[key] = max(0.0, min(1.0, strategy_dict[key]))
                else:
                    strategy_dict[key] += random.uniform(-current * 0.2, current * 0.2)
                    strategy_dict[key] = max(0, strategy_dict[key])  # Keep positive (except isolated_penalty)
        return DirectAgentStrategy(**strategy_dict)
    
    def crossover(self, other: 'DirectAgentStrategy') -> 'DirectAgentStrategy':
        """Create child from two parents"""
        strategy_dict = {}
        for key in asdict(self):
            # Blend strategies (weighted average)
            alpha = random.uniform(0.3, 0.7)  # Blend factor
            strategy_dict[key] = alpha * self.__dict__[key] + (1 - alpha) * other.__dict__[key]
        return DirectAgentStrategy(**strategy_dict)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for DirectAgent"""
        return {
            'score_priority': self.score_priority,
            'min_blocks': self.min_blocks,
            'column_balance': self.column_balance,
            'large_groups': self.large_groups,
            'group_merging': self.group_merging,
            'cascade_potential': self.cascade_potential,
            'cascade_depth': self.cascade_depth,
            'danger_reduction': self.danger_reduction,
            'top_row_clearance': self.top_row_clearance,
            'survival_bonus': self.survival_bonus,
            'space_creation': self.space_creation,
            'color_balance': self.color_balance,
            'isolated_penalty': self.isolated_penalty,
            'look_ahead_bonus': self.look_ahead_bonus,
        }


class EvolvableDirectAgent:
    """DirectAgent wrapper that can be evolved"""
    
    def __init__(self, strategy: DirectAgentStrategy = None):
        self.strategy = strategy if strategy else DirectAgentStrategy.random()
        self.agent = DirectAgent(look_ahead_depth=2, weights=self.strategy.to_dict())
        self.fitness = 0.0
        self.games_played = 0
        self.total_score = 0
        self.best_score = 0
    
    def play_game(self, game_engine: GameEngine = None, num_games: int = 5) -> dict:
        """Play games and return average score"""
        engine = game_engine if game_engine else GameEngine()
        
        scores = []
        for _ in range(num_games):
            result = self.agent.play_game(engine)
            scores.append(result['score'])
            self.total_score += result['score']
            self.best_score = max(self.best_score, result['score'])
        
        self.games_played += num_games
        self.fitness = sum(scores) / len(scores)
        
        return {
            'avg_score': self.fitness,
            'best_score': max(scores),
            'scores': scores
        }
    
    def reset_fitness(self):
        """Reset fitness tracking"""
        self.fitness = 0.0
        self.games_played = 0
        self.total_score = 0
        self.best_score = 0


def evolve_direct_agent(
    population_size: int = 30,
    generations: int = 50,
    games_per_eval: int = 5,
    mutation_rate: float = 0.15,
    elite_size: int = 5
):
    """Evolve DirectAgent weights using genetic algorithm"""
    
    print("Evolving DirectAgent Weights")
    print("=" * 60)
    print(f"Population: {population_size}, Generations: {generations}")
    print(f"Games per evaluation: {games_per_eval}")
    print("=" * 60)
    print()
    
    # Initialize population
    population = [EvolvableDirectAgent() for _ in range(population_size)]
    engine = GameEngine()
    
    for generation in range(generations):
        # Evaluate all agents
        for agent in population:
            agent.reset_fitness()
            agent.play_game(engine, games_per_eval)
        
        # Sort by fitness
        population.sort(key=lambda a: a.fitness, reverse=True)
        
        best_agent = population[0]
        avg_fitness = sum(a.fitness for a in population) / len(population)
        
        print(
            f"Gen {generation+1:3d} | "
            f"Best: {best_agent.fitness:6.1f} (Score: {best_agent.best_score:6.0f}) | "
            f"Avg: {avg_fitness:6.1f}"
        )
        
        # Create next generation
        new_population = []
        
        # Keep elite
        for i in range(elite_size):
            elite = EvolvableDirectAgent(population[i].strategy)
            elite.fitness = population[i].fitness
            elite.best_score = population[i].best_score
            new_population.append(elite)
        
        # Add some random diversity
        for _ in range(int(population_size * 0.1)):
            new_population.append(EvolvableDirectAgent())
        
        # Fill rest with crossover and mutation
        while len(new_population) < population_size:
            parent1 = random.choice(population[:population_size // 2])
            parent2 = random.choice(population[:population_size // 2])
            child_strategy = parent1.strategy.crossover(parent2.strategy)
            mutated_strategy = child_strategy.mutate(mutation_rate)
            new_population.append(EvolvableDirectAgent(mutated_strategy))
        
        population = new_population
    
    # Final evaluation
    print("\nFinal Evaluation (10 games per agent)...")
    for agent in population[:5]:  # Top 5
        agent.reset_fitness()
        result = agent.play_game(engine, 10)
        print(f"  Fitness: {agent.fitness:.1f}, Best: {agent.best_score:.0f}")
    
    # Save best strategy
    best = population[0]
    best_weights = best.strategy.to_dict()
    
    print("\nBest Weights Found:")
    for key, value in sorted(best_weights.items()):
        print(f"  {key}: {value:.2f}")
    
    # Save to file
    output_file = 'direct_agent_weights.json'
    with open(output_file, 'w') as f:
        json.dump(best_weights, f, indent=2)
    print(f"\nSaved best weights to {output_file}")
    
    return best_weights


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Evolve DirectAgent weights')
    parser.add_argument('--generations', type=int, default=50, help='Number of generations')
    parser.add_argument('--population', type=int, default=30, help='Population size')
    parser.add_argument('--games', type=int, default=5, help='Games per evaluation')
    parser.add_argument('--mutation-rate', type=float, default=0.15, help='Mutation rate')
    
    args = parser.parse_args()
    
    best_weights = evolve_direct_agent(
        population_size=args.population,
        generations=args.generations,
        games_per_eval=args.games,
        mutation_rate=args.mutation_rate
    )

