#!/usr/bin/env python3
"""
Tune GeneticAgent using genetic algorithm
This is the approach that was working better earlier
"""

import random
import json
import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
from train_ai import GeneticAgent, GameEngine, Strategy
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import argparse

class EvolvableGeneticAgent:
    """GeneticAgent wrapper that can be evolved"""
    
    def __init__(self, strategy: Optional[Strategy] = None):
        self.strategy = strategy if strategy else Strategy.random()
        self.agent = GeneticAgent(self.strategy)
        self.fitness = 0.0
        self.games_played = 0
        self.total_score = 0
        self.best_score = 0
    
    def play_game(self, game_engine: GameEngine = None, num_games: int = 3) -> dict:
        """Play games and return average score"""
        engine = game_engine if game_engine else GameEngine()
        
        scores = []
        for _ in range(num_games):
            engine.reset()
            score = 0
            steps = 0
            max_steps = 1000
            
            while not engine.game_over and steps < max_steps:
                valid_actions = engine.get_valid_actions()
                if not valid_actions:
                    break
                
                # Evaluate all actions
                best_action = None
                best_score = float('-inf')
                for row, col in valid_actions:
                    action_score = self.agent.evaluate_action(engine, row, col)
                    if action_score > best_score:
                        best_score = action_score
                        best_action = (row, col)
                
                if not best_action:
                    break
                
                result = engine.act(best_action[0], best_action[1])
                if not result['success']:
                    break
                
                score += result['blocks_eliminated']
                steps += 1
            
            scores.append(score)
            self.total_score += score
            self.best_score = max(self.best_score, score)
        
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


def evaluate_agent_worker(args):
    """Worker function for parallel evaluation"""
    strategy_dict, num_games = args
    strategy = Strategy(**strategy_dict)
    agent = EvolvableGeneticAgent(strategy)
    engine = GameEngine()
    agent.play_game(engine, num_games)
    return {
        'fitness': agent.fitness,
        'best_score': agent.best_score,
        'strategy_dict': strategy_dict
    }


def evolve_genetic_agent(
    population_size: int = 30,
    generations: int = None,
    games_per_eval: int = 3,
    mutation_rate: float = 0.15,
    elite_size: int = 5,
    save_freq: int = 10,
    load_file: str = None,
    num_workers: int = None,
    target_fitness: float = 1000.0,
    stagnation_threshold: int = 20,
    diversity_rate: float = 0.2
):
    """Evolve GeneticAgent weights using genetic algorithm"""
    
    print("Evolving GeneticAgent Weights")
    print("=" * 60)
    print(f"Population: {population_size}")
    if generations:
        print(f"Generations: {generations}")
    else:
        print("Generations: UNLIMITED (Press Ctrl+C to stop)")
    print(f"Games per evaluation: {games_per_eval}")
    print(f"Save frequency: Every {save_freq} generations")
    if num_workers:
        print(f"Parallel workers: {num_workers}")
    else:
        print(f"Parallel workers: Auto (CPU count)")
    print(f"Target fitness: {target_fitness} (will explore until reached)")
    print(f"Stagnation threshold: {stagnation_threshold} generations")
    print(f"Diversity rate: {diversity_rate * 100:.0f}%")
    print("=" * 60)
    print()
    
    # Initialize population
    if load_file:
        try:
            with open(load_file, 'r') as f:
                saved_data = json.load(f)
                if 'best_strategy' in saved_data:
                    best_strat_dict = saved_data['best_strategy']
                    best_strategy = Strategy(**best_strat_dict)
                    population = [EvolvableGeneticAgent(best_strategy)]
                    for _ in range(population_size - 1):
                        mutated = best_strategy.mutate(0.3)
                        population.append(EvolvableGeneticAgent(mutated))
                    start_gen = saved_data.get('generation', 0)
                    print(f"Loaded from {load_file}, resuming from generation {start_gen}")
                else:
                    # Old format
                    weights = saved_data
                    best_strategy = Strategy(**weights)
                    population = [EvolvableGeneticAgent(best_strategy)]
                    for _ in range(population_size - 1):
                        mutated = best_strategy.mutate(0.3)
                        population.append(EvolvableGeneticAgent(mutated))
                    start_gen = 0
                    print(f"Loaded weights from {load_file}")
        except Exception as e:
            print(f"Failed to load {load_file}: {e}")
            print("Starting fresh...")
            population = [EvolvableGeneticAgent() for _ in range(population_size)]
            start_gen = 0
    else:
        population = [EvolvableGeneticAgent() for _ in range(population_size)]
        start_gen = 0
    
    best_overall_fitness = 0.0
    best_overall_agent = None
    stagnation_count = 0
    exploration_boost = 1.0
    
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)
    
    try:
        generation = start_gen
        while True:
            if generations and generation >= generations:
                break
            
            generation += 1
            
            # Evaluate all agents in parallel
            if num_workers > 1 and len(population) > 1:
                strategy_dicts = [asdict(agent.strategy) for agent in population]
                
                with ProcessPoolExecutor(max_workers=num_workers) as executor:
                    futures = {
                        executor.submit(evaluate_agent_worker, (strat_dict, games_per_eval)): i
                        for i, strat_dict in enumerate(strategy_dicts)
                    }
                    
                    results = [None] * len(population)
                    for future in as_completed(futures):
                        idx = futures[future]
                        try:
                            results[idx] = future.result()
                        except Exception as e:
                            print(f"Error evaluating agent {idx}: {e}")
                            results[idx] = {
                                'fitness': 0.0,
                                'best_score': 0,
                                'strategy_dict': strategy_dicts[idx]
                            }
                
                for i, result in enumerate(results):
                    population[i].fitness = result['fitness']
                    population[i].best_score = result['best_score']
                    population[i].total_score = result['fitness'] * games_per_eval
                    population[i].games_played = games_per_eval
            else:
                engine = GameEngine()
                for agent in population:
                    agent.reset_fitness()
                    agent.play_game(engine, games_per_eval)
            
            # Sort by fitness
            population.sort(key=lambda a: a.fitness, reverse=True)
            
            best_agent = population[0]
            avg_fitness = sum(a.fitness for a in population) / len(population)
            
            # Track best overall and stagnation
            improved = False
            if best_agent.fitness > best_overall_fitness:
                best_overall_fitness = best_agent.fitness
                best_overall_agent = best_agent
                stagnation_count = 0
                exploration_boost = 1.0
                improved = True
            else:
                stagnation_count += 1
                if best_overall_fitness < target_fitness:
                    exploration_boost = min(1.0 + (stagnation_count / stagnation_threshold) * 2.0, 3.0)
            
            exploration_status = ""
            if best_overall_fitness < target_fitness:
                if stagnation_count >= stagnation_threshold:
                    exploration_status = f" [EXPLORING x{exploration_boost:.1f}]"
                else:
                    exploration_status = f" [Target: {target_fitness}]"
            
            print(
                f"Gen {generation:3d} | "
                f"Best: {best_agent.fitness:6.1f} (Score: {best_agent.best_score:6.0f}) | "
                f"Avg: {avg_fitness:6.1f} | "
                f"Best Overall: {best_overall_fitness:6.1f}{exploration_status}"
            )
            
            # Create next generation
            new_population = []
            
            elite_to_keep = elite_size
            if best_overall_fitness < target_fitness and stagnation_count >= stagnation_threshold:
                elite_to_keep = max(2, elite_size // 2)
            
            for i in range(elite_to_keep):
                elite = EvolvableGeneticAgent(population[i].strategy)
                elite.fitness = population[i].fitness
                elite.best_score = population[i].best_score
                new_population.append(elite)
            
            diversity_count = int(population_size * diversity_rate)
            if best_overall_fitness < target_fitness and stagnation_count >= stagnation_threshold:
                diversity_count = int(population_size * (diversity_rate * 2))
            
            for _ in range(diversity_count):
                new_population.append(EvolvableGeneticAgent())
            
            while len(new_population) < population_size:
                if best_overall_fitness < target_fitness and stagnation_count >= stagnation_threshold:
                    parent_pool_size = int(population_size * 0.8)
                else:
                    parent_pool_size = population_size // 2
                
                parent1 = random.choice(population[:parent_pool_size])
                parent2 = random.choice(population[:parent_pool_size])
                child_strategy = parent1.strategy.crossover(parent2.strategy)
                mutated_strategy = child_strategy.mutate(mutation_rate * exploration_boost)
                new_population.append(EvolvableGeneticAgent(mutated_strategy))
            
            population = new_population
            
            # Save periodically
            if generation % save_freq == 0:
                best_weights = best_overall_agent.strategy.to_dict() if best_overall_agent else best_agent.strategy.to_dict()
                save_data = {
                    'generation': generation,
                    'best_fitness': best_overall_fitness if best_overall_agent else best_agent.fitness,
                    'best_score': best_overall_agent.best_score if best_overall_agent else best_agent.best_score,
                    'best_strategy': best_weights,
                    'current_best_fitness': best_agent.fitness,
                    'current_avg_fitness': avg_fitness,
                }
                output_file = 'genetic_agent_weights.json'
                with open(output_file, 'w') as f:
                    json.dump(save_data, f, indent=2)
                print(f"  -> Saved checkpoint to {output_file}")
    
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user (Ctrl+C)")
        if best_overall_agent:
            best_weights = best_overall_agent.strategy.to_dict()
            save_data = {
                'generation': generation,
                'best_fitness': best_overall_fitness,
                'best_score': best_overall_agent.best_score,
                'best_strategy': best_weights,
            }
            output_file = 'genetic_agent_weights.json'
            with open(output_file, 'w') as f:
                json.dump(save_data, f, indent=2)
            print(f"Progress saved to {output_file}")
    
    # Final save
    best = best_overall_agent if best_overall_agent else population[0]
    best_weights = best.strategy.to_dict()
    
    print("\nBest Weights Found:")
    for key, value in sorted(best_weights.items()):
        print(f"  {key}: {value:.2f}")
    
    output_file = 'genetic_agent_weights.json'
    save_data = {
        'generation': generation,
        'best_fitness': best.fitness,
        'best_score': best.best_score,
        'best_strategy': best_weights,
    }
    with open(output_file, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\nSaved best weights to {output_file}")
    
    return best_weights


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evolve GeneticAgent weights')
    parser.add_argument('--generations', type=int, default=None, help='Number of generations (default: unlimited)')
    parser.add_argument('--population', type=int, default=30, help='Population size')
    parser.add_argument('--games', type=int, default=3, help='Games per evaluation')
    parser.add_argument('--mutation-rate', type=float, default=0.15, help='Mutation rate')
    parser.add_argument('--save-freq', type=int, default=10, help='Save checkpoint every N generations')
    parser.add_argument('--load', type=str, default=None, help='Load from saved file to resume')
    parser.add_argument('--workers', type=int, default=None, help='Number of parallel workers')
    parser.add_argument('--target-fitness', type=float, default=1000.0, help='Target fitness')
    parser.add_argument('--stagnation-threshold', type=int, default=20, help='Stagnation threshold')
    parser.add_argument('--diversity-rate', type=float, default=0.2, help='Diversity rate')
    
    args = parser.parse_args()
    
    best_weights = evolve_genetic_agent(
        population_size=args.population,
        generations=args.generations,
        games_per_eval=args.games,
        mutation_rate=args.mutation_rate,
        save_freq=args.save_freq,
        load_file=args.load,
        num_workers=args.workers,
        target_fitness=args.target_fitness,
        stagnation_threshold=args.stagnation_threshold,
        diversity_rate=args.diversity_rate
    )

