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
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

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
    bottom_touching_penalty: float  # NEW: Avoid clearing groups touching bottom
    isolated_group_bonus: float     # NEW: Reward clearing isolated groups
    
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
            bottom_touching_penalty=random.uniform(-50, -20),  # NEW
            isolated_group_bonus=random.uniform(15, 45),       # NEW
        )
    
    def mutate(self, mutation_rate: float = 0.1, exploration_boost: float = 1.0):
        """Create mutated copy with optional exploration boost"""
        strategy_dict = asdict(self)
        # Scale mutation rate by exploration boost (higher = more exploration)
        effective_rate = mutation_rate * exploration_boost
        
        for key in strategy_dict:
            if random.random() < effective_rate:
                # Add noise proportional to current value
                current = strategy_dict[key]
                # Increase mutation magnitude with exploration boost
                mutation_scale = 0.2 * exploration_boost
                
                if key == 'isolated_penalty':
                    strategy_dict[key] += random.uniform(-5 * exploration_boost, 5 * exploration_boost)
                    strategy_dict[key] = max(-50, min(0, strategy_dict[key]))
                elif key == 'look_ahead_bonus':
                    strategy_dict[key] += random.uniform(-0.1 * exploration_boost, 0.1 * exploration_boost)
                    strategy_dict[key] = max(0.0, min(1.0, strategy_dict[key]))
                elif key == 'bottom_touching_penalty':
                    strategy_dict[key] += random.uniform(-5 * exploration_boost, 5 * exploration_boost)
                    strategy_dict[key] = max(-60, min(-10, strategy_dict[key]))
                else:
                    strategy_dict[key] += random.uniform(-current * mutation_scale, current * mutation_scale)
                    strategy_dict[key] = max(0, strategy_dict[key])  # Keep positive (except penalties)
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
            'bottom_touching_penalty': self.bottom_touching_penalty,  # NEW
            'isolated_group_bonus': self.isolated_group_bonus,         # NEW
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


def evaluate_agent_worker(args):
    """Worker function for parallel evaluation (must be at module level for multiprocessing)"""
    strategy_dict, num_games = args
    strategy = DirectAgentStrategy(**strategy_dict)
    agent = EvolvableDirectAgent(strategy)
    engine = GameEngine()
    agent.play_game(engine, num_games)
    return {
        'fitness': agent.fitness,
        'best_score': agent.best_score,
        'strategy_dict': strategy_dict
    }

def evolve_direct_agent(
    population_size: int = 30,
    generations: int = None,  # None = unlimited
    games_per_eval: int = 5,
    mutation_rate: float = 0.15,
    elite_size: int = 5,
    save_freq: int = 10,  # Save every N generations
    load_file: str = None,  # Load from file to resume
    num_workers: int = None,  # Number of parallel workers (None = auto)
    target_fitness: float = 1000.0,  # Target fitness to maintain exploration until reached
    stagnation_threshold: int = 20,  # Generations without improvement before boosting exploration
    diversity_rate: float = 0.2  # Fraction of population to keep as random diversity
):
    """Evolve DirectAgent weights using genetic algorithm"""
    
    print("Evolving DirectAgent Weights")
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
    print(f"Diversity rate: {diversity_rate * 100:.0f}% (random individuals per generation)")
    print("=" * 60)
    print()
    
    # Initialize population
    if load_file:
        try:
            with open(load_file, 'r') as f:
                saved_data = json.load(f)
                # Load best strategy if available
                if 'best_strategy' in saved_data:
                    best_strat_dict = saved_data['best_strategy']
                    best_strategy = DirectAgentStrategy(**best_strat_dict)
                    # Start with best strategy + random variations
                    population = [EvolvableDirectAgent(best_strategy)]
                    for _ in range(population_size - 1):
                        mutated = best_strategy.mutate(0.3)  # Higher mutation to diversify
                        population.append(EvolvableDirectAgent(mutated))
                    start_gen = saved_data.get('generation', 0)
                    print(f"Loaded from {load_file}, resuming from generation {start_gen}")
                else:
                    # Old format - just weights
                    weights = saved_data
                    # Convert to strategy (approximate)
                    best_strategy = DirectAgentStrategy(
                        score_priority=weights.get('score_priority', 100),
                        min_blocks=weights.get('min_blocks', 40),
                        column_balance=weights.get('column_balance', 60),
                        large_groups=weights.get('large_groups', 25),
                        group_merging=weights.get('group_merging', 30),
                        cascade_potential=weights.get('cascade_potential', 20),
                        cascade_depth=weights.get('cascade_depth', 15),
                        danger_reduction=weights.get('danger_reduction', 25),
                        top_row_clearance=weights.get('top_row_clearance', 20),
                        survival_bonus=weights.get('survival_bonus', 10),
                        space_creation=weights.get('space_creation', 15),
                        color_balance=weights.get('color_balance', 12),
                        isolated_penalty=weights.get('isolated_penalty', -10),
                        look_ahead_bonus=weights.get('look_ahead_bonus', 0),
                        bottom_touching_penalty=weights.get('bottom_touching_penalty', -35),
                        isolated_group_bonus=weights.get('isolated_group_bonus', 30),
                    )
                    population = [EvolvableDirectAgent(best_strategy)]
                    for _ in range(population_size - 1):
                        mutated = best_strategy.mutate(0.3)
                        population.append(EvolvableDirectAgent(mutated))
                    start_gen = 0
                    print(f"Loaded weights from {load_file}")
        except Exception as e:
            print(f"Failed to load {load_file}: {e}")
            print("Starting fresh...")
            population = [EvolvableDirectAgent() for _ in range(population_size)]
            start_gen = 0
    else:
        population = [EvolvableDirectAgent() for _ in range(population_size)]
        start_gen = 0
    
    best_overall_fitness = 0.0
    best_overall_agent = None
    stagnation_count = 0  # Count generations without improvement
    exploration_boost = 1.0  # Multiplier for mutation rate (increases with stagnation)
    
    # Determine number of workers
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)  # Use all but one core
    
    try:
        generation = start_gen
        while True:
            if generations and generation >= generations:
                break
            
            generation += 1
            
            # Evaluate all agents in parallel
            if num_workers > 1 and len(population) > 1:
                # Parallel evaluation
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
                            # Fallback: create random result
                            results[idx] = {
                                'fitness': 0.0,
                                'best_score': 0,
                                'strategy_dict': strategy_dicts[idx]
                            }
                
                # Update agents with results
                for i, result in enumerate(results):
                    population[i].fitness = result['fitness']
                    population[i].best_score = result['best_score']
                    population[i].total_score = result['fitness'] * games_per_eval
                    population[i].games_played = games_per_eval
            else:
                # Sequential evaluation (fallback or single worker)
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
                exploration_boost = 1.0  # Reset exploration boost on improvement
                improved = True
            else:
                stagnation_count += 1
                # Increase exploration if stagnating and below target
                if best_overall_fitness < target_fitness:
                    # Boost exploration: 1.0 -> 1.5 -> 2.0 -> 2.5 (caps at 3.0)
                    exploration_boost = min(1.0 + (stagnation_count / stagnation_threshold) * 2.0, 3.0)
            
            # Show exploration status
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
        
        # Create next generation with diversity maintenance
        new_population = []
        
        # Keep elite (but fewer if we're exploring)
        elite_to_keep = elite_size
        if best_overall_fitness < target_fitness and stagnation_count >= stagnation_threshold:
            # Reduce elite when exploring to allow more diversity
            elite_to_keep = max(2, elite_size // 2)
        
        for i in range(elite_to_keep):
            elite = EvolvableDirectAgent(population[i].strategy)
            elite.fitness = population[i].fitness
            elite.best_score = population[i].best_score
            new_population.append(elite)
        
        # Add random diversity (more if stagnating and below target)
        diversity_count = int(population_size * diversity_rate)
        if best_overall_fitness < target_fitness and stagnation_count >= stagnation_threshold:
            # Increase diversity when stagnating below target
            diversity_count = int(population_size * (diversity_rate * 2))
        
        for _ in range(diversity_count):
            new_population.append(EvolvableDirectAgent())
        
        # Fill rest with crossover and mutation (with exploration boost)
        while len(new_population) < population_size:
            # Select parents from broader pool when exploring
            if best_overall_fitness < target_fitness and stagnation_count >= stagnation_threshold:
                # Use larger parent pool for more diversity
                parent_pool_size = int(population_size * 0.8)
            else:
                parent_pool_size = population_size // 2
            
            parent1 = random.choice(population[:parent_pool_size])
            parent2 = random.choice(population[:parent_pool_size])
            child_strategy = parent1.strategy.crossover(parent2.strategy)
            # Apply exploration boost to mutation
            mutated_strategy = child_strategy.mutate(mutation_rate, exploration_boost)
            new_population.append(EvolvableDirectAgent(mutated_strategy))
        
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
                output_file = 'direct_agent_weights.json'
                with open(output_file, 'w') as f:
                    json.dump(save_data, f, indent=2)
                print(f"  -> Saved checkpoint to {output_file}")
    
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user (Ctrl+C)")
        # Save on exit
        if best_overall_agent:
            best_weights = best_overall_agent.strategy.to_dict()
            save_data = {
                'generation': generation,
                'best_fitness': best_overall_fitness,
                'best_score': best_overall_agent.best_score,
                'best_strategy': best_weights,
            }
            output_file = 'direct_agent_weights.json'
            with open(output_file, 'w') as f:
                json.dump(save_data, f, indent=2)
            print(f"Progress saved to {output_file}")
    
    # Final evaluation
    print("\nFinal Evaluation (10 games per agent)...")
    for agent in population[:5]:  # Top 5
        agent.reset_fitness()
        result = agent.play_game(engine, 10)
        print(f"  Fitness: {agent.fitness:.1f}, Best: {agent.best_score:.0f}")
    
    # Save best strategy
    best = best_overall_agent if best_overall_agent else population[0]
    best_weights = best.strategy.to_dict()
    
    print("\nBest Weights Found:")
    for key, value in sorted(best_weights.items()):
        print(f"  {key}: {value:.2f}")
    
    # Final save
    output_file = 'direct_agent_weights.json'
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
    import argparse
    
    parser = argparse.ArgumentParser(description='Evolve DirectAgent weights')
    parser.add_argument('--generations', type=int, default=None, help='Number of generations (default: unlimited - runs until Ctrl+C)')
    parser.add_argument('--population', type=int, default=30, help='Population size')
    parser.add_argument('--games', type=int, default=3, help='Games per evaluation (fewer = faster)')
    parser.add_argument('--mutation-rate', type=float, default=0.15, help='Mutation rate')
    parser.add_argument('--save-freq', type=int, default=10, help='Save checkpoint every N generations')
    parser.add_argument('--load', type=str, default=None, help='Load from saved file to resume training')
    parser.add_argument('--workers', type=int, default=None, help='Number of parallel workers (default: auto)')
    parser.add_argument('--target-fitness', type=float, default=1000.0, help='Target fitness - keeps exploring until reached')
    parser.add_argument('--stagnation-threshold', type=int, default=20, help='Generations without improvement before boosting exploration')
    parser.add_argument('--diversity-rate', type=float, default=0.2, help='Fraction of population kept as random diversity (0.0-1.0)')
    
    args = parser.parse_args()
    
    best_weights = evolve_direct_agent(
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

