#!/usr/bin/env python3
"""
Fast Python training script for Crazy Blocks AI
Runs much faster than JavaScript version using NumPy and parallel processing
"""

import json
import os
import random
import time
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# Game constants (must match JavaScript)
COLUMNS = 5
ROWS = 17
COLORS = ['#6BA85A', '#A08FB8', '#D4A5A0', '#7DADB5']
COLOR_INDICES = {color: idx for idx, color in enumerate(COLORS)}


@dataclass
class Strategy:
    """AI strategy with weights for decision making"""
    groupSizeWeight: float
    cascadePotentialWeight: float
    topClearWeight: float
    bottomAvoidWeight: float
    colorDiversityWeight: float
    isolationPenalty: float
    spaceCreationWeight: float
    columnHeightWeight: float = 0.0
    dangerReductionWeight: float = 0.0
    colorBalanceWeight: float = 0.0
    topDensityWeight: float = 0.0
    multiCascadeWeight: float = 0.0
    minimumGroupSizeWeight: float = 0.0  # NEW: Penalty for groups < 5 blocks
    largeGroupBonusWeight: float = 0.0  # NEW: Bonus for groups >= 5 blocks
    columnBalanceWeight: float = 0.0  # NEW: Column height balance (keep columns similar height)
    groupMergingWeight: float = 0.0  # NEW: Group merging potential (reward small eliminations that merge large groups)
    averageBlocksWeight: float = 0.0  # NEW: Average blocks per turn (reward 5+ blocks eliminated)
    maxColumnReductionWeight: float = 0.0  # NEW: Reward moves that reduce the highest column (prevents game over)
    smallClearEnablerWeight: float = 0.0  # NEW: Reward small clears that enable future large clears
    bottomTouchingAvoidanceWeight: float = 0.0  # NEW: Avoid clearing groups touching bottom (they can grow)
    colorConcentrationWeight: float = 0.0  # NEW: Reward clearing colors that will concentrate remaining colors
    topRowAvoidanceWeight: float = 0.0  # NEW: Avoid clearing blocks at very top - clear lower for falling groups
    columnIsolationPenaltyWeight: float = 0.0  # NEW: Penalize isolated tall columns (less likely to match)
    criticalColumnFocusWeight: float = 0.0  # NEW: MUST focus on columns right below max (about to lose)
    sideConnectionRequirementWeight: float = 0.0  # NEW: Avoid clearing blocks with no side connections

    @classmethod
    def random(cls):
        """Create a random strategy"""
        return cls(
            groupSizeWeight=random.uniform(-1, 1),
            cascadePotentialWeight=random.uniform(-1, 1),
            topClearWeight=random.uniform(-1, 1),
            bottomAvoidWeight=random.uniform(-1, 1),
            colorDiversityWeight=random.uniform(-1, 1),
            isolationPenalty=random.uniform(-1, 1),
            spaceCreationWeight=random.uniform(-1, 1),
            columnHeightWeight=random.uniform(-1, 1),
            dangerReductionWeight=random.uniform(-1, 1),
            colorBalanceWeight=random.uniform(-1, 1),
            topDensityWeight=random.uniform(-1, 1),
            multiCascadeWeight=random.uniform(-1, 1),
            minimumGroupSizeWeight=random.uniform(-1, 1),
            largeGroupBonusWeight=random.uniform(-1, 1),
            columnBalanceWeight=random.uniform(-1, 1),
            groupMergingWeight=random.uniform(-1, 1),
            averageBlocksWeight=random.uniform(-1, 1),
            maxColumnReductionWeight=random.uniform(-1, 1),
            smallClearEnablerWeight=random.uniform(-1, 1),
            bottomTouchingAvoidanceWeight=random.uniform(-1, 1),
            colorConcentrationWeight=random.uniform(-1, 1),
            topRowAvoidanceWeight=random.uniform(-1, 1),
            columnIsolationPenaltyWeight=random.uniform(-1, 1),
            criticalColumnFocusWeight=random.uniform(-1, 1),
            sideConnectionRequirementWeight=random.uniform(-1, 1)
        )

    def mutate(self, mutation_rate: float = 0.1):
        """Create a mutated copy"""
        strategy_dict = asdict(self)
        for key in strategy_dict:
            if random.random() < mutation_rate:
                strategy_dict[key] += random.uniform(-0.2, 0.2)
                strategy_dict[key] = max(-2, min(2, strategy_dict[key]))
        return Strategy(**strategy_dict)

    def crossover(self, other: 'Strategy') -> 'Strategy':
        """Create a child strategy from two parents"""
        strategy_dict = {}
        for key in asdict(self):
            strategy_dict[key] = self.__dict__[key] if random.random() < 0.5 else other.__dict__[key]
        return Strategy(**strategy_dict)


class GameEngine:
    """Headless game engine (matches JavaScript version)"""
    
    def __init__(self):
        self.grid = self._initialize_grid()
        self.score = 0
        self.game_over = False
        self.turn_count = 0

    def _initialize_grid(self):
        return [[None for _ in range(COLUMNS)] for _ in range(ROWS)]

    def reset(self):
        """Reset game and add 4 initial rows"""
        self.grid = self._initialize_grid()
        self.score = 0
        self.game_over = False
        self.turn_count = 0
        # Start with 4 rows
        for _ in range(4):
            self.add_new_row()
        return self

    def add_new_row(self):
        """Add a new row at the bottom"""
        if self.game_over:
            return
        
        new_row = [random.choice(COLORS) for _ in range(COLUMNS)]
        # Shift all rows up and add new row at bottom
        self.grid = self.grid[1:] + [new_row]
        
        # Check if game over (top row has blocks)
        if any(self.grid[0]):
            self.game_over = True

    def find_connected_blocks(self, row: int, col: int, color: str, visited: set = None) -> List[Tuple[int, int]]:
        """Find all connected blocks of the same color (flood fill)"""
        if visited is None:
            visited = set()
        
        key = (row, col)
        if key in visited:
            return []
        
        # Check bounds
        if row < 0 or row >= ROWS or col < 0 or col >= COLUMNS:
            return []
        
        # Check if block exists and matches color
        if self.grid[row][col] != color:
            return []
        
        visited.add(key)
        connected = [(row, col)]
        
        # Check adjacent blocks (not diagonal)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            connected.extend(self.find_connected_blocks(new_row, new_col, color, visited))
        
        return connected

    def _find_connected_in_grid(self, grid, row: int, col: int, color: str, visited: set = None) -> List[Tuple[int, int]]:
        """Find all connected blocks of the same color in a given grid (helper for merge detection)"""
        if visited is None:
            visited = set()
        
        key = (row, col)
        if key in visited:
            return []
        
        # Check bounds
        if row < 0 or row >= ROWS or col < 0 or col >= COLUMNS:
            return []
        
        # Check if block exists and matches color
        if grid[row][col] != color:
            return []
        
        visited.add(key)
        connected = [(row, col)]
        
        # Check adjacent blocks (not diagonal)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            connected.extend(self._find_connected_in_grid(grid, new_row, new_col, color, visited))
        
        return connected
    
    def apply_gravity(self):
        """Drop blocks down after elimination"""
        for col in range(COLUMNS):
            write_index = ROWS - 1
            for row in range(ROWS - 1, -1, -1):
                if self.grid[row][col] is not None:
                    if write_index != row:
                        self.grid[write_index][col] = self.grid[row][col]
                        self.grid[row][col] = None
                    write_index -= 1

    def act(self, row: int, col: int) -> Dict:
        """Perform an action (click a block)"""
        if self.game_over or row == ROWS - 1 or self.grid[row][col] is None:
            return {'success': False, 'blocks_eliminated': 0, 'reward': 0}
        
        color = self.grid[row][col]
        connected_blocks = self.find_connected_blocks(row, col, color)
        
        if len(connected_blocks) == 0:
            return {'success': False, 'blocks_eliminated': 0, 'reward': 0}
        
        # Eliminate blocks
        for r, c in connected_blocks:
            self.grid[r][c] = None
        
        blocks_eliminated = len(connected_blocks)
        
        # Calculate column heights BEFORE elimination (for balance reward)
        column_heights_before = [sum(1 for r in range(ROWS) if self.grid[r][c] is not None) 
                                 for c in range(COLUMNS)]
        
        self.score += blocks_eliminated
        
        # Apply gravity
        self.apply_gravity()
        
        # Add new row (turn ends)
        self.add_new_row()
        self.turn_count += 1
        
        # Calculate reward - OVERALL SCORE IS ABSOLUTELY PRIMARY
        # The final score (cumulative blocks eliminated) is ALL that matters
        # Make score reward completely dominant
        
        # PRIMARY REWARD: Score itself (blocks eliminated this turn)
        # This directly contributes to overall score - make it HUGE
        score_reward = blocks_eliminated * 50.0  # MASSIVELY prioritize score (was 10.0)
        
        # Tiny strategic bonuses (100x smaller than score reward)
        # These provide minimal guidance but NEVER overshadow score
        
        # Efficiency bonus: Tiny reward for meeting 5+ blocks requirement
        if blocks_eliminated >= 5:
            efficiency_bonus = 0.1  # Tiny (was 1.0)
            if blocks_eliminated > 5:
                efficiency_bonus += (blocks_eliminated - 5) * 0.01  # Minuscule
        else:
            # Tiny penalty for not meeting requirement
            efficiency_bonus = -(5 - blocks_eliminated) * 0.05  # Tiny (was 0.5)
        
        # Large group bonus: Tiny extra for big eliminations
        large_group_bonus = 0
        if blocks_eliminated >= 10:
            large_group_bonus = 0.2  # Tiny (was 2.0)
        elif blocks_eliminated >= 8:
            large_group_bonus = 0.1  # Tiny (was 1.0)
        
        # Column balance: Minuscule bonus (helps but never dominates)
        column_heights_after = [sum(1 for r in range(ROWS) if self.grid[r][c] is not None) 
                               for c in range(COLUMNS)]
        if column_heights_after:
            avg_height = sum(column_heights_after) / len(column_heights_after)
            variance = sum((h - avg_height) ** 2 for h in column_heights_after) / len(column_heights_after)
            std_dev = variance ** 0.5
            balance_bonus = max(0, (8 - std_dev) / 8.0) * 0.05  # Minuscule (was 0.5)
        else:
            balance_bonus = 0
        
        # GROUP MERGING BONUS: Explicitly reward eliminating small separators that merge large groups
        # This makes the neural network learn this strategy faster
        group_merging_bonus = 0.0
        if blocks_eliminated <= 3:  # Small elimination (1-3 blocks)
            # Check if there's a large merged group of the same color after gravity
            # This indicates we eliminated a small separator between two large groups
            visited = set()
            largest_merged_group = 0
            
            # Find the largest group of the same color as what we eliminated
            for r in range(ROWS - 1):  # Don't check bottom row (can't click)
                for c in range(COLUMNS):
                    if self.grid[r][c] == color and (r, c) not in visited:
                        merged_group = self._find_connected_in_grid(self.grid, r, c, color, visited)
                        if len(merged_group) > largest_merged_group:
                            largest_merged_group = len(merged_group)
            
            # If the merged group is significantly larger than what we eliminated, it's a good merge
            # Example: eliminate 2 blocks, create a 15-block group = merged two ~7-block groups
            if largest_merged_group >= 10 and largest_merged_group > blocks_eliminated * 3:
                # Calculate bonus: larger merged group relative to eliminated blocks = better
                # Scale: (merged_size - eliminated_size) / merged_size, capped at reasonable value
                merge_ratio = (largest_merged_group - blocks_eliminated) / max(largest_merged_group, 1)
                # Bonus scales with merge quality, but still much smaller than score reward
                # This provides a clear signal without overshadowing the actual score
                group_merging_bonus = merge_ratio * 25.0  # Moderate bonus (25-50 for good merges)
                # Cap it so it doesn't exceed score reward for very small eliminations
                group_merging_bonus = min(group_merging_bonus, 50.0)
        
        # Game over penalty: Strong negative (but score reward still dominates)
        game_over_penalty = -500.0 if self.game_over else 0  # Stronger penalty
        
        # Total reward: Score is primary, bonuses provide guidance
        reward = score_reward + efficiency_bonus + large_group_bonus + balance_bonus + group_merging_bonus + game_over_penalty
        
        return {
            'success': True,
            'blocks_eliminated': blocks_eliminated,
            'reward': reward,
            'game_over': self.game_over
        }

    def get_valid_actions(self) -> List[Tuple[int, int]]:
        """Get all valid actions (clickable positions)"""
        actions = []
        for row in range(ROWS - 1):  # Exclude bottom row
            for col in range(COLUMNS):
                if self.grid[row][col] is not None:
                    actions.append((row, col))
        return actions


class GeneticAgent:
    """Genetic algorithm agent"""
    
    def __init__(self, strategy: Optional[Strategy] = None):
        self.strategy = strategy if strategy else Strategy.random()
        self.fitness = 0.0
        self.games_played = 0
        self.total_score = 0
        self.best_score = 0

    def evaluate_action(self, game_engine: GameEngine, row: int, col: int) -> float:
        """Evaluate an action based on strategy"""
        if row == ROWS - 1 or game_engine.grid[row][col] is None:
            return float('-inf')
        
        color = game_engine.grid[row][col]
        connected_blocks = game_engine.find_connected_blocks(row, col, color)
        group_size = len(connected_blocks)
        
        if group_size == 0:
            return float('-inf')
        
        score = 0.0
        
        # Group size preference
        score += self.strategy.groupSizeWeight * group_size
        
        # NEW: Critical game mechanic - continuous reward/penalty based on blocks removed
        # The reward/punishment scales smoothly with the actual number of blocks
        # 1 block = maximum penalty, 4 blocks = small penalty, 5 blocks = neutral
        # 8 blocks = moderate bonus, 10 blocks = larger bonus, etc.
        
        if group_size < 5:
            # Penalty scales with how small the group is
            # 1 block = -1.0, 2 blocks = -0.75, 3 blocks = -0.5, 4 blocks = -0.25
            penalty = (5 - group_size) / 4.0  # 1.0 to 0.25 (worse for smaller groups)
            score += self.strategy.minimumGroupSizeWeight * (-penalty * 10)  # Strong negative signal
        else:
            # Bonus scales continuously with size above 5
            # 5 blocks = 0.0, 8 blocks = 0.3, 10 blocks = 0.5, 15 blocks = 1.0
            bonus = min((group_size - 5) / 10.0, 1.0)  # Normalized 0.0 to 1.0, capped at 1.0
            score += self.strategy.largeGroupBonusWeight * (bonus * 10)  # Strong positive signal
        
        # Top clearing preference
        score += self.strategy.topClearWeight * (row / ROWS)
        
        # Bottom avoidance
        score += self.strategy.bottomAvoidWeight * ((ROWS - row) / ROWS)
        
        # Color diversity (sample top rows for speed)
        color_count = 0
        sample_size = min(5, ROWS)
        for r in range(sample_size):
            for c in range(COLUMNS):
                if game_engine.grid[r][c] == color:
                    color_count += 1
        score += self.strategy.colorDiversityWeight * (color_count / (sample_size * COLUMNS))
        
        # Simulate move for cascade potential
        temp_grid = [row[:] for row in game_engine.grid]
        for r, c in connected_blocks:
            temp_grid[r][c] = None
        
        # Apply gravity
        temp_grid_after = [row[:] for row in temp_grid]
        for c in range(COLUMNS):
            write_idx = ROWS - 1
            for r in range(ROWS - 1, -1, -1):
                if temp_grid_after[r][c] is not None:
                    if write_idx != r:
                        temp_grid_after[write_idx][c] = temp_grid_after[r][c]
                        temp_grid_after[r][c] = None
                    write_idx -= 1
        
        # Check cascade potential (sample top rows)
        cascade_potential = 0
        cascade_sample = min(8, ROWS - 1)
        for r in range(cascade_sample):
            for c in range(COLUMNS):
                if temp_grid_after[r][c] is not None:
                    check_color = temp_grid_after[r][c]
                    if r > 0 and temp_grid_after[r - 1][c] == check_color:
                        cascade_potential += 1
                    if r < cascade_sample - 1 and temp_grid_after[r + 1][c] == check_color:
                        cascade_potential += 1
                    if c > 0 and temp_grid_after[r][c - 1] == check_color:
                        cascade_potential += 1
                    if c < COLUMNS - 1 and temp_grid_after[r][c + 1] == check_color:
                        cascade_potential += 1
        
        score += self.strategy.cascadePotentialWeight * (cascade_potential / 50)
        
        # Space creation
        empty_spaces = sum(1 for r in range(ROWS) for c in range(COLUMNS) if temp_grid_after[r][c] is None)
        score += self.strategy.spaceCreationWeight * (empty_spaces / (ROWS * COLUMNS))
        
        # NEW: Column height analysis - prefer clearing from taller columns
        column_heights = []
        for c in range(COLUMNS):
            height = sum(1 for r in range(ROWS) if game_engine.grid[r][c] is not None)
            column_heights.append(height)
        clicked_column_height = column_heights[col]
        score += self.strategy.columnHeightWeight * (clicked_column_height / ROWS)
        
        # NEW: Danger assessment - how close columns are to top
        max_danger = 0
        for c in range(COLUMNS):
            for r in range(ROWS):
                if game_engine.grid[r][c] is not None:
                    danger = (ROWS - r) / ROWS
                    max_danger = max(max_danger, danger)
                    break
        
        max_danger_after = 0
        for c in range(COLUMNS):
            for r in range(ROWS):
                if temp_grid_after[r][c] is not None:
                    danger = (ROWS - r) / ROWS
                    max_danger_after = max(max_danger_after, danger)
                    break
        
        danger_reduction = max_danger - max_danger_after
        score += self.strategy.dangerReductionWeight * danger_reduction
        
        # NEW: Isolated block detection (sample top 12 rows for speed, but always calculate)
        isolated_blocks = 0
        sample_rows = min(12, ROWS - 1)  # Sample more rows
        for r in range(sample_rows):
            for c in range(COLUMNS):
                if temp_grid_after[r][c] is not None:
                    block_color = temp_grid_after[r][c]
                    has_neighbor = False
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < sample_rows and 0 <= nc < COLUMNS:
                            if temp_grid_after[nr][nc] == block_color:
                                has_neighbor = True
                                break
                    if not has_neighbor:
                        isolated_blocks += 1
        score += self.strategy.isolationPenalty * (-isolated_blocks / (sample_rows * COLUMNS))
        
        # NEW: Color balance (sample top 14 rows for speed, but always calculate)
        color_counts = {color: 0 for color in COLORS}
        # Sample top 14 rows for speed (most important area)
        sample_rows = min(14, ROWS)
        for r in range(sample_rows):
            for c in range(COLUMNS):
                block_color = game_engine.grid[r][c]
                if block_color in color_counts:
                    color_counts[block_color] += 1
        
        color_values = list(color_counts.values())
        if color_values and sum(color_values) > 0:
            avg = sum(color_values) / len(color_values)
            variance = sum((val - avg) ** 2 for val in color_values) / len(color_values)
            balance_score = 1 / (1 + variance / 100)
            score += self.strategy.colorBalanceWeight * balance_score
        
        # NEW: Top row density reduction
        top_density_before = sum(1 for r in range(min(3, ROWS)) for c in range(COLUMNS) if game_engine.grid[r][c] is not None)
        top_density_after = sum(1 for r in range(min(3, ROWS)) for c in range(COLUMNS) if temp_grid_after[r][c] is not None)
        top_density_reduction = (top_density_before - top_density_after) / (3 * COLUMNS)
        score += self.strategy.topDensityWeight * top_density_reduction
        
        # NEW: Multi-cascade depth (simplified - count potential cascades)
        # OPTIMIZED: Only calculate if weight is significant, use depth 1 for speed
        if abs(self.strategy.multiCascadeWeight) > 0.1:  # Only if weight is significant
            cascade_depth = self._calculate_cascade_depth(temp_grid_after, 0, 1)  # Reduced from 2 to 1
            score += self.strategy.multiCascadeWeight * (cascade_depth / 10)
        
        # NEW: Column height balance - penalize moves that create large height differences
        # Calculate column heights before move
        column_heights_before = []
        for c in range(COLUMNS):
            height = sum(1 for r in range(ROWS) if game_engine.grid[r][c] is not None)
            column_heights_before.append(height)
        
        # Calculate column heights after move
        column_heights_after = []
        for c in range(COLUMNS):
            height = sum(1 for r in range(ROWS) if temp_grid_after[r][c] is not None)
            column_heights_after.append(height)
        
        # Calculate variance (standard deviation) of column heights
        def calculate_variance(heights):
            if not heights:
                return 0.0
            avg = sum(heights) / len(heights)
            variance = sum((h - avg) ** 2 for h in heights) / len(heights)
            return variance ** 0.5  # Standard deviation
        
        variance_before = calculate_variance(column_heights_before)
        variance_after = calculate_variance(column_heights_after)
        balance_improvement = (variance_before - variance_after) / ROWS  # Normalized
        score += self.strategy.columnBalanceWeight * balance_improvement * 10  # Scale up for impact
        
        # NEW: Group merging potential - detect if small elimination merges large groups
        # This rewards eliminating 1-2 blocks if it will merge two large groups
        if group_size <= 2:
            # Check if there are large groups of the same color after gravity
            # that are significantly larger than what we eliminated (indicating a merge)
            visited = set()
            largest_merged_group = 0
            
            # Find the largest group of the same color as what we eliminated
            for r in range(ROWS - 1):
                for c in range(COLUMNS):
                    if temp_grid_after[r][c] == color and (r, c) not in visited:
                        merged_group = self._find_connected_in_grid(temp_grid_after, r, c, color, visited)
                        if len(merged_group) > largest_merged_group:
                            largest_merged_group = len(merged_group)
            
            # If the merged group is much larger than what we eliminated, it's a good merge
            # Example: eliminate 2 blocks, create a 12-block group = merged two 5-block groups
            if largest_merged_group >= 10 and largest_merged_group > group_size * 3:
                # Strong bonus for creating a large merged group through small elimination
                # The larger the merged group relative to eliminated blocks, the better
                merging_bonus = min((largest_merged_group - group_size) / 15.0, 1.0)  # Normalized, capped at 1.0
                score += self.strategy.groupMergingWeight * merging_bonus * 25  # Very strong positive signal
        
        # NEW: Average blocks per turn - reward moves that eliminate 5+ blocks
        # Since 5 blocks are added per turn, we need to eliminate at least 5 on average
        if group_size >= 5:
            # Bonus for meeting the minimum requirement
            base_bonus = 1.0
            # Additional bonus for exceeding (8+ blocks gets extra reward)
            excess_bonus = (group_size - 5) / 10.0 if group_size >= 8 else 0.0
            score += self.strategy.averageBlocksWeight * (base_bonus + excess_bonus) * 10
        else:
            # Penalty for not meeting the requirement (scales with how far below 5)
            deficit = (5 - group_size) / 5.0  # 0.0 to 0.8 (for 1-4 blocks)
            score += self.strategy.averageBlocksWeight * (-deficit * 10)  # Strong negative signal
        
        # NEW STRATEGY 1: Max column reduction (prevents game over)
        # Reward moves that reduce the highest column - this is critical for survival
        max_height_before = max(column_heights_before) if column_heights_before else 0
        max_height_after = max(column_heights_after) if column_heights_after else 0
        max_column_reduction = (max_height_before - max_height_after) / ROWS  # Normalized
        # Strong reward - preventing game over is critical
        score += self.strategy.maxColumnReductionWeight * max_column_reduction * 20
        
        # NEW STRATEGY 2: Small clear enabler (reward small clears that enable large clears)
        # Check if a small clear (1-4 blocks) creates opportunities for large clears
        if group_size <= 4:
            # After this small clear, check for potential large groups
            # Count how many large groups (8+ blocks) exist after the move
            visited = set()
            large_groups_after = 0
            for r in range(ROWS - 1):
                for c in range(COLUMNS):
                    if temp_grid_after[r][c] is not None and (r, c) not in visited:
                        block_color = temp_grid_after[r][c]
                        group = self._find_connected_in_grid(temp_grid_after, r, c, block_color, visited)
                        if len(group) >= 8:  # Large group
                            large_groups_after += 1
            
            # Reward if small clear enables large groups
            if large_groups_after > 0:
                enabler_bonus = min(large_groups_after / 3.0, 1.0)  # Normalized, capped
                score += self.strategy.smallClearEnablerWeight * enabler_bonus * 15
        
        # NEW STRATEGY 3: Bottom-touching avoidance (don't clear groups touching bottom)
        # Check if this group touches the bottom row (ROWS - 1)
        touches_bottom = any(r == ROWS - 1 for r, c in connected_blocks)
        if touches_bottom:
            # Group touches bottom - avoid clearing it (it can grow when new row is added)
            # Penalty scales with group size (larger groups = bigger potential loss)
            if group_size >= 5:
                # Strong penalty for clearing large groups that touch bottom
                penalty = (group_size / 15.0)  # Normalized
                score += self.strategy.bottomTouchingAvoidanceWeight * (-penalty * 20)
            # Small groups (< 5) can still be cleared (not worth the penalty)
        else:
            # Group is isolated (doesn't touch bottom) - safe to eliminate
            # Small bonus for isolated groups (they won't grow)
            if group_size >= 5:
                isolation_bonus = (group_size / 20.0)  # Normalized
                score += self.strategy.bottomTouchingAvoidanceWeight * isolation_bonus * 5
        
        # NEW STRATEGY 4: Color concentration (reward clearing colors that concentrate remaining colors)
        # If we clear one color, the remaining colors become more concentrated
        # This makes future large clears more likely
        
        # Count colors before move (sample top rows for speed)
        color_counts_before = {c: 0 for c in COLORS}
        color_counts_after = {c: 0 for c in COLORS}
        sample_rows = min(12, ROWS)  # Sample top 12 rows
        
        for r in range(sample_rows):
            for c in range(COLUMNS):
                before_color = game_engine.grid[r][c]
                after_color = temp_grid_after[r][c]
                if before_color in color_counts_before:
                    color_counts_before[before_color] += 1
                if after_color in color_counts_after:
                    color_counts_after[after_color] += 1
        
        # Calculate concentration (variance) - higher variance = more concentrated
        def calculate_concentration(counts):
            values = [v for v in counts.values() if v > 0]
            if not values or len(values) < 2:
                return 0.0
            avg = sum(values) / len(values)
            variance = sum((v - avg) ** 2 for v in values) / len(values)
            return variance
        
        concentration_before = calculate_concentration(color_counts_before)
        concentration_after = calculate_concentration(color_counts_after)
        concentration_increase = (concentration_after - concentration_before) / 100.0  # Normalized
        
        # Reward moves that increase color concentration (makes large clears more likely)
        score += self.strategy.colorConcentrationWeight * concentration_increase * 10
        
        # NEW STRATEGY 5: Top row avoidance (avoid clearing blocks at very top)
        # Clear lower so larger groups may fall into place
        # Top 2 rows (rows 0-1) are the "very top"
        if row <= 1:
            # Strong penalty for clearing at very top
            top_penalty = (2 - row) / 2.0  # Row 0 = 1.0, Row 1 = 0.5
            score += self.strategy.topRowAvoidanceWeight * (-top_penalty * 15)
        else:
            # Small bonus for clearing lower (allows groups to fall)
            lower_bonus = (row / ROWS) * 0.3  # Normalized
            score += self.strategy.topRowAvoidanceWeight * lower_bonus * 5
        
        # NEW STRATEGY 6: Column isolation penalty (isolated tall columns are bad)
        # Calculate column heights after move
        column_heights_after = []
        for c in range(COLUMNS):
            height = sum(1 for r in range(ROWS) if temp_grid_after[r][c] is not None)
            column_heights_after.append(height)
        
        if column_heights_after:
            avg_height = sum(column_heights_after) / len(column_heights_after)
            clicked_col_height = column_heights_after[col]
            
            # Check if clicked column is isolated and taller than neighbors
            neighbor_heights = []
            if col > 0:
                neighbor_heights.append(column_heights_after[col - 1])
            if col < COLUMNS - 1:
                neighbor_heights.append(column_heights_after[col + 1])
            
            if neighbor_heights:
                avg_neighbor_height = sum(neighbor_heights) / len(neighbor_heights)
                # If this column is significantly taller than neighbors, it's isolated
                if clicked_col_height > avg_neighbor_height + 2:
                    isolation_penalty = (clicked_col_height - avg_neighbor_height) / ROWS
                    score += self.strategy.columnIsolationPenaltyWeight * (-isolation_penalty * 20)
        
        # NEW STRATEGY 7: Critical column focus (MUST focus on columns right below max)
        # If any column is at ROWS-1 (one below max), we're about to lose
        # MUST focus on reducing that column
        max_height = max(column_heights_after) if column_heights_after else 0
        critical_threshold = ROWS - 1  # One below max (ROWS is max)
        
        if max_height >= critical_threshold:
            # We're in critical danger - MUST focus on the tallest column
            clicked_col_height = column_heights_after[col] if column_heights_after else 0
            if clicked_col_height >= critical_threshold:
                # This move targets a critical column - HUGE reward
                critical_bonus = 1.0 + (clicked_col_height - critical_threshold) * 0.5
                score += self.strategy.criticalColumnFocusWeight * critical_bonus * 50  # VERY strong
            else:
                # Not targeting critical column - penalty
                score += self.strategy.criticalColumnFocusWeight * (-0.5 * 30)  # Strong penalty
        
        # NEW STRATEGY 8: Side connection requirement (avoid blocks with no side connections)
        # Blocks with no side connections are less flexible - focus lower for more possibilities
        has_left_connection = False
        has_right_connection = False
        
        # Check if any block in the group has side connections
        for r, c in connected_blocks:
            # Check left
            if c > 0:
                left_color = game_engine.grid[r][c - 1]
                if left_color is not None and left_color != color:
                    has_left_connection = True
            # Check right
            if c < COLUMNS - 1:
                right_color = game_engine.grid[r][c + 1]
                if right_color is not None and right_color != color:
                    has_right_connection = True
        
        if not (has_left_connection or has_right_connection):
            # No side connections - this is less flexible, penalize it
            # Penalty is stronger for larger groups (waste of opportunity)
            side_penalty = (group_size / 15.0)  # Normalized
            score += self.strategy.sideConnectionRequirementWeight * (-side_penalty * 15)
        else:
            # Has side connections - bonus (more flexible, better for column balance)
            side_bonus = 0.3 if (has_left_connection and has_right_connection) else 0.15
            score += self.strategy.sideConnectionRequirementWeight * side_bonus * 5
        
        return score
    
    def _calculate_cascade_depth(self, grid, depth, max_depth):
        """Calculate potential cascade depth recursively"""
        if depth >= max_depth:
            return depth
        
        # Find all potential cascade groups
        visited = set()
        cascade_groups = []
        
        for r in range(ROWS - 1):
            for c in range(COLUMNS):
                if grid[r][c] is not None and (r, c) not in visited:
                    color = grid[r][c]
                    group = self._find_connected_in_grid(grid, r, c, color, visited)
                    if len(group) >= 2:
                        cascade_groups.append(group)
        
        if not cascade_groups:
            return depth
        
        # Simulate largest cascade
        largest_group = max(cascade_groups, key=len)
        new_grid = [row[:] for row in grid]
        for r, c in largest_group:
            new_grid[r][c] = None
        
        # Apply gravity
        for c in range(COLUMNS):
            write_idx = ROWS - 1
            for r in range(ROWS - 1, -1, -1):
                if new_grid[r][c] is not None:
                    if write_idx != r:
                        new_grid[write_idx][c] = new_grid[r][c]
                        new_grid[r][c] = None
                    write_idx -= 1
        
        return self._calculate_cascade_depth(new_grid, depth + 1, max_depth)
    
    def _find_connected_in_grid(self, grid, row, col, color, visited):
        """Find connected blocks in a given grid"""
        if (row, col) in visited:
            return []
        if row < 0 or row >= ROWS or col < 0 or col >= COLUMNS:
            return []
        if grid[row][col] != color:
            return []
        
        visited.add((row, col))
        connected = [(row, col)]
        
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            connected.extend(self._find_connected_in_grid(grid, row + dr, col + dc, color, visited))
        
        return connected

    def choose_action(self, game_engine: GameEngine) -> Optional[Tuple[int, int]]:
        """Choose best action based on strategy"""
        valid_actions = game_engine.get_valid_actions()
        if not valid_actions:
            return None
        
        # Pre-filter: only evaluate actions with neighbors
        promising_actions = []
        for row, col in valid_actions:
            color = game_engine.grid[row][col]
            neighbor_count = 0
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = row + dr, col + dc
                if 0 <= nr < ROWS - 1 and 0 <= nc < COLUMNS:
                    if game_engine.grid[nr][nc] == color:
                        neighbor_count += 1
                        break
            if neighbor_count > 0:
                promising_actions.append((row, col))
        
        # OPTIMIZED: Limit to 12 actions for speed (was 15)
        actions_to_evaluate = (promising_actions if promising_actions else valid_actions[:10])[:12]
        
        best_action = actions_to_evaluate[0]
        best_score = float('-inf')
        
        for action in actions_to_evaluate:
            score = self.evaluate_action(game_engine, action[0], action[1])
            if score > best_score:
                best_score = score
                best_action = action
        
        return best_action

    def play_game(self, game_engine: Optional[GameEngine] = None) -> Dict:
        """Play a game and return results"""
        engine = game_engine if game_engine else GameEngine()
        engine.reset()
        
        steps = 0
        max_steps = 1000
        
        while not engine.game_over and steps < max_steps:
            action = self.choose_action(engine)
            if not action:
                break
            
            result = engine.act(action[0], action[1])
            if not result['success']:
                break
            
            steps += 1
        
        self.games_played += 1
        self.total_score += engine.score
        # Improved fitness: weighted combination of average and best score
        # This rewards both consistency and peak performance
        avg_score = self.total_score / self.games_played
        self.best_score = max(self.best_score, engine.score)
        # Fitness = 70% average + 30% best (encourages both consistency and peaks)
        self.fitness = 0.7 * avg_score + 0.3 * self.best_score
        
        return {
            'score': engine.score,
            'steps': steps,
            'fitness': self.fitness
        }

    def reset_fitness(self):
        """Reset fitness tracking"""
        self.fitness = 0.0
        self.games_played = 0
        self.total_score = 0
        self.best_score = 0


def evaluate_agent_worker(args: Tuple) -> Tuple[int, float, int, int]:
    """Evaluate an agent (for parallel processing) - returns serializable data"""
    import pickle
    agent_data, games_per_eval = args
    
    # Reconstruct agent from data
    strategy = Strategy(**agent_data['strategy'])
    agent = GeneticAgent(strategy)
    agent.reset_fitness()
    engine = GameEngine()
    
    for _ in range(games_per_eval):
        agent.play_game(engine)
    
    # Return serializable results
    return (agent.fitness, agent.best_score, agent.games_played, agent.total_score)


class GeneticAlgorithm:
    """Genetic algorithm trainer"""
    
    def __init__(
        self,
        population_size: int = 75,
        mutation_rate: float = 0.15,
        crossover_rate: float = 0.7,
        elite_size: int = 7,  # Increased for better preservation
        games_per_evaluation: int = 5,  # Increased for better fitness estimates
        output_file: str = 'crazyblocks-strategies.json',
        use_parallel: bool = True,
        save_freq: int = 5  # Save every N generations (for speed)
    ):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        self.games_per_evaluation = games_per_evaluation
        self.output_file = output_file
        self.use_parallel = use_parallel and mp.cpu_count() > 1
        self.save_freq = save_freq
        self.generations_since_save = 0
        self.exploration_threshold = 500.0  # Average score threshold for exploration
        self.exploration_boost = 1.0  # Multiplier for mutation/diversity when exploring
        
        self.population: List[GeneticAgent] = []
        self.generation = 0
        self.history = []
        
        # Load existing or initialize new
        if not self.load_population():
            self.initialize_population()

    def initialize_population(self):
        """Initialize random population"""
        self.population = [GeneticAgent() for _ in range(self.population_size)]

    def load_population(self) -> bool:
        """Load population from JSON file"""
        try:
            if not Path(self.output_file).exists():
                return False
            
            with open(self.output_file, 'r') as f:
                data = json.load(f)
            
            # Load strategies
            strategies_data = data.get('fullPopulation', data.get('eliteStrategies', data.get('population', [])))
            if not strategies_data:
                return False
            
            self.population = []
            for agent_data in strategies_data:
                strategy_dict = agent_data.get('strategy', {})
                # Add default values for new heuristics if missing (backward compatibility)
                defaults = {
                    'columnHeightWeight': 0.0,
                    'dangerReductionWeight': 0.0,
                    'colorBalanceWeight': 0.0,
                    'topDensityWeight': 0.0,
                    'multiCascadeWeight': 0.0,
                    'minimumGroupSizeWeight': 0.0,
                    'largeGroupBonusWeight': 0.0,
                    'columnBalanceWeight': 0.0,
                    'groupMergingWeight': 0.0,
                    'averageBlocksWeight': 0.0,
                    'maxColumnReductionWeight': 0.0,
                    'smallClearEnablerWeight': 0.0,
                    'bottomTouchingAvoidanceWeight': 0.0,
                    'colorConcentrationWeight': 0.0,
                    'topRowAvoidanceWeight': 0.0,
                    'columnIsolationPenaltyWeight': 0.0,
                    'criticalColumnFocusWeight': 0.0,
                    'sideConnectionRequirementWeight': 0.0
                }
                for key, default_val in defaults.items():
                    if key not in strategy_dict:
                        strategy_dict[key] = default_val
                strategy = Strategy(**strategy_dict)
                agent = GeneticAgent(strategy)
                agent.fitness = agent_data.get('fitness', 0)
                agent.best_score = agent_data.get('bestScore', 0)
                agent.games_played = agent_data.get('gamesPlayed', 0)
                self.population.append(agent)
            
            # Fill to population size
            while len(self.population) < self.population_size:
                self.population.append(GeneticAgent())
            
            # Trim if too many
            if len(self.population) > self.population_size:
                self.population = self.population[:self.population_size]
            
            self.generation = data.get('generation', 0)
            print(f"Loaded {len(self.population)} strategies from generation {self.generation}")
            return True
            
        except Exception as e:
            print(f"Failed to load: {e}")
            return False

    def evaluate_population(self):
        """Evaluate fitness of all agents"""
        if self.use_parallel:
            # Parallel evaluation - use serializable data
            try:
                num_workers = min(mp.cpu_count(), self.population_size, 8)  # Limit workers
                agent_data_list = [
                    {
                        'strategy': asdict(agent.strategy),
                        'index': i
                    }
                    for i, agent in enumerate(self.population)
                ]
                
                with ProcessPoolExecutor(max_workers=num_workers) as executor:
                    futures = {
                        executor.submit(evaluate_agent_worker, (data, self.games_per_evaluation)): data['index']
                        for data in agent_data_list
                    }
                    
                    results = {}
                    for future in as_completed(futures):
                        idx = futures[future]
                        fitness, best_score, games_played, total_score = future.result()
                        results[idx] = (fitness, best_score, games_played, total_score)
                
                # Update agents with results
                for i, agent in enumerate(self.population):
                    if i in results:
                        fitness, best_score, games_played, total_score = results[i]
                        agent.fitness = fitness
                        agent.best_score = best_score
                        agent.games_played = games_played
                        agent.total_score = total_score
            except Exception as e:
                print(f"⚠️  Parallel processing failed: {e}")
                print("   Falling back to sequential processing...")
                self.use_parallel = False
                # Fall through to sequential
                engine = GameEngine()
                for agent in self.population:
                    agent.reset_fitness()
                    for _ in range(self.games_per_evaluation):
                        agent.play_game(engine)
        else:
            # Sequential evaluation
            engine = GameEngine()
            for agent in self.population:
                agent.reset_fitness()
                for _ in range(self.games_per_evaluation):
                    agent.play_game(engine)
        
        # Sort by fitness
        self.population.sort(key=lambda a: a.fitness, reverse=True)

    def tournament_selection(self, tournament_size: int = 3) -> GeneticAgent:
        """Select parent using tournament selection"""
        tournament = random.sample(self.population, min(tournament_size, len(self.population)))
        return max(tournament, key=lambda a: a.fitness)

    def create_next_generation(self, avg_fitness: float = 0.0):
        """Create next generation with adaptive exploration"""
        # Calculate exploration boost based on average fitness
        if avg_fitness < self.exploration_threshold:
            # Below threshold - increase exploration
            # Boost scales from 1.0 (at threshold) to 3.0 (at 0)
            fitness_ratio = max(0, avg_fitness / self.exploration_threshold)
            self.exploration_boost = 1.0 + (1.0 - fitness_ratio) * 2.0  # 1.0 to 3.0
        else:
            # Above threshold - normal exploration
            self.exploration_boost = 1.0
        
        new_population = []
        
        # Keep elite (fewer if exploring heavily)
        elite_to_keep = self.elite_size
        if self.exploration_boost > 1.5:
            # Reduce elite when heavily exploring
            elite_to_keep = max(2, int(self.elite_size * 0.5))
        
        for i in range(elite_to_keep):
            elite = GeneticAgent(Strategy(**asdict(self.population[i].strategy)))
            elite.fitness = self.population[i].fitness
            elite.best_score = self.population[i].best_score
            new_population.append(elite)
        
        # Add random diversity (more if exploring)
        base_diversity = 0.05
        if self.exploration_boost > 1.5:
            # Increase diversity when exploring
            base_diversity = 0.15  # 15% random diversity
        num_random = max(1, int(self.population_size * base_diversity))
        for _ in range(num_random):
            new_population.append(GeneticAgent())
        
        # Fill rest with crossover and mutation (with exploration boost)
        effective_mutation_rate = self.mutation_rate * self.exploration_boost
        effective_mutation_rate = min(effective_mutation_rate, 0.5)  # Cap at 50%
        
        while len(new_population) < self.population_size:
            if random.random() < self.crossover_rate and len(new_population) < self.population_size - 1:
                # Crossover
                parent1 = self.tournament_selection()
                parent2 = self.tournament_selection()
                child_strategy = parent1.strategy.crossover(parent2.strategy)
                new_population.append(GeneticAgent(child_strategy.mutate(effective_mutation_rate)))
            else:
                # Mutation only
                parent = self.tournament_selection()
                new_population.append(GeneticAgent(parent.strategy.mutate(effective_mutation_rate)))
        
        self.population = new_population
        self.generation += 1

    def run_generation(self) -> Dict:
        """Run one generation"""
        start_time = time.time()
        self.evaluate_population()
        
        best_agent = self.population[0]
        avg_fitness = sum(a.fitness for a in self.population) / len(self.population)
        
        elapsed = time.time() - start_time
        
        # Pass avg_fitness to create_next_generation for adaptive exploration
        self.create_next_generation(avg_fitness)
        
        # Save periodically (not every generation for speed)
        self.generations_since_save += 1
        should_save = (self.generations_since_save >= self.save_freq)
        if should_save:
            self.save_population()
            self.generations_since_save = 0
        
        return {
            'generation': self.generation - 1,
            'best_fitness': best_agent.fitness,
            'best_score': best_agent.best_score,
            'avg_fitness': avg_fitness,
            'elapsed_time': elapsed,
            'saved': should_save,
            'exploration_boost': self.exploration_boost
        }

    def save_population(self):
        """Save population to JSON file"""
        self.population.sort(key=lambda a: a.fitness, reverse=True)
        
        data = {
            'version': '1.0',
            'generation': self.generation,
            'timestamp': int(time.time() * 1000),
            'bestFitness': self.population[0].fitness,
            'bestScore': self.population[0].best_score,
            'populationSize': len(self.population),
            'eliteStrategies': [
                {
                    'strategy': asdict(agent.strategy),
                    'fitness': agent.fitness,
                    'bestScore': agent.best_score,
                    'gamesPlayed': agent.games_played
                }
                for agent in self.population[:self.elite_size]
            ],
            'fullPopulation': [
                {
                    'strategy': asdict(agent.strategy),
                    'fitness': agent.fitness,
                    'bestScore': agent.best_score,
                    'gamesPlayed': agent.games_played
                }
                for agent in self.population
            ]
        }
        
        with open(self.output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        # Also copy to public folder for web interface
        public_file = os.path.join('public', os.path.basename(self.output_file))
        if os.path.exists('public'):
            with open(public_file, 'w') as f:
                json.dump(data, f, indent=2)
            if self.save_freq > 1:
                print(f"Saved to {self.output_file} and synced to {public_file}")
        else:
            if self.save_freq > 1:
                print(f"Saved to {self.output_file}")

    def get_stats(self) -> Dict:
        """Get current statistics"""
        if not self.population:
            return {
                'generation': 0,
                'best_fitness': 0,
                'best_score': 0,
                'avg_fitness': 0
            }
        
        self.population.sort(key=lambda a: a.fitness, reverse=True)
        best_agent = self.population[0]
        avg_fitness = sum(a.fitness for a in self.population) / len(self.population)
        
        return {
            'generation': self.generation,
            'best_fitness': best_agent.fitness,
            'best_score': best_agent.best_score,
            'avg_fitness': avg_fitness,
            'population_size': len(self.population)
        }


def main():
    parser = argparse.ArgumentParser(description='Train Crazy Blocks AI using Genetic Algorithm')
    parser.add_argument('--generations', type=int, default=None, help='Number of generations to train (default: unlimited - runs until Ctrl+C)')
    parser.add_argument('--population', type=int, default=50, help='Population size')
    parser.add_argument('--games', type=int, default=3, help='Games per evaluation (fewer = faster, default: 3 for speed)')
    parser.add_argument('--output', type=str, default='crazyblocks-strategies.json', help='Output JSON file')
    parser.add_argument('--no-parallel', action='store_true', help='Disable parallel processing')
    parser.add_argument('--mutation-rate', type=float, default=0.15, help='Mutation rate')
    parser.add_argument('--crossover-rate', type=float, default=0.7, help='Crossover rate')
    parser.add_argument('--save-freq', type=int, default=5, help='Save every N generations (default: 5, set to 1 for every generation)')
    parser.add_argument('--exploration-threshold', type=float, default=500.0, help='Average score threshold - explores heavily below this (default: 500)')
    
    args = parser.parse_args()
    
    print("Crazy Blocks AI Trainer (Python)")
    print("=" * 50)
    
    ga = GeneticAlgorithm(
        population_size=args.population,
        mutation_rate=args.mutation_rate,
        crossover_rate=args.crossover_rate,
        games_per_evaluation=args.games,
        output_file=args.output,
        use_parallel=not args.no_parallel,
        save_freq=args.save_freq
    )
    # Set exploration threshold
    ga.exploration_threshold = args.exploration_threshold
    
    start_generation = ga.generation
    
    if start_generation > 0:
        print(f"Resuming from generation {start_generation}")
        best_agent = max(ga.population, key=lambda a: a.fitness)
        print(f"Previous best: Score {best_agent.best_score}, Fitness {best_agent.fitness:.2f}")
    else:
        print(f"Starting fresh training")
    
    if args.generations:
        target_generation = start_generation + args.generations
        print(f"Will train {args.generations} more generations")
        print(f"Target: generation {target_generation} (currently at {start_generation})")
    else:
        print(f"Training continuously (Press Ctrl+C to stop)")
        target_generation = None
    
    print(f"Population: {args.population}, Games per eval: {args.games}")
    print(f"Parallel processing: {'Enabled' if ga.use_parallel else 'Disabled'}")
    print(f"Saving every {args.save_freq} generation(s)")
    if target_generation:
        print(f"Will train from generation {start_generation} to {target_generation}")
    print("=" * 50)
    print()
    
    start_time = time.time()
    generations_trained = 0
    
    try:
        while True:
            # Check if we've reached target generation (if specified)
            # We check BEFORE running because run_generation increments the counter
            if target_generation is not None:
                if ga.generation >= target_generation:
                    print(f"\nReached target generation {target_generation}")
                    break
            
            # Run one generation (this increments ga.generation)
            result = ga.run_generation()
            current_gen = result['generation']
            generations_trained += 1
            
            save_indicator = " [SAVED]" if result.get('saved', False) else ""
            exploration_indicator = ""
            if result['avg_fitness'] < args.exploration_threshold:
                boost = result.get('exploration_boost', 1.0)
                exploration_indicator = f" [EXPLORING x{boost:.1f}]"
            
            print(
                f"Gen {current_gen:4d} | "
                f"Best: {result['best_score']:6.0f} "
                f"(Fitness: {result['best_fitness']:7.2f}) | "
                f"Avg: {result['avg_fitness']:7.2f} | "
                f"Time: {result['elapsed_time']:5.2f}s{exploration_indicator}{save_indicator}"
            )
            
            # If no target, continue forever. If target specified, loop will break when reached.
            
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user (Ctrl+C)")
        # Save one more time to be safe
        ga.save_population()
        print("Progress saved!")
    
    total_time = time.time() - start_time
    stats = ga.get_stats()
    
    print("=" * 50)
    print("Training stopped!")
    print(f"Final Generation: {stats['generation']}")
    print(f"Best Score: {stats['best_score']}")
    print(f"Best Fitness: {stats['best_fitness']:.2f}")
    print(f"Total Time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
    print(f"Saved to: {args.output}")
    if start_generation > 0:
        print(f"Trained {generations_trained} new generations (from {start_generation} to {stats['generation']})")
    print("\nYou can import this file into the JavaScript app!")
    print("Run again to continue training from where you left off!")


if __name__ == '__main__':
    # Required for Windows multiprocessing
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        # Already set, ignore
        pass
    main()

