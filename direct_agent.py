#!/usr/bin/env python3
"""
Direct Playing Agent for Crazy Blocks
Uses strategic heuristics based on expert insights - no training needed!
"""

from train_ai import GameEngine, ROWS, COLUMNS, COLORS
from typing import Optional, Tuple, List
import math
import random

class DirectAgent:
    """
    Advanced Direct Playing Agent with sophisticated features
    Based on expert insights with look-ahead and advanced planning
    Can use evolved weights from genetic algorithm or default hand-tuned weights
    """
    
    def __init__(self, look_ahead_depth: int = 0, weights: dict = None):
        # Look-ahead disabled - new blocks are random so it's not useful
        self.look_ahead_depth = 0
        
        # Strategic weights - use provided or default hand-tuned values
        if weights:
            self.weights = weights
        else:
            # Default hand-tuned weights (good starting point)
            self.weights = {
                # Core priorities
                'score_priority': 100.0,      # Score is everything
                'min_blocks': 40.0,          # Must eliminate 5+ blocks (critical!)
                'column_balance': 60.0,      # Keep columns similar height (your key insight)
                
                # Group strategies
                'large_groups': 25.0,        # Prefer large eliminations
                'group_merging': 30.0,       # Reward merging groups strategically
                'cascade_potential': 20.0,   # Prefer moves that create cascades
                'cascade_depth': 15.0,       # Reward deeper cascades
                
                # Survival and danger
                'danger_reduction': 25.0,    # Reduce columns near top
                'top_row_clearance': 20.0,   # Clear blocks from top rows
                'survival_bonus': 10.0,      # Bonus for staying alive longer
                
                # Strategic planning
                'space_creation': 15.0,      # Create empty spaces for future moves
                'color_balance': 12.0,       # Balance colors across board
                'isolated_penalty': -10.0,   # Penalty for leaving isolated blocks
                
                # NEW: Bottom-touching avoidance / Isolated group preference
                'bottom_touching_penalty': -35.0,  # Avoid clearing groups touching bottom (they can grow!)
                'isolated_group_bonus': 30.0,     # Reward clearing large isolated groups (safe to eliminate)
                
            # Look-ahead disabled (new blocks are random)
            'look_ahead_bonus': 0.0,     # Not used
            }
    
    def evaluate_move(self, game_engine: GameEngine, row: int, col: int, depth: int = 0) -> float:
        """Evaluate a move and return a score (with optional look-ahead)"""
        if row == ROWS - 1 or game_engine.grid[row][col] is None:
            return float('-inf')
        
        color = game_engine.grid[row][col]
        connected_blocks = game_engine.find_connected_blocks(row, col, color)
        group_size = len(connected_blocks)
        
        if group_size == 0:
            return float('-inf')
        
        score = 0.0
        
        # 1. SCORE PRIORITY (most important)
        # Direct score contribution
        score += self.weights['score_priority'] * group_size
        
        # 2. MINIMUM BLOCKS REQUIREMENT (critical)
        # Must eliminate 5+ blocks to break even
        if group_size >= 5:
            score += self.weights['min_blocks'] * 2.0  # Strong bonus
            if group_size > 5:
                score += self.weights['min_blocks'] * (group_size - 5) * 0.5  # Extra for exceeding
        else:
            # Strong penalty for not meeting requirement
            penalty = (5 - group_size) * self.weights['min_blocks']
            score -= penalty
        
        # 3. COLUMN BALANCE (your key insight)
        # Calculate column heights before move
        column_heights_before = [
            sum(1 for r in range(ROWS) if game_engine.grid[r][c] is not None)
            for c in range(COLUMNS)
        ]
        
        # Simulate move
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
        
        # Calculate column heights after
        column_heights_after = [
            sum(1 for r in range(ROWS) if temp_grid_after[r][c] is not None)
            for c in range(COLUMNS)
        ]
        
        # Calculate variance (lower = more balanced)
        if column_heights_after:
            avg_before = sum(column_heights_before) / len(column_heights_before)
            avg_after = sum(column_heights_after) / len(column_heights_after)
            variance_before = sum((h - avg_before) ** 2 for h in column_heights_before) / len(column_heights_before)
            variance_after = sum((h - avg_after) ** 2 for h in column_heights_after) / len(column_heights_after)
            std_dev_before = math.sqrt(variance_before)
            std_dev_after = math.sqrt(variance_after)
            
            # Reward moves that improve balance (reduce variance)
            balance_improvement = (std_dev_before - std_dev_after) / ROWS
            score += self.weights['column_balance'] * balance_improvement * 10
        
        # 4. LARGE GROUPS (prefer big eliminations)
        if group_size >= 10:
            score += self.weights['large_groups'] * 3.0
        elif group_size >= 8:
            score += self.weights['large_groups'] * 2.0
        elif group_size >= 6:
            score += self.weights['large_groups'] * 1.0
        
        # 5. GROUP MERGING (strategic small eliminations)
        # Check if eliminating small group creates large merged group
        if group_size <= 2:
            # Find largest group of same color after move
            visited = set()
            largest_merged = 0
            for r in range(ROWS - 1):
                for c in range(COLUMNS):
                    if temp_grid_after[r][c] == color and (r, c) not in visited:
                        merged_group = self._find_connected(temp_grid_after, r, c, color, visited)
                        if len(merged_group) > largest_merged:
                            largest_merged = len(merged_group)
            
            # If merged group is much larger, it's a good strategic move
            if largest_merged >= 10 and largest_merged > group_size * 3:
                merging_bonus = (largest_merged - group_size) / 10.0
                score += self.weights['group_merging'] * merging_bonus * 5
        
        # 6. CASCADE POTENTIAL
        # Check for potential cascades after move
        cascade_count = 0
        for r in range(min(8, ROWS - 1)):
            for c in range(COLUMNS):
                if temp_grid_after[r][c] is not None:
                    check_color = temp_grid_after[r][c]
                    # Count adjacent same-color blocks
                    if r > 0 and temp_grid_after[r - 1][c] == check_color:
                        cascade_count += 1
                    if r < ROWS - 2 and temp_grid_after[r + 1][c] == check_color:
                        cascade_count += 1
                    if c > 0 and temp_grid_after[r][c - 1] == check_color:
                        cascade_count += 1
                    if c < COLUMNS - 1 and temp_grid_after[r][c + 1] == check_color:
                        cascade_count += 1
        
        score += self.weights['cascade_potential'] * (cascade_count / 50.0)
        
        # 7. DANGER REDUCTION (reduce columns near top)
        # Find highest column before
        max_height_before = max(column_heights_before) if column_heights_before else 0
        max_height_after = max(column_heights_after) if column_heights_after else 0
        danger_reduction = (max_height_before - max_height_after) / ROWS
        score += self.weights['danger_reduction'] * danger_reduction * 5
        
        # 8. TOP ROW CLEARANCE (critical for survival)
        # Count blocks in top 3 rows before and after
        top_rows_before = sum(1 for r in range(min(3, ROWS)) for c in range(COLUMNS) 
                             if game_engine.grid[r][c] is not None)
        top_rows_after = sum(1 for r in range(min(3, ROWS)) for c in range(COLUMNS) 
                            if temp_grid_after[r][c] is not None)
        top_clearance = (top_rows_before - top_rows_after) / (3 * COLUMNS)
        score += self.weights['top_row_clearance'] * top_clearance * 10
        
        # 9. CASCADE DEPTH (simulate actual cascades) - OPTIMIZED: limit depth for speed
        cascade_depth = self._simulate_cascade_depth(temp_grid_after, 0, 2)  # Reduced from 3 to 2
        score += self.weights['cascade_depth'] * (cascade_depth / 10.0)
        
        # 10. SPACE CREATION (empty spaces for future moves)
        empty_spaces_before = sum(1 for r in range(ROWS) for c in range(COLUMNS) 
                                  if game_engine.grid[r][c] is None)
        empty_spaces_after = sum(1 for r in range(ROWS) for c in range(COLUMNS) 
                                 if temp_grid_after[r][c] is None)
        space_created = (empty_spaces_after - empty_spaces_before) / (ROWS * COLUMNS)
        score += self.weights['space_creation'] * space_created * 5
        
        # 11. COLOR BALANCE (balance colors across board)
        color_counts_before = {c: 0 for c in COLORS}
        color_counts_after = {c: 0 for c in COLORS}
        
        for r in range(ROWS):
            for c in range(COLUMNS):
                if game_engine.grid[r][c] in color_counts_before:
                    color_counts_before[game_engine.grid[r][c]] += 1
                if temp_grid_after[r][c] in color_counts_after:
                    color_counts_after[temp_grid_after[r][c]] += 1
        
        # Calculate variance (lower = more balanced)
        if color_counts_after:
            avg_before = sum(color_counts_before.values()) / len(color_counts_before)
            avg_after = sum(color_counts_after.values()) / len(color_counts_after)
            variance_before = sum((v - avg_before) ** 2 for v in color_counts_before.values()) / len(color_counts_before)
            variance_after = sum((v - avg_after) ** 2 for v in color_counts_after.values()) / len(color_counts_after)
            color_balance_improvement = (variance_before - variance_after) / 100.0
            score += self.weights['color_balance'] * color_balance_improvement
        
        # 12. ISOLATED BLOCKS PENALTY (blocks with no same-color neighbors)
        isolated_count = 0
        for r in range(ROWS - 1):
            for c in range(COLUMNS):
                if temp_grid_after[r][c] is not None:
                    block_color = temp_grid_after[r][c]
                    has_neighbor = False
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < ROWS - 1 and 0 <= nc < COLUMNS:
                            if temp_grid_after[nr][nc] == block_color:
                                has_neighbor = True
                                break
                    if not has_neighbor:
                        isolated_count += 1
        
        score += self.weights['isolated_penalty'] * (isolated_count / (ROWS * COLUMNS)) * 10
        
        # 13. BOTTOM-TOUCHING AVOIDANCE / ISOLATED GROUP PREFERENCE (your new strategy!)
        # Check if this group touches the bottom row (ROWS - 1)
        # Bottom row is ROWS - 1 (can't click it, but blocks can be there)
        # We check if any block in the group is in the bottom row
        touches_bottom = any(r == ROWS - 1 for r, c in connected_blocks)
        
        if touches_bottom:
            # Group touches bottom - AVOID clearing it (it can grow when new row is added)
            # Penalty scales with group size (larger groups = bigger potential loss)
            # But allow small groups to be cleared (they're not worth saving)
            if group_size >= 5:
                # Strong penalty for clearing large groups that touch bottom
                penalty = (group_size / 10.0)  # Scale penalty with size
                score += self.weights['bottom_touching_penalty'] * penalty * 2.0
            # Small groups (< 5) can still be cleared (not worth the penalty)
        else:
            # Group is isolated (doesn't touch bottom) - SAFE to eliminate
            # Reward clearing large isolated groups (they won't grow)
            if group_size >= 8:
                # Strong bonus for large isolated groups
                bonus = (group_size / 15.0)  # Scale bonus with size
                score += self.weights['isolated_group_bonus'] * bonus * 2.0
            elif group_size >= 5:
                # Moderate bonus for medium isolated groups
                bonus = (group_size / 10.0)
                score += self.weights['isolated_group_bonus'] * bonus
        
        # 14. LOOK-AHEAD (disabled - new blocks are random, so look-ahead isn't useful)
        # We can't predict what blocks will appear, so evaluating future moves doesn't help
        if False:  # Permanently disabled
            # Simulate this move properly
            simulated_engine = GameEngine()
            simulated_engine.grid = [row[:] for row in temp_grid_after]
            simulated_engine.score = game_engine.score + group_size
            simulated_engine.turn_count = game_engine.turn_count + 1
            simulated_engine.game_over = game_engine.game_over
            
            # Add new row (simulate turn progression)
            if not simulated_engine.game_over:
                new_row = [random.choice(COLORS) for _ in range(COLUMNS)]
                simulated_engine.grid = simulated_engine.grid[1:] + [new_row]
                if any(simulated_engine.grid[0]):
                    simulated_engine.game_over = True
            
            # Find best next move (limit to avoid slowdown)
            best_next_score = float('-inf')
            next_actions = simulated_engine.get_valid_actions()
            if next_actions:
                # Only evaluate top 5 moves for speed
                for next_row, next_col in next_actions[:5]:
                    try:
                        next_score = self.evaluate_move(simulated_engine, next_row, next_col, depth + 1)
                        if next_score > best_next_score:
                            best_next_score = next_score
                    except:
                        pass  # Skip if evaluation fails
            
            # Add look-ahead bonus (discounted by depth)
            if best_next_score > float('-inf'):
                look_ahead_bonus = best_next_score * self.weights['look_ahead_bonus'] * (0.7 ** depth)
                score += look_ahead_bonus
        
        return score
    
    def _find_connected(self, grid, row, col, color, visited):
        """Find connected blocks of same color"""
        if (row, col) in visited:
            return []
        if row < 0 or row >= ROWS or col < 0 or col >= COLUMNS:
            return []
        if grid[row][col] != color:
            return []
        
        visited.add((row, col))
        connected = [(row, col)]
        
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            connected.extend(self._find_connected(grid, row + dr, col + dc, color, visited))
        
        return connected
    
    def _simulate_cascade_depth(self, grid, current_depth: int, max_depth: int) -> int:
        """Simulate cascades recursively to find cascade depth"""
        if current_depth >= max_depth:
            return current_depth
        
        # Find all potential cascade groups (size >= 2)
        visited = set()
        cascade_groups = []
        
        for r in range(ROWS - 1):
            for c in range(COLUMNS):
                if grid[r][c] is not None and (r, c) not in visited:
                    color = grid[r][c]
                    group = self._find_connected(grid, r, c, color, visited)
                    if len(group) >= 2:
                        cascade_groups.append(group)
        
        if not cascade_groups:
            return current_depth
        
        # Simulate the largest cascade
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
        
        # Recursively check for more cascades
        return self._simulate_cascade_depth(new_grid, current_depth + 1, max_depth)
    
    def choose_action(self, game_engine: GameEngine) -> Optional[Tuple[int, int]]:
        """Choose the best action with sophisticated evaluation"""
        valid_actions = game_engine.get_valid_actions()
        if not valid_actions:
            return None
        
        # Pre-filter: Only consider moves that eliminate 2+ blocks (quick filter)
        promising_actions = []
        for row, col in valid_actions:
            color = game_engine.grid[row][col]
            if color:
                connected = game_engine.find_connected_blocks(row, col, color)
                if len(connected) >= 2:  # At least 2 blocks
                    promising_actions.append((row, col))
        
        # If we have promising actions, use those; otherwise use all
        # OPTIMIZED: Limit actions evaluated for speed (was 20, now 15)
        actions_to_evaluate = promising_actions if promising_actions else valid_actions[:15]
        
        best_action = None
        best_score = float('-inf')
        
        for row, col in actions_to_evaluate:
            move_score = self.evaluate_move(game_engine, row, col, depth=0)
            if move_score > best_score:
                best_score = move_score
                best_action = (row, col)
        
        return best_action
    
    def play_game(self, game_engine: GameEngine = None) -> dict:
        """Play a game and return results"""
        engine = game_engine if game_engine else GameEngine()
        engine.reset()
        
        steps = 0
        max_steps = 1000
        
        while not engine.game_over and steps < max_steps:
            action = self.choose_action(engine)
            if not action:
                break
            
            row, col = action
            result = engine.act(row, col)
            if not result['success']:
                break
            
            steps += 1
        
        return {
            'score': engine.score,
            'steps': steps,
            'game_over': engine.game_over
        }


def load_weights(filepath: str) -> dict:
    """Load weights from JSON file"""
    import json
    with open(filepath, 'r') as f:
        return json.load(f)


def main():
    """Test the direct agent"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test DirectAgent')
    parser.add_argument('--weights', type=str, default=None, help='Load weights from JSON file')
    parser.add_argument('--games', type=int, default=10, help='Number of games to play')
    args = parser.parse_args()
    
    print("Direct Playing Agent for Crazy Blocks")
    print("=" * 50)
    
    # Load weights if provided
    weights = None
    if args.weights:
        weights = load_weights(args.weights)
        print(f"Loaded weights from {args.weights}")
    
    agent = DirectAgent(weights=weights)
    engine = GameEngine()
    
    # Play multiple games
    num_games = args.games
    scores = []
    
    for i in range(num_games):
        result = agent.play_game(engine)
        scores.append(result['score'])
        print(f"Game {i+1}: Score = {result['score']:6.0f}, Steps = {result['steps']}")
    
    avg_score = sum(scores) / len(scores)
    best_score = max(scores)
    worst_score = min(scores)
    
    print()
    print("=" * 50)
    print(f"Results after {num_games} games:")
    print(f"  Average Score: {avg_score:.1f}")
    print(f"  Best Score: {best_score:.0f}")
    print(f"  Worst Score: {worst_score:.0f}")
    print("=" * 50)


if __name__ == '__main__':
    main()

