// Genetic Algorithm-based agent for Crazy Blocks
import { GameEngine, ROWS, COLUMNS, COLORS } from '../game/GameEngine'

export class GeneticAgent {
  constructor(strategy = null) {
    // Strategy is a set of weights/heuristics for decision making
    // We'll use a simple heuristic-based strategy with weights
    if (strategy) {
      // Backward compatibility: add default values for new heuristics
      this.strategy = {
        groupSizeWeight: strategy.groupSizeWeight || 0,
        cascadePotentialWeight: strategy.cascadePotentialWeight || 0,
        topClearWeight: strategy.topClearWeight || 0,
        bottomAvoidWeight: strategy.bottomAvoidWeight || 0,
        colorDiversityWeight: strategy.colorDiversityWeight || 0,
        isolationPenalty: strategy.isolationPenalty || 0,
        spaceCreationWeight: strategy.spaceCreationWeight || 0,
        columnHeightWeight: strategy.columnHeightWeight || 0,
        dangerReductionWeight: strategy.dangerReductionWeight || 0,
        colorBalanceWeight: strategy.colorBalanceWeight || 0,
        topDensityWeight: strategy.topDensityWeight || 0,
        multiCascadeWeight: strategy.multiCascadeWeight || 0,
        minimumGroupSizeWeight: strategy.minimumGroupSizeWeight || 0,
        largeGroupBonusWeight: strategy.largeGroupBonusWeight || 0,
        columnBalanceWeight: strategy.columnBalanceWeight || 0,
        groupMergingWeight: strategy.groupMergingWeight || 0,
        averageBlocksWeight: strategy.averageBlocksWeight || 0,
      }
    } else {
      // Initialize random strategy
      this.strategy = {
        // Weight for preferring larger groups
        groupSizeWeight: Math.random() * 2 - 1, // -1 to 1
        // Weight for preferring groups that create cascades
        cascadePotentialWeight: Math.random() * 2 - 1,
        // Weight for preferring to clear from top
        topClearWeight: Math.random() * 2 - 1,
        // Weight for preferring to clear from bottom (avoid bottom row)
        bottomAvoidWeight: Math.random() * 2 - 1,
        // Weight for color diversity (prefer clearing colors that appear less)
        colorDiversityWeight: Math.random() * 2 - 1,
        // Weight for preferring actions that leave fewer isolated blocks
        isolationPenalty: Math.random() * 2 - 1,
        // Weight for preferring actions that create more empty space
        spaceCreationWeight: Math.random() * 2 - 1,
        // NEW: Column height danger (prefer clearing from tall columns)
        columnHeightWeight: Math.random() * 2 - 1,
        // NEW: Danger reduction (prefer moves that reduce columns near top)
        dangerReductionWeight: Math.random() * 2 - 1,
        // NEW: Color balance (prefer moves that balance colors across board)
        colorBalanceWeight: Math.random() * 2 - 1,
        // NEW: Top row density (prefer moves that reduce blocks in top rows)
        topDensityWeight: Math.random() * 2 - 1,
        // NEW: Multi-cascade depth (prefer moves that create longer chains)
        multiCascadeWeight: Math.random() * 2 - 1,
        // NEW: Minimum group size penalty (penalize groups < 5 blocks)
        minimumGroupSizeWeight: Math.random() * 2 - 1,
        // NEW: Large group bonus (reward groups >= 5 blocks)
        largeGroupBonusWeight: Math.random() * 2 - 1,
        // NEW: Column height balance (prefer moves that keep columns similar height)
        columnBalanceWeight: Math.random() * 2 - 1,
        // NEW: Group merging potential (reward small eliminations that merge large groups)
        groupMergingWeight: Math.random() * 2 - 1,
        // NEW: Average blocks per turn (reward moves that eliminate 5+ blocks)
        averageBlocksWeight: Math.random() * 2 - 1,
      }
    }
    
    this.fitness = 0 // Average score
    this.gamesPlayed = 0
    this.totalScore = 0
    this.bestScore = 0
  }

  // Evaluate an action based on strategy
  evaluateAction(gameEngine, row, col) {
    if (row === ROWS - 1 || gameEngine.grid[row][col] === null) {
      return -Infinity
    }

    const color = gameEngine.grid[row][col]
    const connectedBlocks = gameEngine.findConnectedBlocks(row, col, color)
    const groupSize = connectedBlocks.length

    if (groupSize === 0) {
      return -Infinity
    }

    // Calculate heuristic score based on strategy weights
    let score = 0

    // Group size preference
    score += this.strategy.groupSizeWeight * groupSize

    // NEW: Critical game mechanic - continuous reward/penalty based on blocks removed
    // The reward/punishment scales smoothly with the actual number of blocks
    // 1 block = maximum penalty, 4 blocks = small penalty, 5 blocks = neutral
    // 8 blocks = moderate bonus, 10 blocks = larger bonus, etc.
    
    if (groupSize < 5) {
      // Penalty scales with how small the group is
      // 1 block = -1.0, 2 blocks = -0.75, 3 blocks = -0.5, 4 blocks = -0.25
      const penalty = (5 - groupSize) / 4.0 // 1.0 to 0.25 (worse for smaller groups)
      score += this.strategy.minimumGroupSizeWeight * (-penalty * 10) // Strong negative signal
    } else {
      // Bonus scales continuously with size above 5
      // 5 blocks = 0.0, 8 blocks = 0.3, 10 blocks = 0.5, 15 blocks = 1.0
      const bonus = Math.min((groupSize - 5) / 10.0, 1.0) // Normalized 0.0 to 1.0, capped at 1.0
      score += this.strategy.largeGroupBonusWeight * (bonus * 10) // Strong positive signal
    }

    // Top clearing preference (higher row = better)
    score += this.strategy.topClearWeight * (row / ROWS)

    // Bottom avoidance (prefer not clearing near bottom row)
    score += this.strategy.bottomAvoidWeight * ((ROWS - row) / ROWS)

    // Color diversity - quick estimate (sample instead of full scan)
    // Sample a few rows instead of scanning entire grid
    let colorCount = 0
    const sampleSize = Math.min(5, ROWS)
    for (let r = 0; r < sampleSize; r++) {
      for (let c = 0; c < COLUMNS; c++) {
        if (gameEngine.grid[r][c] === color) colorCount++
      }
    }
    score += this.strategy.colorDiversityWeight * (colorCount / (sampleSize * COLUMNS))

    // Simulate the move to check cascade potential
    const tempGrid = gameEngine.grid.map(row => [...row])
    connectedBlocks.forEach(([r, c]) => {
      tempGrid[r][c] = null
    })

    // Apply gravity to temp grid
    const tempGridAfterGravity = tempGrid.map(row => [...row])
    for (let c = 0; c < COLUMNS; c++) {
      let writeIndex = ROWS - 1
      for (let r = ROWS - 1; r >= 0; r--) {
        if (tempGridAfterGravity[r][c] !== null) {
          if (writeIndex !== r) {
            tempGridAfterGravity[writeIndex][c] = tempGridAfterGravity[r][c]
            tempGridAfterGravity[r][c] = null
          }
          writeIndex--
        }
      }
    }

    // Check for potential cascades (sample top rows only - most important area)
    let cascadePotential = 0
    const cascadeSampleRows = Math.min(8, ROWS - 1) // Only check top 8 rows
    for (let r = 0; r < cascadeSampleRows; r++) {
      for (let c = 0; c < COLUMNS; c++) {
        if (tempGridAfterGravity[r][c] !== null) {
          const checkColor = tempGridAfterGravity[r][c]
          // Check adjacent
          if (r > 0 && tempGridAfterGravity[r - 1][c] === checkColor) cascadePotential++
          if (r < cascadeSampleRows - 1 && tempGridAfterGravity[r + 1][c] === checkColor) cascadePotential++
          if (c > 0 && tempGridAfterGravity[r][c - 1] === checkColor) cascadePotential++
          if (c < COLUMNS - 1 && tempGridAfterGravity[r][c + 1] === checkColor) cascadePotential++
        }
      }
    }
    score += this.strategy.cascadePotentialWeight * (cascadePotential / 50)

    // Space creation - count empty spaces after move
    let emptySpaces = 0
    for (let r = 0; r < ROWS; r++) {
      for (let c = 0; c < COLUMNS; c++) {
        if (tempGridAfterGravity[r][c] === null) emptySpaces++
      }
    }
    score += this.strategy.spaceCreationWeight * (emptySpaces / (ROWS * COLUMNS))

    // NEW: Column height analysis - prefer clearing from taller columns
    const columnHeights = []
    for (let c = 0; c < COLUMNS; c++) {
      let height = 0
      for (let r = 0; r < ROWS; r++) {
        if (gameEngine.grid[r][c] !== null) height++
      }
      columnHeights.push(height)
    }
    const clickedColumnHeight = columnHeights[col]
    score += this.strategy.columnHeightWeight * (clickedColumnHeight / ROWS)

    // NEW: Danger assessment - how close columns are to top
    let maxDanger = 0
    for (let c = 0; c < COLUMNS; c++) {
      // Find topmost block in column
      for (let r = 0; r < ROWS; r++) {
        if (gameEngine.grid[r][c] !== null) {
          const danger = (ROWS - r) / ROWS // Higher = more dangerous
          maxDanger = Math.max(maxDanger, danger)
          break
        }
      }
    }
    // After move, check if danger is reduced
    let maxDangerAfter = 0
    for (let c = 0; c < COLUMNS; c++) {
      for (let r = 0; r < ROWS; r++) {
        if (tempGridAfterGravity[r][c] !== null) {
          const danger = (ROWS - r) / ROWS
          maxDangerAfter = Math.max(maxDangerAfter, danger)
          break
        }
      }
    }
    const dangerReduction = maxDanger - maxDangerAfter
    score += this.strategy.dangerReductionWeight * dangerReduction

    // NEW: Isolated block detection (blocks with no same-color neighbors)
    let isolatedBlocks = 0
    for (let r = 0; r < ROWS - 1; r++) {
      for (let c = 0; c < COLUMNS; c++) {
        if (tempGridAfterGravity[r][c] !== null) {
          const blockColor = tempGridAfterGravity[r][c]
          let hasNeighbor = false
          const directions = [[-1, 0], [1, 0], [0, -1], [0, 1]]
          for (const [dr, dc] of directions) {
            const nr = r + dr
            const nc = c + dc
            if (nr >= 0 && nr < ROWS - 1 && nc >= 0 && nc < COLUMNS) {
              if (tempGridAfterGravity[nr][nc] === blockColor) {
                hasNeighbor = true
                break
              }
            }
          }
          if (!hasNeighbor) isolatedBlocks++
        }
      }
    }
    score += this.strategy.isolationPenalty * (-isolatedBlocks / (ROWS * COLUMNS)) // Negative = penalty

    // NEW: Color balance across entire board
    const colorCounts = { [COLORS[0]]: 0, [COLORS[1]]: 0, [COLORS[2]]: 0, [COLORS[3]]: 0 }
    for (let r = 0; r < ROWS; r++) {
      for (let c = 0; c < COLUMNS; c++) {
        const blockColor = gameEngine.grid[r][c]
        if (blockColor && colorCounts.hasOwnProperty(blockColor)) {
          colorCounts[blockColor]++
        }
      }
    }
    // Calculate variance (lower = more balanced)
    const colorValues = Object.values(colorCounts)
    const avg = colorValues.reduce((a, b) => a + b, 0) / colorValues.length
    const variance = colorValues.reduce((sum, val) => sum + Math.pow(val - avg, 2), 0) / colorValues.length
    const balanceScore = 1 / (1 + variance / 100) // Normalized
    score += this.strategy.colorBalanceWeight * balanceScore

    // NEW: Top row density (blocks in top 3 rows)
    let topDensity = 0
    for (let r = 0; r < Math.min(3, ROWS); r++) {
      for (let c = 0; c < COLUMNS; c++) {
        if (tempGridAfterGravity[r][c] !== null) topDensity++
      }
    }
    const topDensityBefore = (() => {
      let count = 0
      for (let r = 0; r < Math.min(3, ROWS); r++) {
        for (let c = 0; c < COLUMNS; c++) {
          if (gameEngine.grid[r][c] !== null) count++
        }
      }
      return count
    })()
    const topDensityReduction = (topDensityBefore - topDensity) / (3 * COLUMNS)
    score += this.strategy.topDensityWeight * topDensityReduction

    // NEW: Multi-cascade depth (simulate cascades recursively)
    const cascadeDepth = this.calculateCascadeDepth(tempGridAfterGravity, 0, 3)
    score += this.strategy.multiCascadeWeight * (cascadeDepth / 10)

    // NEW: Column height balance - penalize moves that create large height differences
    // Calculate column heights before move
    const columnHeightsBefore = []
    for (let c = 0; c < COLUMNS; c++) {
      let height = 0
      for (let r = 0; r < ROWS; r++) {
        if (gameEngine.grid[r][c] !== null) height++
      }
      columnHeightsBefore.push(height)
    }
    
    // Calculate column heights after move
    const columnHeightsAfter = []
    for (let c = 0; c < COLUMNS; c++) {
      let height = 0
      for (let r = 0; r < ROWS; r++) {
        if (tempGridAfterGravity[r][c] !== null) height++
      }
      columnHeightsAfter.push(height)
    }
    
    // Calculate variance (standard deviation) of column heights
    const calculateVariance = (heights) => {
      const avg = heights.reduce((a, b) => a + b, 0) / heights.length
      const variance = heights.reduce((sum, h) => sum + Math.pow(h - avg, 2), 0) / heights.length
      return Math.sqrt(variance) // Standard deviation
    }
    
    const varianceBefore = calculateVariance(columnHeightsBefore)
    const varianceAfter = calculateVariance(columnHeightsAfter)
    const balanceImprovement = (varianceBefore - varianceAfter) / ROWS // Normalized
    score += this.strategy.columnBalanceWeight * balanceImprovement * 10 // Scale up for impact

    // NEW: Group merging potential - detect if small elimination merges large groups
    // This rewards eliminating 1-2 blocks if it will merge two large groups
    if (groupSize <= 2) {
      // Check if there are large groups of the same color after gravity
      // that are significantly larger than what we eliminated (indicating a merge)
      const visited = new Set()
      let largestMergedGroup = 0
      
      // Find the largest group of the same color as what we eliminated
      for (let r = 0; r < ROWS - 1; r++) {
        for (let c = 0; c < COLUMNS; c++) {
          if (tempGridAfterGravity[r][c] === color && !visited.has(`${r},${c}`)) {
            const mergedGroup = this.findConnectedBlocksInGrid(tempGridAfterGravity, r, c, color, visited)
            if (mergedGroup.length > largestMergedGroup) {
              largestMergedGroup = mergedGroup.length
            }
          }
        }
      }
      
      // If the merged group is much larger than what we eliminated, it's a good merge
      // Example: eliminate 2 blocks, create a 12-block group = merged two 5-block groups
      if (largestMergedGroup >= 10 && largestMergedGroup > groupSize * 3) {
        // Strong bonus for creating a large merged group through small elimination
        // The larger the merged group relative to eliminated blocks, the better
        const mergingBonus = Math.min((largestMergedGroup - groupSize) / 15.0, 1.0) // Normalized, capped at 1.0
        score += this.strategy.groupMergingWeight * mergingBonus * 25 // Very strong positive signal
      }
    }

    // NEW: Average blocks per turn - reward moves that eliminate 5+ blocks
    // Since 5 blocks are added per turn, we need to eliminate at least 5 on average
    if (groupSize >= 5) {
      // Bonus for meeting the minimum requirement
      const baseBonus = 1.0
      // Additional bonus for exceeding (8+ blocks gets extra reward)
      const excessBonus = groupSize >= 8 ? (groupSize - 5) / 10.0 : 0
      score += this.strategy.averageBlocksWeight * (baseBonus + excessBonus) * 10
    } else {
      // Penalty for not meeting the requirement (scales with how far below 5)
      const deficit = (5 - groupSize) / 5.0 // 0.0 to 0.8 (for 1-4 blocks)
      score += this.strategy.averageBlocksWeight * (-deficit * 10) // Strong negative signal
    }

    return score
  }

  // Calculate potential cascade depth (recursive)
  calculateCascadeDepth(grid, depth, maxDepth) {
    if (depth >= maxDepth) return depth
    
    let maxCascade = depth
    // Find all potential cascades
    const cascadeGroups = []
    const visited = new Set()
    
    for (let r = 0; r < ROWS - 1; r++) {
      for (let c = 0; c < COLUMNS; c++) {
        if (grid[r][c] !== null && !visited.has(`${r},${c}`)) {
          const color = grid[r][c]
          const group = this.findConnectedBlocksInGrid(grid, r, c, color, visited)
          if (group.length >= 2) {
            cascadeGroups.push(group)
          }
        }
      }
    }
    
    if (cascadeGroups.length === 0) return depth
    
    // Simulate the largest cascade
    const largestGroup = cascadeGroups.reduce((a, b) => a.length > b.length ? a : b)
    const newGrid = grid.map(row => [...row])
    largestGroup.forEach(([r, c]) => {
      newGrid[r][c] = null
    })
    
    // Apply gravity
    for (let c = 0; c < COLUMNS; c++) {
      let writeIndex = ROWS - 1
      for (let r = ROWS - 1; r >= 0; r--) {
        if (newGrid[r][c] !== null) {
          if (writeIndex !== r) {
            newGrid[writeIndex][c] = newGrid[r][c]
            newGrid[r][c] = null
          }
          writeIndex--
        }
      }
    }
    
    return this.calculateCascadeDepth(newGrid, depth + 1, maxDepth)
  }

  // Helper to find connected blocks in a given grid
  findConnectedBlocksInGrid(grid, row, col, color, visited = new Set()) {
    const key = `${row},${col}`
    if (visited.has(key)) return []
    if (row < 0 || row >= ROWS || col < 0 || col >= COLUMNS) return []
    if (grid[row][col] !== color) return []
    
    visited.add(key)
    const connected = [[row, col]]
    const directions = [[-1, 0], [1, 0], [0, -1], [0, 1]]
    
    for (const [dr, dc] of directions) {
      connected.push(...this.findConnectedBlocksInGrid(grid, row + dr, col + dc, color, visited))
    }
    
    return connected
  }

  // Choose best action based on strategy
  chooseAction(gameEngine) {
    const validActions = gameEngine.getValidActions()
    if (validActions.length === 0) {
      return null
    }

    // Quick pre-filter: only evaluate actions with groups of size >= 2
    // This eliminates many bad actions quickly
    const promisingActions = []
    for (const action of validActions) {
      const color = gameEngine.grid[action.row][action.col]
      if (color) {
        // Quick check: count immediate neighbors of same color
        let neighborCount = 0
        const directions = [[-1, 0], [1, 0], [0, -1], [0, 1]]
        for (const [dr, dc] of directions) {
          const nr = action.row + dr
          const nc = action.col + dc
          if (nr >= 0 && nr < ROWS - 1 && nc >= 0 && nc < COLUMNS) {
            if (gameEngine.grid[nr][nc] === color) neighborCount++
          }
        }
        if (neighborCount > 0) {
          promisingActions.push(action)
        }
      }
    }

    // If we have promising actions, only evaluate those
    const actionsToEvaluate = promisingActions.length > 0 ? promisingActions : validActions.slice(0, 10)

    let bestAction = actionsToEvaluate[0]
    let bestScore = -Infinity

    for (const action of actionsToEvaluate) {
      const score = this.evaluateAction(gameEngine, action.row, action.col)
      if (score > bestScore) {
        bestScore = score
        bestAction = action
      }
    }

    return bestAction
  }

  // Play a game and return score
  playGame(gameEngine) {
    const engine = gameEngine || new GameEngine()
    engine.reset()
    
    let steps = 0
    const maxSteps = 1000

    while (!engine.gameOver && steps < maxSteps) {
      const action = this.chooseAction(engine)
      if (!action) break

      const result = engine.act(action.row, action.col)
      if (!result.success) break

      steps++
    }

    this.gamesPlayed++
    this.totalScore += engine.score
    this.fitness = this.totalScore / this.gamesPlayed
    this.bestScore = Math.max(this.bestScore, engine.score)

    return {
      score: engine.score,
      steps,
      fitness: this.fitness
    }
  }

  // Create a mutated copy
  mutate(mutationRate = 0.1) {
    const mutated = new GeneticAgent(this.strategy)
    
    for (const key in mutated.strategy) {
      if (Math.random() < mutationRate) {
        // Add random noise
        mutated.strategy[key] += (Math.random() * 2 - 1) * 0.2
        // Clamp to reasonable range
        mutated.strategy[key] = Math.max(-2, Math.min(2, mutated.strategy[key]))
      }
    }
    
    return mutated
  }

  // Crossover with another agent
  crossover(other) {
    const child = new GeneticAgent()
    
    for (const key in this.strategy) {
      // Blend strategies (could also do uniform crossover)
      if (Math.random() < 0.5) {
        child.strategy[key] = this.strategy[key]
      } else {
        child.strategy[key] = other.strategy[key]
      }
    }
    
    return child
  }

  // Reset fitness tracking
  resetFitness() {
    this.fitness = 0
    this.gamesPlayed = 0
    this.totalScore = 0
    this.bestScore = 0
  }
}

