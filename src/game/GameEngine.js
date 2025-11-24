// Headless game engine for RL training
export const COLUMNS = 5
export const ROWS = 17
export const COLORS = ['#6BA85A', '#A08FB8', '#D4A5A0', '#7DADB5'] // Rich Emerald, Amethyst, Rose Amber, Blue Teal
export const COLOR_INDICES = [0, 1, 2, 3] // For easier state representation

export class GameEngine {
  constructor() {
    this.grid = this.initializeGrid()
    this.score = 0
    this.gameOver = false
    this.turnCount = 0
  }

  initializeGrid() {
    return Array(ROWS).fill(null).map(() => Array(COLUMNS).fill(null))
  }

  reset() {
    this.grid = this.initializeGrid()
    this.score = 0
    this.gameOver = false
    this.turnCount = 0
    // Start with 4 rows
    this.addNewRow()
    this.addNewRow()
    this.addNewRow()
    this.addNewRow()
    return this.getState()
  }

  addNewRow() {
    if (this.gameOver) return
    
    const newRow = Array(COLUMNS).fill(null).map(() => 
      COLORS[Math.floor(Math.random() * COLORS.length)]
    )
    // Shift all rows up (remove top row) and add new row at bottom
    this.grid = [...this.grid.slice(1), newRow]
    
    // Check if game over (top row has blocks)
    if (this.grid[0].some(block => block !== null)) {
      this.gameOver = true
    }
  }

  findConnectedBlocks(row, col, color, visited = new Set()) {
    const key = `${row},${col}`
    if (visited.has(key)) return []
    
    // Check bounds
    if (row < 0 || row >= ROWS || col < 0 || col >= COLUMNS) return []
    
    // Check if block exists and matches color
    if (this.grid[row][col] !== color) return []
    
    visited.add(key)
    const connected = [[row, col]]
    
    // Check adjacent blocks (not diagonal): up, down, left, right
    const directions = [[-1, 0], [1, 0], [0, -1], [0, 1]]
    
    for (const [dr, dc] of directions) {
      const newRow = row + dr
      const newCol = col + dc
      connected.push(...this.findConnectedBlocks(newRow, newCol, color, visited))
    }
    
    return connected
  }

  applyGravity() {
    const newGrid = this.grid.map(row => [...row])
    
    // For each column, drop blocks down
    for (let col = 0; col < COLUMNS; col++) {
      let writeIndex = ROWS - 1
      for (let row = ROWS - 1; row >= 0; row--) {
        if (newGrid[row][col] !== null) {
          if (writeIndex !== row) {
            newGrid[writeIndex][col] = newGrid[row][col]
            newGrid[row][col] = null
          }
          writeIndex--
        }
      }
    }
    
    this.grid = newGrid
  }

  // Action: click on a block at (row, col)
  // Returns: { success: boolean, blocksEliminated: number, reward: number }
  act(row, col) {
    if (this.gameOver || row === ROWS - 1 || this.grid[row][col] === null) {
      return { success: false, blocksEliminated: 0, reward: 0 }
    }
    
    const color = this.grid[row][col]
    const connectedBlocks = this.findConnectedBlocks(row, col, color)
    
    if (connectedBlocks.length === 0) {
      return { success: false, blocksEliminated: 0, reward: 0 }
    }
    
    // Eliminate blocks
    connectedBlocks.forEach(([r, c]) => {
      this.grid[r][c] = null
    })
    
    const blocksEliminated = connectedBlocks.length
    this.score += blocksEliminated
    
    // Apply gravity
    this.applyGravity()
    
    // Add new row (turn ends)
    this.addNewRow()
    this.turnCount++
    
    // Calculate reward: points for blocks eliminated, bonus for large groups
    const baseReward = blocksEliminated
    const bonusReward = blocksEliminated >= 10 ? blocksEliminated * 0.5 : 0
    const reward = baseReward + bonusReward
    
    // Penalty for game over
    const gameOverPenalty = this.gameOver ? -100 : 0
    
    return {
      success: true,
      blocksEliminated,
      reward: reward + gameOverPenalty,
      gameOver: this.gameOver
    }
  }

  // Get current state representation for RL
  getState() {
    // Flatten grid and convert colors to indices
    const state = []
    for (let row = 0; row < ROWS; row++) {
      for (let col = 0; col < COLUMNS; col++) {
        const block = this.grid[row][col]
        if (block === null) {
          state.push(0) // Empty
        } else {
          const colorIndex = COLORS.indexOf(block)
          state.push(colorIndex + 1) // 1-4 for colors
        }
      }
    }
    
    // Add metadata
    state.push(this.score / 1000) // Normalized score
    state.push(this.turnCount / 100) // Normalized turn count
    state.push(this.gameOver ? 1 : 0) // Game over flag
    
    return state
  }

  // Get all valid actions (clickable positions)
  getValidActions() {
    const actions = []
    for (let row = 0; row < ROWS - 1; row++) { // Exclude bottom row
      for (let col = 0; col < COLUMNS; col++) {
        if (this.grid[row][col] !== null) {
          actions.push({ row, col })
        }
      }
    }
    return actions
  }

  // Get action index from row, col
  actionToIndex(row, col) {
    return row * COLUMNS + col
  }

  // Get row, col from action index
  indexToAction(index) {
    const row = Math.floor(index / COLUMNS)
    const col = index % COLUMNS
    return { row, col }
  }
}

