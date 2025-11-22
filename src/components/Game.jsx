import React, { useState, useEffect, useCallback, useRef } from 'react'
import './Game.css'

const COLUMNS = 5
const ROWS = 17
const COLORS = ['#6BA85A', '#A08FB8', '#D4A5A0', '#7DADB5'] // Rich Emerald, Amethyst, Rose Amber, Blue Teal

function Game() {
  const [grid, setGrid] = useState(() => initializeGrid())
  const [score, setScore] = useState(0)
  const [gameOver, setGameOver] = useState(false)
  const [isProcessing, setIsProcessing] = useState(false)
  const hasInitialized = useRef(false)
  const gameOverRef = useRef(false)

  // Initialize empty grid
  function initializeGrid() {
    return Array(ROWS).fill(null).map(() => Array(COLUMNS).fill(null))
  }

  // Add a new row at the bottom
  const addNewRow = useCallback(() => {
    if (gameOverRef.current) return
    
    setGrid(prevGrid => {
      const newRow = Array(COLUMNS).fill(null).map(() => 
        COLORS[Math.floor(Math.random() * COLORS.length)]
      )
      // Shift all rows up (remove top row) and add new row at bottom
      // grid[0] is top, grid[ROWS-1] is bottom
      const newGrid = [...prevGrid.slice(1), newRow]
      
      // Check if game over (top row has blocks)
      if (newGrid[0].some(block => block !== null)) {
        gameOverRef.current = true
        setGameOver(true)
      }
      
      return newGrid
    })
  }, [])

  // Flood fill to find all connected blocks of the same color
  const findConnectedBlocks = useCallback((row, col, color, visited = new Set()) => {
    const key = `${row},${col}`
    if (visited.has(key)) return []
    
    // Check bounds
    if (row < 0 || row >= ROWS || col < 0 || col >= COLUMNS) return []
    
    // Check if block exists and matches color
    if (grid[row][col] !== color) return []
    
    visited.add(key)
    const connected = [[row, col]]
    
    // Check adjacent blocks (not diagonal): up, down, left, right
    const directions = [[-1, 0], [1, 0], [0, -1], [0, 1]]
    
    for (const [dr, dc] of directions) {
      const newRow = row + dr
      const newCol = col + dc
      connected.push(...findConnectedBlocks(newRow, newCol, color, visited))
    }
    
    return connected
  }, [grid])

  // Handle block click
  const handleBlockClick = useCallback((row, col) => {
    // Prevent clicking on the bottom row
    if (row === ROWS - 1) return
    if (gameOver || isProcessing || grid[row][col] === null) return
    
    const color = grid[row][col]
    const connectedBlocks = findConnectedBlocks(row, col, color)
    
    // Need at least 1 block (the clicked one)
    if (connectedBlocks.length === 0) return
    
    setIsProcessing(true)
    
    // Update score
    setScore(prev => prev + connectedBlocks.length)
    
    // Eliminate blocks
    setGrid(prevGrid => {
      const newGrid = prevGrid.map(row => [...row])
      connectedBlocks.forEach(([r, c]) => {
        newGrid[r][c] = null
      })
      return newGrid
    })
    
    // Apply gravity after a short delay for visual effect
    setTimeout(() => {
      setGrid(prevGrid => {
        const newGrid = prevGrid.map(row => [...row])
        
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
        
        return newGrid
      })
      
      setIsProcessing(false)
      
      // Add new row after gravity is applied
      setTimeout(() => {
        addNewRow()
      }, 100)
    }, 300)
  }, [gameOver, isProcessing, grid, findConnectedBlocks, addNewRow])

  // Start the game with initial rows
  useEffect(() => {
    if (hasInitialized.current) return
    hasInitialized.current = true
    
    addNewRow()
    setTimeout(() => {
      addNewRow()
      setTimeout(() => {
        addNewRow()
        setTimeout(() => {
          addNewRow()
        }, 100)
      }, 100)
    }, 100)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []) // Only run once on mount

  // Restart game
  const handleRestart = () => {
    setGrid(initializeGrid())
    setScore(0)
    setGameOver(false)
    gameOverRef.current = false
    setIsProcessing(false)
    hasInitialized.current = false
    setTimeout(() => {
      addNewRow()
      setTimeout(() => {
        addNewRow()
        setTimeout(() => {
          addNewRow()
          setTimeout(() => {
            addNewRow()
          }, 100)
        }, 100)
      }, 100)
    }, 100)
  }

  return (
    <div className="game-container">
      <div className="game-header">
        <h1>Crazy Blocks</h1>
        <div className="score">Score: {score}</div>
        {gameOver && (
          <div className="game-over">
            <h2>Game Over!</h2>
            <button onClick={handleRestart} className="restart-button">
              Play Again
            </button>
          </div>
        )}
      </div>
      
      <div className="grid-container">
        <div className="grid">
          {grid.map((row, rowIndex) =>
            row.map((block, colIndex) => {
              const isBottomRow = rowIndex === ROWS - 1
              return (
                <div
                  key={`${rowIndex}-${colIndex}`}
                  className={`block ${block ? 'filled' : 'empty'} ${isBottomRow ? 'bottom-row' : ''}`}
                  style={{ backgroundColor: block || '#2c3e50' }}
                  onClick={() => handleBlockClick(rowIndex, colIndex)}
                />
              )
            })
          )}
        </div>
      </div>
      
      {!gameOver && (
        <div className="instructions">
          Click blocks to eliminate connected groups of the same color!
        </div>
      )}
    </div>
  )
}

export default Game

