// Simple Q-Learning agent for Crazy Blocks
export class RLAgent {
  constructor(learningRate = 0.1, discountFactor = 0.95, epsilon = 1.0, epsilonDecay = 0.995, epsilonMin = 0.01) {
    this.learningRate = learningRate
    this.discountFactor = discountFactor
    this.epsilon = epsilon // Exploration rate
    this.epsilonDecay = epsilonDecay
    this.epsilonMin = epsilonMin
    
    // Q-table: state -> action -> Q-value
    // Using a Map for sparse representation
    this.qTable = new Map()
    
    // Statistics
    this.totalGames = 0
    this.totalScore = 0
    this.bestScore = 0
    this.episodeScores = []
    this.episodeRewards = []
  }

  // Convert state array to string key
  stateToKey(state) {
    return state.join(',')
  }

  // Get Q-value for state-action pair
  getQValue(state, actionIndex) {
    const stateKey = this.stateToKey(state)
    if (!this.qTable.has(stateKey)) {
      this.qTable.set(stateKey, new Map())
    }
    const actionMap = this.qTable.get(stateKey)
    if (!actionMap.has(actionIndex)) {
      actionMap.set(actionIndex, 0)
    }
    return actionMap.get(actionIndex)
  }

  // Set Q-value for state-action pair
  setQValue(state, actionIndex, value) {
    const stateKey = this.stateToKey(state)
    if (!this.qTable.has(stateKey)) {
      this.qTable.set(stateKey, new Map())
    }
    this.qTable.get(stateKey).set(actionIndex, value)
  }

  // Choose action using epsilon-greedy policy
  chooseAction(state, validActions) {
    if (Math.random() < this.epsilon) {
      // Explore: random action
      const randomIndex = Math.floor(Math.random() * validActions.length)
      return validActions[randomIndex]
    } else {
      // Exploit: best action
      let bestAction = validActions[0]
      let bestQValue = -Infinity
      
      for (const action of validActions) {
        const actionIndex = action.row * 5 + action.col
        const qValue = this.getQValue(state, actionIndex)
        if (qValue > bestQValue) {
          bestQValue = qValue
          bestAction = action
        }
      }
      
      return bestAction
    }
  }

  // Update Q-value using Q-learning formula
  updateQValue(state, actionIndex, reward, nextState, nextValidActions) {
    const currentQ = this.getQValue(state, actionIndex)
    
    // Calculate max Q-value for next state
    let maxNextQ = 0
    if (nextValidActions && nextValidActions.length > 0) {
      maxNextQ = Math.max(
        ...nextValidActions.map(action => {
          const idx = action.row * 5 + action.col
          return this.getQValue(nextState, idx)
        })
      )
    }
    
    // Q-learning update: Q(s,a) = Q(s,a) + α[r + γ*max(Q(s',a')) - Q(s,a)]
    const newQ = currentQ + this.learningRate * (reward + this.discountFactor * maxNextQ - currentQ)
    this.setQValue(state, actionIndex, newQ)
  }

  // Train on one episode
  trainEpisode(gameEngine) {
    let state = gameEngine.reset()
    let totalReward = 0
    let steps = 0
    const maxSteps = 1000 // Prevent infinite loops
    
    while (!gameEngine.gameOver && steps < maxSteps) {
      const validActions = gameEngine.getValidActions()
      
      if (validActions.length === 0) {
        break
      }
      
      // Choose action
      const action = this.chooseAction(state, validActions)
      const actionIndex = action.row * 5 + action.col
      
      // Take action
      const result = gameEngine.act(action.row, action.col)
      
      if (result.success) {
        const nextState = gameEngine.getState()
        const nextValidActions = gameEngine.getValidActions()
        
        // Update Q-value
        this.updateQValue(state, actionIndex, result.reward, nextState, nextValidActions)
        
        totalReward += result.reward
        state = nextState
      } else {
        // Invalid action, small penalty
        const nextState = gameEngine.getState()
        const nextValidActions = gameEngine.getValidActions()
        this.updateQValue(state, actionIndex, -1, nextState, nextValidActions)
        break
      }
      
      steps++
    }
    
    // Decay epsilon
    this.epsilon = Math.max(this.epsilonMin, this.epsilon * this.epsilonDecay)
    
    // Update statistics
    this.totalGames++
    this.totalScore += gameEngine.score
    this.bestScore = Math.max(this.bestScore, gameEngine.score)
    this.episodeScores.push(gameEngine.score)
    this.episodeRewards.push(totalReward)
    
    // Keep only last 100 episodes for stats
    if (this.episodeScores.length > 100) {
      this.episodeScores.shift()
      this.episodeRewards.shift()
    }
    
    return {
      score: gameEngine.score,
      reward: totalReward,
      steps,
      epsilon: this.epsilon
    }
  }

  // Get statistics
  getStats() {
    const avgScore = this.episodeScores.length > 0
      ? this.episodeScores.reduce((a, b) => a + b, 0) / this.episodeScores.length
      : 0
    const avgReward = this.episodeRewards.length > 0
      ? this.episodeRewards.reduce((a, b) => a + b, 0) / this.episodeRewards.length
      : 0
    
    return {
      totalGames: this.totalGames,
      avgScore: avgScore.toFixed(2),
      bestScore: this.bestScore,
      avgReward: avgReward.toFixed(2),
      epsilon: this.epsilon.toFixed(4),
      qTableSize: this.qTable.size
    }
  }
}

