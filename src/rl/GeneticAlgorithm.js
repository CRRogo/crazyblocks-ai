// Genetic Algorithm for evolving strategies
import { GeneticAgent } from './GeneticAgent'
import { GameEngine } from '../game/GameEngine'

export class GeneticAlgorithm {
  constructor(
    populationSize = 50,
    mutationRate = 0.1,
    crossoverRate = 0.7,
    eliteSize = 5,
    gamesPerEvaluation = 5
  ) {
    this.populationSize = populationSize
    this.mutationRate = mutationRate
    this.crossoverRate = crossoverRate
    this.eliteSize = eliteSize
    this.gamesPerEvaluation = gamesPerEvaluation
    
    this.population = []
    this.generation = 0
    this.gameEngine = new GameEngine()
    this.history = []
    this.storageKey = 'crazyblocks-genetic-algorithm'
    
    // Try to load saved population, otherwise initialize new
    if (!this.loadPopulation()) {
      this.initializePopulation()
    }
  }

  initializePopulation() {
    this.population = []
    for (let i = 0; i < this.populationSize; i++) {
      this.population.push(new GeneticAgent())
    }
  }

  // Evaluate fitness of all agents
  async evaluatePopulation() {
    for (const agent of this.population) {
      agent.resetFitness()
      
      // Play multiple games to get average fitness
      for (let i = 0; i < this.gamesPerEvaluation; i++) {
        agent.playGame(this.gameEngine)
      }
    }
    
    // Sort by fitness (descending)
    this.population.sort((a, b) => b.fitness - a.fitness)
  }

  // Select parents using tournament selection
  tournamentSelection(tournamentSize = 3) {
    const tournament = []
    for (let i = 0; i < tournamentSize; i++) {
      const randomIndex = Math.floor(Math.random() * this.population.length)
      tournament.push(this.population[randomIndex])
    }
    tournament.sort((a, b) => b.fitness - a.fitness)
    return tournament[0]
  }

  // Create next generation
  createNextGeneration() {
    const newPopulation = []
    
    // Keep elite (best performers)
    for (let i = 0; i < this.eliteSize; i++) {
      const elite = new GeneticAgent(this.population[i].strategy)
      elite.fitness = this.population[i].fitness
      elite.bestScore = this.population[i].bestScore
      newPopulation.push(elite)
    }
    
    // Fill rest of population
    while (newPopulation.length < this.populationSize) {
      if (Math.random() < this.crossoverRate && newPopulation.length < this.populationSize - 1) {
        // Crossover
        const parent1 = this.tournamentSelection()
        const parent2 = this.tournamentSelection()
        const child = parent1.crossover(parent2)
        newPopulation.push(child.mutate(this.mutationRate))
      } else {
        // Mutation only
        const parent = this.tournamentSelection()
        newPopulation.push(parent.mutate(this.mutationRate))
      }
    }
    
    this.population = newPopulation
    this.generation++
  }

  // Run one generation
  async runGeneration() {
    await this.evaluatePopulation()
    
    const bestAgent = this.population[0]
    const avgFitness = this.population.reduce((sum, a) => sum + a.fitness, 0) / this.population.length
    
    this.history.push({
      generation: this.generation,
      bestFitness: bestAgent.fitness,
      bestScore: bestAgent.bestScore,
      avgFitness: avgFitness,
      population: this.population.map(a => ({
        fitness: a.fitness,
        bestScore: a.bestScore
      }))
    })
    
    // Keep only last 100 generations in history
    if (this.history.length > 100) {
      this.history.shift()
    }
    
    this.createNextGeneration()
    
    // Save population after each generation
    this.savePopulation()
    
    return {
      generation: this.generation - 1,
      bestFitness: bestAgent.fitness,
      bestScore: bestAgent.bestScore,
      avgFitness: avgFitness
    }
  }

  // Get best agent
  getBestAgent() {
    if (this.population.length === 0) return null
    this.population.sort((a, b) => b.fitness - a.fitness)
    return this.population[0]
  }

  // Save population to localStorage
  savePopulation() {
    try {
      // Sort by fitness before saving
      this.population.sort((a, b) => b.fitness - a.fitness)
      
      const data = {
        generation: this.generation,
        population: this.population.map(agent => ({
          strategy: agent.strategy,
          fitness: agent.fitness,
          bestScore: agent.bestScore,
          gamesPlayed: agent.gamesPlayed,
          totalScore: agent.totalScore
        })),
        timestamp: Date.now(),
        bestFitness: this.population[0]?.fitness || 0,
        bestScore: this.population[0]?.bestScore || 0
      }
      
      localStorage.setItem(this.storageKey, JSON.stringify(data))
      return true
    } catch (error) {
      console.error('Failed to save population:', error)
      return false
    }
  }

  // Load population from localStorage
  loadPopulation() {
    try {
      const saved = localStorage.getItem(this.storageKey)
      if (!saved) {
        return false
      }
      
      const data = JSON.parse(saved)
      
      // Restore population
      this.population = data.population.map(agentData => {
        const agent = new GeneticAgent(agentData.strategy)
        agent.fitness = agentData.fitness || 0
        agent.bestScore = agentData.bestScore || 0
        agent.gamesPlayed = agentData.gamesPlayed || 0
        agent.totalScore = agentData.totalScore || 0
        return agent
      })
      
      // Ensure we have enough agents (fill with new random if needed)
      while (this.population.length < this.populationSize) {
        this.population.push(new GeneticAgent())
      }
      
      // Trim if we have too many
      if (this.population.length > this.populationSize) {
        this.population = this.population.slice(0, this.populationSize)
      }
      
      this.generation = data.generation || 0
      
      return true
    } catch (error) {
      console.error('Failed to load population:', error)
      return false
    }
  }

  // Get saved data info
  getSavedDataInfo() {
    try {
      const saved = localStorage.getItem(this.storageKey)
      if (!saved) {
        return null
      }
      
      const data = JSON.parse(saved)
      return {
        generation: data.generation,
        bestFitness: data.bestFitness,
        bestScore: data.bestScore,
        timestamp: data.timestamp,
        populationSize: data.population?.length || 0
      }
    } catch (error) {
      return null
    }
  }

  // Export strategies to JSON (for sharing/backup)
  exportStrategies() {
    this.population.sort((a, b) => b.fitness - a.fitness)
    
    return {
      version: '1.0',
      generation: this.generation,
      timestamp: Date.now(),
      bestFitness: this.population[0]?.fitness || 0,
      bestScore: this.population[0]?.bestScore || 0,
      populationSize: this.population.length,
      eliteStrategies: this.population.slice(0, this.eliteSize).map(agent => ({
        strategy: agent.strategy,
        fitness: agent.fitness,
        bestScore: agent.bestScore,
        gamesPlayed: agent.gamesPlayed
      })),
      fullPopulation: this.population.map(agent => ({
        strategy: agent.strategy,
        fitness: agent.fitness,
        bestScore: agent.bestScore,
        gamesPlayed: agent.gamesPlayed
      }))
    }
  }

  // Import strategies from JSON
  importStrategies(data) {
    try {
      if (!data.population || data.population.length === 0) {
        throw new Error('Invalid strategy data')
      }

      this.population = data.fullPopulation || data.eliteStrategies || data.population
        .map(agentData => {
          const agent = new GeneticAgent(agentData.strategy)
          agent.fitness = agentData.fitness || 0
          agent.bestScore = agentData.bestScore || 0
          agent.gamesPlayed = agentData.gamesPlayed || 0
          return agent
        })

      // Fill to population size if needed
      while (this.population.length < this.populationSize) {
        this.population.push(new GeneticAgent())
      }

      // Trim if too many
      if (this.population.length > this.populationSize) {
        this.population = this.population.slice(0, this.populationSize)
      }

      this.generation = data.generation || 0
      this.savePopulation()
      
      return true
    } catch (error) {
      console.error('Failed to import strategies:', error)
      return false
    }
  }

  // Reset and clear saved data
  reset() {
    localStorage.removeItem(this.storageKey)
    this.population = []
    this.generation = 0
    this.history = []
    this.initializePopulation()
  }

  // Get statistics
  getStats() {
    if (this.population.length === 0) {
      return {
        generation: 0,
        bestFitness: 0,
        bestScore: 0,
        avgFitness: 0
      }
    }
    
    this.population.sort((a, b) => b.fitness - a.fitness)
    const bestAgent = this.population[0]
    const avgFitness = this.population.reduce((sum, a) => sum + a.fitness, 0) / this.population.length
    
    return {
      generation: this.generation,
      bestFitness: bestAgent.fitness.toFixed(2),
      bestScore: bestAgent.bestScore,
      avgFitness: avgFitness.toFixed(2),
      populationSize: this.population.length
    }
  }
}

