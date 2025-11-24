import React, { useState, useEffect, useRef } from 'react'
import { GameEngine } from '../game/GameEngine'
import { RLAgent } from '../rl/RLAgent'
import { GeneticAlgorithm } from '../rl/GeneticAlgorithm'
import './TrainingView.css'

function TrainingView() {
  const [method, setMethod] = useState('genetic') // 'qlearning' or 'genetic'
  const [isTraining, setIsTraining] = useState(false)
  const [stats, setStats] = useState(null)
  const [episodes, setEpisodes] = useState(0)
  const [targetEpisodes, setTargetEpisodes] = useState(1000)
  const [targetGenerations, setTargetGenerations] = useState(50)
  const [savedDataInfo, setSavedDataInfo] = useState(null)
  
  // Q-Learning refs
  const qAgentRef = useRef(null)
  const gameEngineRef = useRef(null)
  
  // Genetic Algorithm refs
  const gaRef = useRef(null)
  
  const trainingIntervalRef = useRef(null)
  const isTrainingRef = useRef(false)

  useEffect(() => {
    // Initialize based on method
    if (method === 'qlearning') {
      qAgentRef.current = new RLAgent(0.1, 0.95, 1.0, 0.9995, 0.01)
      gameEngineRef.current = new GameEngine()
      setSavedDataInfo(null)
    } else {
      // Use faster settings: fewer games per evaluation for speed
      // Can be adjusted: (populationSize, mutationRate, crossoverRate, eliteSize, gamesPerEvaluation)
      gaRef.current = new GeneticAlgorithm(50, 0.1, 0.7, 5, 3) // Reduced from 5 to 3 games per eval
      // Check if strategies were loaded
      const savedInfo = gaRef.current.getSavedDataInfo()
      setSavedDataInfo(savedInfo)
      if (savedInfo) {
        setStats(gaRef.current.getStats())
        setEpisodes(savedInfo.generation)
      }
    }
    
    return () => {
      if (trainingIntervalRef.current) {
        clearInterval(trainingIntervalRef.current)
      }
    }
  }, [method])

  const startTraining = async () => {
    if (isTraining) return
    
    setIsTraining(true)
    isTrainingRef.current = true
    setEpisodes(0)
    
    if (method === 'qlearning') {
      startQLearning()
    } else {
      startGeneticAlgorithm()
    }
  }

  const startQLearning = () => {
    const batchSize = 10
    let currentEpisodes = 0
    
    const interval = setInterval(() => {
      if (!isTrainingRef.current || currentEpisodes >= targetEpisodes) {
        clearInterval(interval)
        trainingIntervalRef.current = null
        setIsTraining(false)
        isTrainingRef.current = false
        return
      }
      
      // Train a batch
      for (let i = 0; i < batchSize && currentEpisodes < targetEpisodes; i++) {
        qAgentRef.current.trainEpisode(gameEngineRef.current)
        currentEpisodes++
      }
      
      setEpisodes(currentEpisodes)
      setStats(qAgentRef.current.getStats())
    }, 10)
    
    trainingIntervalRef.current = interval
  }

  const startGeneticAlgorithm = async () => {
    const startGeneration = gaRef.current.generation
    const targetGen = startGeneration + targetGenerations
    
    const runGeneration = async () => {
      if (!isTrainingRef.current || gaRef.current.generation >= targetGen) {
        setIsTraining(false)
        isTrainingRef.current = false
        trainingIntervalRef.current = null
        return
      }
      
      // Run one generation (this is async)
      const result = await gaRef.current.runGeneration()
      setEpisodes(gaRef.current.generation)
      setStats(gaRef.current.getStats())
      setSavedDataInfo(gaRef.current.getSavedDataInfo())
      
      // Schedule next generation
      if (isTrainingRef.current && gaRef.current.generation < targetGen) {
        setTimeout(runGeneration, 50) // Small delay to allow UI updates
      } else {
        setIsTraining(false)
        isTrainingRef.current = false
        trainingIntervalRef.current = null
      }
    }
    
    runGeneration()
  }

  const stopTraining = () => {
    isTrainingRef.current = false
    setIsTraining(false)
    if (trainingIntervalRef.current) {
      clearInterval(trainingIntervalRef.current)
      trainingIntervalRef.current = null
    }
  }

  const trainSingleEpisode = async () => {
    if (method === 'qlearning') {
      if (qAgentRef.current && gameEngineRef.current) {
        qAgentRef.current.trainEpisode(gameEngineRef.current)
        setEpisodes(prev => prev + 1)
        setStats(qAgentRef.current.getStats())
      }
    } else {
      if (gaRef.current) {
        await gaRef.current.runGeneration()
        setEpisodes(prev => prev + 1)
        setStats(gaRef.current.getStats())
        setSavedDataInfo(gaRef.current.getSavedDataInfo())
      }
    }
  }

  const resetTraining = () => {
    if (method === 'genetic' && gaRef.current) {
      if (window.confirm('Are you sure you want to reset? This will clear all saved strategies and start fresh.')) {
        gaRef.current.reset()
        setEpisodes(0)
        setStats(null)
        setSavedDataInfo(null)
      }
    }
  }

  const exportStrategies = () => {
    if (method === 'genetic' && gaRef.current) {
      const data = gaRef.current.exportStrategies()
      const json = JSON.stringify(data, null, 2)
      const blob = new Blob([json], { type: 'application/json' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `crazyblocks-strategies-gen${data.generation}-${Date.now()}.json`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      URL.revokeObjectURL(url)
    }
  }

  const importStrategies = (event) => {
    const file = event.target.files[0]
    if (!file) return

    const reader = new FileReader()
    reader.onload = (e) => {
      try {
        const data = JSON.parse(e.target.result)
        if (method === 'genetic' && gaRef.current) {
          if (gaRef.current.importStrategies(data)) {
            setStats(gaRef.current.getStats())
            setSavedDataInfo(gaRef.current.getSavedDataInfo())
            setEpisodes(gaRef.current.generation)
            alert('Strategies imported successfully!')
          } else {
            alert('Failed to import strategies. Please check the file format.')
          }
        }
      } catch (error) {
        alert('Error reading file: ' + error.message)
      }
    }
    reader.readAsText(file)
    // Reset input so same file can be selected again
    event.target.value = ''
  }

  const renderStats = () => {
    if (!stats) {
      return <p>No training data yet. Start training to see statistics.</p>
    }

    if (method === 'qlearning') {
      return (
        <div className="stats-grid">
          <div className="stat-item">
            <span className="stat-label">Episodes:</span>
            <span className="stat-value">{episodes} / {targetEpisodes}</span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Average Score:</span>
            <span className="stat-value">{stats.avgScore}</span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Best Score:</span>
            <span className="stat-value">{stats.bestScore}</span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Average Reward:</span>
            <span className="stat-value">{stats.avgReward}</span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Exploration Rate (ε):</span>
            <span className="stat-value">{stats.epsilon}</span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Q-Table Size:</span>
            <span className="stat-value">{stats.qTableSize.toLocaleString()}</span>
          </div>
        </div>
      )
    } else {
      return (
        <div className="stats-grid">
          <div className="stat-item">
            <span className="stat-label">Generation:</span>
            <span className="stat-value">{episodes} / {targetGenerations}</span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Best Fitness:</span>
            <span className="stat-value">{stats.bestFitness}</span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Best Score:</span>
            <span className="stat-value">{stats.bestScore}</span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Average Fitness:</span>
            <span className="stat-value">{stats.avgFitness}</span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Population Size:</span>
            <span className="stat-value">{stats.populationSize}</span>
          </div>
        </div>
      )
    }
  }

  return (
    <div className="training-view">
      <h2>AI Training</h2>
      
      <div className="method-selector">
        <button
          className={method === 'qlearning' ? 'active' : ''}
          onClick={() => setMethod('qlearning')}
          disabled={isTraining}
        >
          Q-Learning
        </button>
        <button
          className={method === 'genetic' ? 'active' : ''}
          onClick={() => setMethod('genetic')}
          disabled={isTraining}
        >
          Genetic Algorithm
        </button>
      </div>
      
      <div className="training-controls">
        <div className="control-group">
          <label>
            {method === 'qlearning' ? 'Target Episodes:' : 'Target Generations:'}
            <input
              type="number"
              value={method === 'qlearning' ? targetEpisodes : targetGenerations}
              onChange={(e) => {
                const value = parseInt(e.target.value) || (method === 'qlearning' ? 1000 : 50)
                if (method === 'qlearning') {
                  setTargetEpisodes(value)
                } else {
                  setTargetGenerations(value)
                }
              }}
              min="1"
              max={method === 'qlearning' ? "100000" : "1000"}
              disabled={isTraining}
            />
          </label>
        </div>
        
        <div className="button-group">
          <button onClick={startTraining} disabled={isTraining}>
            Start Training
          </button>
          <button onClick={stopTraining} disabled={!isTraining}>
            Stop Training
          </button>
          <button onClick={trainSingleEpisode} disabled={isTraining}>
            {method === 'qlearning' ? 'Train 1 Episode' : 'Run 1 Generation'}
          </button>
          {method === 'genetic' && (
            <>
              <button onClick={resetTraining} disabled={isTraining} className="reset-button">
                Reset Strategies
              </button>
              <button onClick={exportStrategies} disabled={isTraining} className="export-button">
                Export Strategies
              </button>
              <label className="import-button">
                Import Strategies
                <input
                  type="file"
                  accept=".json"
                  onChange={importStrategies}
                  disabled={isTraining}
                  style={{ display: 'none' }}
                />
              </label>
            </>
          )}
        </div>
      </div>

      {savedDataInfo && method === 'genetic' && (
        <div className="saved-data-info">
          <h4>📦 Loaded Previous Training</h4>
          <p>
            Continuing from Generation {savedDataInfo.generation} with best score of {savedDataInfo.bestScore} 
            (Fitness: {savedDataInfo.bestFitness.toFixed(2)})
          </p>
          <p className="saved-timestamp">
            Last saved: {new Date(savedDataInfo.timestamp).toLocaleString()}
          </p>
        </div>
      )}

      <div className="training-stats">
        <h3>Training Statistics</h3>
        {renderStats()}
      </div>

      {isTraining && (
        <div className="training-progress">
          <div className="progress-bar">
            <div
              className="progress-fill"
              style={{ 
                width: `${((method === 'qlearning' 
                  ? episodes / targetEpisodes 
                  : (episodes - (savedDataInfo?.generation || 0)) / targetGenerations) * 100)}%` 
              }}
            />
          </div>
          <p>
            Training in progress... {episodes} {method === 'qlearning' ? `episodes` : `generations`}
            {method === 'genetic' && savedDataInfo && ` (${episodes - savedDataInfo.generation} new this session)`}
          </p>
        </div>
      )}
      
      {method === 'genetic' && (
        <div className="info-box">
          <h4>🧬 Genetic Algorithm & Your Artifact</h4>
          <p>
            <strong>What you're building:</strong> A collection of evolved AI strategies (weights/heuristics) 
            that determine how the AI plays the game. Each strategy is a set of 7 numerical weights that 
            control decision-making.
          </p>
          <p>
            <strong>Where it's stored:</strong> 
          </p>
          <ul>
            <li><strong>Browser localStorage:</strong> Auto-saved after each generation</li>
            <li><strong>Export file:</strong> Click "Export Strategies" to download as JSON</li>
            <li><strong>Import:</strong> Load previously exported strategies from any device</li>
          </ul>
          <p>
            <strong>How it evolves:</strong>
          </p>
          <ul>
            <li>Evaluates each agent's performance over multiple games</li>
            <li>Selecting the best performers (elite)</li>
            <li>Creating new strategies through crossover and mutation</li>
            <li>Evolving better strategies over generations</li>
          </ul>
          {savedDataInfo && (
            <p className="artifact-info">
              <strong>Current Artifact:</strong> Generation {savedDataInfo.generation} with {savedDataInfo.populationSize} strategies. 
              Best score: {savedDataInfo.bestScore} (Fitness: {savedDataInfo.bestFitness.toFixed(2)})
            </p>
          )}
        </div>
      )}
    </div>
  )
}

export default TrainingView

