import React, { useState } from 'react'
import Game from './components/Game'
import TrainingView from './components/TrainingView'
import './App.css'

function App() {
  const [view, setView] = useState('game') // 'game' or 'training'

  return (
    <div className="App">
      <nav className="app-nav">
        <button
          className={view === 'game' ? 'active' : ''}
          onClick={() => setView('game')}
        >
          Play Game
        </button>
        <button
          className={view === 'training' ? 'active' : ''}
          onClick={() => setView('training')}
        >
          AI Training
        </button>
      </nav>
      {view === 'game' ? <Game /> : <TrainingView />}
    </div>
  )
}

export default App

