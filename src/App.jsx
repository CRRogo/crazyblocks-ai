import React, { useState } from 'react'
import Game from './components/Game'
import TrainingView from './components/TrainingView'
import './App.css'

function App() {
  const [view, setView] = useState('game') // 'game' or 'training'

  return (
    <div className="App">
      {view === 'game' ? <Game /> : <TrainingView />}
    </div>
  )
}

export default App

