# Crazy Blocks

A puzzle game that combines elements of Tetris and Candy Crush! Click on blocks to eliminate connected groups of the same color.

## How to Play

1. Each turn, a new row of 5 randomly colored blocks appears at the bottom, pushing all blocks above up by one row.
2. Click on any block to eliminate all connected blocks of the same color (adjacent horizontally or vertically, not diagonally).
3. After elimination, blocks above drop down to fill empty spaces.
4. Score 1 point for each block eliminated.
5. The game ends when any stack reaches the top row (17 blocks high).

## Getting Started

### Prerequisites

- Node.js (v14 or higher)
- npm or yarn

### Installation

1. Install dependencies:
```bash
npm install
```

2. Start the development server:
```bash
npm run dev
```

3. Open your browser and navigate to the URL shown in the terminal (usually `http://localhost:5173`)

### Build for Production

```bash
npm run build
```

The built files will be in the `dist` directory.

## Technologies Used

- React 18
- Vite
- CSS3

## Game Features

- 5 columns × 17 rows grid
- 4 different block colors
- Flood fill algorithm for connected block detection
- Gravity system for block dropping
- Score tracking
- Game over detection
- Restart functionality

