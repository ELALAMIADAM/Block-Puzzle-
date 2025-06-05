# 🌳 Wood Block Puzzle 9x9

A beautiful and addictive block puzzle game built with React, featuring drag-and-drop mechanics, wooden aesthetics, and unique 3x3 square clearing!

## 🎮 Game Features

- **9x9 Grid**: Strategic puzzle board with smooth wooden theme
- **Drag & Drop**: Intuitive block placement with visual feedback
- **Triple Clearing**: Complete rows, columns, OR 3×3 squares to earn points
- **3x3 Squares**: Visual grid lines show the nine 3×3 squares for easy identification
- **Scoring System**: Points for block placement and line clearing
- **Best Score Tracking**: Persistent high score storage
- **Pause/Resume**: Game state management
- **Responsive Design**: Works on desktop and mobile devices
- **Beautiful UI**: Wooden theme with gradients and animations

## 🎯 How to Play

1. **Drag blocks** from the tray at the bottom onto the 9×9 game board
2. **Complete rows, columns, or 3×3 squares** to clear them and earn bonus points
3. **Use the visual grid lines** to identify the nine 3×3 squares (like Sudoku!)
4. **Strategically place blocks** to maximize clearing opportunities
5. **Game ends** when no available blocks can fit on the board
6. **Beat your high score** and challenge yourself!

## 🚀 Getting Started

### Prerequisites

- Node.js (version 14 or higher)
- npm or yarn package manager

### Installation

1. **Install dependencies**:
   ```bash
   npm install
   ```

2. **Start the development server**:
   ```bash
   npm start
   ```

3. **Open your browser** and navigate to `http://localhost:3000`

### Building for Production

```bash
npm run build
```

This creates an optimized production build in the `build` folder.

## 🛠️ Technologies Used

- **React 18** - Modern React with hooks
- **React DnD** - Drag and drop functionality
- **CSS3** - Advanced styling with gradients and animations
- **HTML5** - Semantic markup
- **Local Storage** - Persistent score tracking

## 📁 Project Structure

```
src/
├── components/
│   ├── GameBoard.js      # Main game grid with drop functionality
│   ├── BlockTray.js      # Container for available blocks
│   ├── DraggableBlock.js # Individual draggable block component
│   ├── ScoreDisplay.js   # Score and statistics display
│   └── GameOverModal.js  # Game over screen
├── utils/
│   └── gameLogic.js      # Game logic and block generation
├── App.js               # Main application component
├── index.js             # React application entry point
└── index.css            # Global styles and wooden theme
```

## 🎨 Customization

### Adding New Block Shapes

Edit `src/utils/gameLogic.js` and add new shapes to the `BLOCK_SHAPES` array:

```javascript
// Example: Add a new cross shape
[[false, true, false],
 [true, true, true],
 [false, true, false]]
```

### Changing the Theme

Modify the CSS variables in `src/index.css` to customize colors:

```css
/* Change wood colors */
.grid-cell {
  background: linear-gradient(145deg, #YourColor1, #YourColor2);
}
```

### Adjusting Game Difficulty

- Modify `GRID_SIZE` in `App.js` to change board size
- Adjust scoring values in the `placeBlock` function
- Change the number of available blocks in `generateRandomBlocks()`

## 🎵 Optional Enhancements

The game is ready for additional features:

- **Sound Effects**: Add audio feedback for block placement and line clearing
- **Animations**: Enhanced block placement and line clearing animations
- **Power-ups**: Special blocks with unique abilities
- **Themes**: Multiple visual themes (dark mode, seasonal themes)
- **Leaderboards**: Online high score sharing
- **Daily Challenges**: Special puzzle configurations

## 🐛 Troubleshooting

### Common Issues

1. **Blocks not dragging**: Ensure React DnD backend is properly initialized
2. **Styling issues**: Check if CSS classes are being applied correctly
3. **Performance issues**: Consider optimizing the grid rendering for larger boards

### Development Tips

- Use React Developer Tools for debugging component state
- Check browser console for any JavaScript errors
- Test drag and drop functionality on different devices

## 📱 Mobile Support

The game includes responsive design for mobile devices:
- Touch-friendly drag and drop
- Optimized cell sizes for smaller screens
- Adapted layout for portrait orientation

## 🤝 Contributing

Feel free to contribute to this project:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 License

This project is open source and available under the MIT License.

## 🎉 Enjoy Playing!

Have fun with your Wood Block Puzzle game! The game automatically saves your best score, so you can always try to beat your personal record.

---

*Made with ❤️ using React* 