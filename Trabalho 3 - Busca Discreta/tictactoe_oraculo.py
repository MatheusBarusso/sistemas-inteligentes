# -*- coding: utf-8 -*-
"""
Recriacão do Jogo da Velha

@author: Prof. Daniel Cavalcanti Jeronymo

Para usar as dicas do oraculo -> Pressione a tecla 'h' enquanto joga
"""

import pygame
    
import sys
import os
import traceback
import random
import numpy as np
import copy

class GameConstants:
    #                  R    G    B
    ColorWhite     = (255, 255, 255)
    ColorBlack     = (  0,   0,   0)
    ColorRed       = (255,   0,   0)
    ColorGreen     = (  0, 255,   0)
    ColorBlue     = (  0, 0,   255)
    ColorDarkGreen = (  0, 155,   0)
    ColorDarkGray  = ( 40,  40,  40)
    BackgroundColor = ColorBlack
    
    screenScale = 1
    screenWidth = screenScale*600
    screenHeight = screenScale*600
    
    # grid size in units
    gridWidth = 3
    gridHeight = 3
    
    # grid size in pixels
    gridMarginSize = 5
    gridCellWidth = screenWidth//gridWidth - 2*gridMarginSize
    gridCellHeight = screenHeight//gridHeight - 2*gridMarginSize
    
    randomSeed = 0
    
    FPS = 30
    
    fontSize = 20

class TicTacToeOracle:
    def __init__(self):
        self.best_move = None
    
    def get_empty_cells(self, grid):
        empty = []
        for i in range(GameConstants.gridHeight):
            for j in range(GameConstants.gridWidth):
                if grid[i][j] == 0:
                    empty.append((i, j))
        return empty
    
    def is_game_over(self, grid):
        # Horizontal
        for i in range(3):
            if grid[i][0] == grid[i][1] == grid[i][2] != 0:
                return True
        # Vertical
        for i in range(3):
            if grid[0][i] == grid[1][i] == grid[2][i] != 0:
                return True
        # Diagonals
        if grid[0][0] == grid[1][1] == grid[2][2] != 0:
            return True
        if grid[0][2] == grid[1][1] == grid[2][0] != 0:
            return True
        # Check for draw
        if len(self.get_empty_cells(grid)) == 0:
            return True
        return False
    
    def evaluate(self, grid, maximizing_player):
        # Check for victory conditions
        for i in range(3):
            if grid[i][0] == grid[i][1] == grid[i][2]:
                if grid[i][0] == maximizing_player:
                    return 10
                elif grid[i][0] != 0:
                    return -10
                    
        for i in range(3):
            if grid[0][i] == grid[1][i] == grid[2][i]:
                if grid[0][i] == maximizing_player:
                    return 10
                elif grid[0][i] != 0:
                    return -10
        
        if grid[0][0] == grid[1][1] == grid[2][2]:
            if grid[0][0] == maximizing_player:
                return 10
            elif grid[0][0] != 0:
                return -10
                
        if grid[0][2] == grid[1][1] == grid[2][0]:
            if grid[0][2] == maximizing_player:
                return 10
            elif grid[0][2] != 0:
                return -10
                
        return 0

    # DFS-based minimax: usa busca em profundidade recursiva e penaliza vitórias mais longas
    def dfs(self, grid, depth, is_maximizing, player, opponent):
        score = self.evaluate(grid, player)
        if score == 10:
            return 10 - depth  # vitória do player: quanto menor depth, melhor
        if score == -10:
            return -10 + depth  # derrota do player: quanto maior depth, "menos pior"
        if self.is_game_over(grid):
            return 0
            
        if is_maximizing:
            best = -10000
            for move in self.get_empty_cells(grid):
                i, j = move
                grid[i][j] = player
                val = self.dfs(grid, depth + 1, False, player, opponent)
                grid[i][j] = 0
                if val > best:
                    best = val
            return best
        else:
            best = 10000
            for move in self.get_empty_cells(grid):
                i, j = move
                grid[i][j] = opponent
                val = self.dfs(grid, depth + 1, True, player, opponent)
                grid[i][j] = 0
                if val < best:
                    best = val
            return best
    
    def find_best_move(self, grid, player):
        opponent = 1 if player == 2 else 2
        best_value = -10000
        self.best_move = (-1, -1)
        
        for move in self.get_empty_cells(grid):
            i, j = move
            grid[i][j] = player
            move_value = self.dfs(grid, 0, False, player, opponent)
            grid[i][j] = 0
            
            if move_value > best_value:
                self.best_move = move
                best_value = move_value
                
        return self.best_move

class Game:
    class GameState:
        # 0 empty, 1 X, 2 O
        grid = np.zeros((GameConstants.gridHeight, GameConstants.gridWidth))
        currentPlayer = 0
    
    def __init__(self, expectUserInputs=True):
        self.expectUserInputs = expectUserInputs
        
        # Game state list - stores a state for each time step (initial state)
        gs = Game.GameState()
        self.states = [gs]
        
        # Determines if simulation is active or not
        self.alive = True
        
        self.currentPlayer = 1
        
        # Journal of inputs by users (stack)
        self.eventJournal = []
        
        self.oracle = TicTacToeOracle()
        
    def checkObjectiveState(self, gs):
        # Complete line?
        for i in range(3):
            s = set(gs.grid[i, :])
            if len(s) == 1 and min(s) != 0:
                return s.pop()
            
        # Complete column?
        for i in range(3):
            s = set(gs.grid[:, i])
            if len(s) == 1 and min(s) != 0:
                return s.pop()
            
        # Complete diagonal (main)?
        s = set([gs.grid[i, i] for i in range(3)])
        if len(s) == 1 and min(s) != 0:
            return s.pop()
        
        # Complete diagonal (opposite)?
        s = set([gs.grid[-i-1, i] for i in range(3)])
        if len(s) == 1 and min(s) != 0:
            return s.pop()
            
        # nope, not an objective state
        return 0
    
    
    # Implements a game tick
    # Each call simulates a world step
    def update(self):  
        # If the game is done or there is no event, do nothing
        if not self.alive or not self.eventJournal:
            return
        
        # Get the current (last) game state
        gs = copy.copy(self.states[-1])
        
        # Switch player turn
        if gs.currentPlayer == 0:
            gs.currentPlayer = 1
        elif gs.currentPlayer == 1:
            gs.currentPlayer = 2
        elif gs.currentPlayer == 2:
            gs.currentPlayer = 1
            
        # Mark the cell clicked by this player if it's an empty cell
        x,y = self.eventJournal.pop()

        # Check if in bounds
        if x < 0 or y < 0 or x >= GameConstants.gridCellHeight or y >= GameConstants.gridCellWidth:
            return

        # Check if cell is empty
        if gs.grid[x][y] == 0:
            gs.grid[x][y] = gs.currentPlayer
        else: # invalid move
            return
        
        # Check if end of game
        if self.checkObjectiveState(gs):
            self.alive = False
                
        # Add the new modified state
        self.states += [gs]

    def get_oracle_move(self):
        gs = self.states[-1]
        return self.oracle.find_best_move(gs.grid, gs.currentPlayer)

       

def drawGrid(screen, game):
    rects = []

    rects = [screen.fill(GameConstants.BackgroundColor)]
    
    # Get the current game state
    gs = game.states[-1]
    grid = gs.grid
 
    # Draw the grid
    for row in range(GameConstants.gridHeight):
        for column in range(GameConstants.gridWidth):
            color = GameConstants.ColorWhite
            
            if grid[row][column] == 1:
                color = GameConstants.ColorRed
            elif grid[row][column] == 2:
                color = GameConstants.ColorBlue
            
            m = GameConstants.gridMarginSize
            w = GameConstants.gridCellWidth
            h = GameConstants.gridCellHeight
            rects += [pygame.draw.rect(screen, color, [(2*m+w) * column + m, (2*m+h) * row + m, w, h])]    
    
    return rects


def draw(screen, font, game):
    rects = []
            
    rects += drawGrid(screen, game)

    return rects


def initialize():
    random.seed(GameConstants.randomSeed)
    pygame.init()
    game = Game()
    font = pygame.font.SysFont('Courier', GameConstants.fontSize)
    fpsClock = pygame.time.Clock()

    # Create display surface
    screen = pygame.display.set_mode((GameConstants.screenWidth, GameConstants.screenHeight), pygame.DOUBLEBUF)
    screen.fill(GameConstants.BackgroundColor)
        
    return screen, font, game, fpsClock


def handleEvents(game):
    gs = game.states[-1]
    
    for event in pygame.event.get():
        if event.type == pygame.MOUSEBUTTONUP:
            pos = pygame.mouse.get_pos()
            
            col = pos[0] // (GameConstants.screenWidth // GameConstants.gridWidth)
            row = pos[1] // (GameConstants.screenHeight // GameConstants.gridHeight)
            
            # send player action to game
            game.eventJournal.append((row, col))
        
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_h:
            best_move = game.get_oracle_move()
            print(f"Movimento sugerido (Linha, Coluna): {best_move}")
            
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            pygame.quit()
            sys.exit()

            
def mainGamePlayer():
    try:
        # Initialize pygame and etc.
        screen, font, game, fpsClock = initialize()
              
        # Main game loop
        while game.alive:
            # Handle events
            handleEvents(game)
                    
            # Update world
            game.update()
            
            # Draw this world frame
            rects = draw(screen, font, game)     
            pygame.display.update(rects)
            
            # Delay for required FPS
            fpsClock.tick(GameConstants.FPS)
            
        # close up shop
        pygame.quit()
    except SystemExit:
        pass
    except Exception as e:
        #print("Unexpected error:", sys.exc_info()[0])
        traceback.print_exc(file=sys.stdout)
        pygame.quit()
        #raise Exception from e
    
    
if __name__ == "__main__":
    # Set the working directory (where we expect to find files) to the same
    # directory this .py file is in. You can leave this out of your own
    # code, but it is needed to easily run the examples using "python -m"
    file_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(file_path)

    mainGamePlayer()