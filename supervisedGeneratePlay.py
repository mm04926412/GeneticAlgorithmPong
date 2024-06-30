import model
import pongGame
import pandas as pd

if __name__ == '__main__':
    #Runs the pong game and saves the game and control states to a csv for supervised training
    pongGame.runGameHuman(log=True)
