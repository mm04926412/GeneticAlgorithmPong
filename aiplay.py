import model
import pongGame
from pickle import load

if __name__ == '__main__':
    agent = load(open("bestModel.pk","rb"))
    pongGame.runGameManualAI(agent)
