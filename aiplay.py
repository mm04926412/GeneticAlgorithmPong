import model
import ponggame
from pickle import load

if __name__ == '__main__':
    agent = load(open("best_model.pk","rb"))
    ponggame.runGameAgent(agent)
