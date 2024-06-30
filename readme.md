This is a quick project to learn the basics of genetic algorithms and reinforcement learning

---library files---

ponggame.py - A simple single player pong game made with pygame with its internal state exposed
model.py - Defines a simple multilayer perceptron neural network and an agent wrapper for it

---files to run---

supervisedGeneratePlay.py - Runs the pong with the human in control and dumps the frame and control states to a csv file after you shut the game off. Dumps to playerOutput.csv
dataBalance.py - Turns the raw player data output into a balanced dataset of states where the bat moves left and states where the bat moves right. Dumps to balancedPlayerOutput.csv
aiTrain.py - first does supervised training to make a neural network match the players output for a given game state then uses that as the initial solution for a genetic algorithm that maximises game score. Saves to bestModel.pk
aiplay.py - Runs the pong game and puts the saved ai agent in bestModel.pk in control
trivialplay.py - Runs the pong game using a trivial one line solution that tries to align the paddle to the ball. This trivial solution makes this a toy project