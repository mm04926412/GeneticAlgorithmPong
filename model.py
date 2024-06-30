import torch
import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes=2):
        super(SimpleMLP, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size  # Store hidden_size as an attribute
        self.num_classes = num_classes
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x
    
class pongAgent():
    @staticmethod
    def processInput(game):
        ball_x = (game.ball_x-400)/400
        ball_y = (game.ball_x-400)/400
        ball_dx = game.ball_dx/5
        ball_dy = game.ball_dy/5
        paddle_x = game.paddle_x/5
        return torch.tensor([ball_x,ball_y,ball_dx,ball_dy,paddle_x], dtype=torch.float32)

    def __init__(self,hidden_size=16):
        #Input: [ball_x, ball_y, ball_x, ball_dy, paddle_x]
        self.model = SimpleMLP(5,hidden_size,2)
        self.model.eval()
    
    def getOutput(self,game):
        input = pongAgent.processInput(game)
        output = self.model(input)
        return output
    



