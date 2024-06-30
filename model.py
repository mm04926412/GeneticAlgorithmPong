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
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
class pongAgent():
    def __init__(self,hidden_size=16):
        #Input: [ball_x, ball_y, ball_dx, ball_dy, paddle_x]
        self.model = SimpleMLP(5,hidden_size,2)
    
    def getOutput(self,game):
        input = torch.tensor([game.ball_x,game.ball_y,game.ball_dx,game.ball_dy,game.paddle_x], dtype=torch.float32)
        self.model.eval()
        output = self.model(input)
        return output
    



