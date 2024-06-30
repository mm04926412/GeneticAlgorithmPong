import pygame
import random
import model
import pandas as pd
from time import sleep


# Game constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
PADDLE_WIDTH = 100
PADDLE_HEIGHT = 10
BALL_SIZE = 10
BALL_SPEED = 5
FPS = 60

class Game:
    def __init__(self):
        self.paddle_x = SCREEN_WIDTH // 2 - PADDLE_WIDTH // 2
        self.ball_x = SCREEN_WIDTH // 2
        self.ball_y = SCREEN_HEIGHT // 2
        self.ball_dx = random.choice([-1, 1]) * BALL_SPEED * random.uniform(0,1)
        self.ball_dy = BALL_SPEED
        self.score = 0

    def update(self,left = False, right = False):
        # Move the paddle
        if left:
            self.paddle_x = max(0, self.paddle_x - 10)
        if right:
            self.paddle_x = min(SCREEN_WIDTH - PADDLE_WIDTH, self.paddle_x + 10)

        # Move the ball
        self.ball_x += self.ball_dx
        self.ball_y += self.ball_dy

        # Ball collision with walls
        if self.ball_x <= 0 or self.ball_x >= SCREEN_WIDTH - BALL_SIZE:
            self.ball_dx *= -1

        # Ball collision with top wall (score point)
        if self.ball_y <= 0:
            self.ball_dy *= -1
            self.score += 1

        # Ball collision with paddle
        if (self.ball_y >= SCREEN_HEIGHT - PADDLE_HEIGHT - BALL_SIZE and
            self.paddle_x < self.ball_x < self.paddle_x + PADDLE_WIDTH):
            self.ball_dy = -BALL_SPEED
            self.ball_dx = min(max(self.ball_dx + random.uniform(-3,3) + right - left,-5),5)

        # Reset ball if it goes below the paddle
        if self.ball_y > SCREEN_HEIGHT:
            self.ball_x = SCREEN_WIDTH // 2
            self.ball_y = SCREEN_HEIGHT // 2
            self.ball_dx = random.choice([-1, 1]) * BALL_SPEED * random.uniform(0,1)
            self.ball_dy = BALL_SPEED

def runGameHuman(log = False):
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("AI Pong")

    game = Game()

    running = True

    if log:
        state_timeline = pd.DataFrame(columns=["ball_x","ball_y","ball_dx","ball_dy","paddle_x","leftHeld","RightHeld"])

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if log:
            state_timeline.loc[len(state_timeline)] = [game.ball_x,game.ball_y,game.ball_dx,game.ball_dy,game.paddle_x,keys[pygame.K_LEFT],keys[pygame.K_RIGHT]]
        game.update(left=keys[pygame.K_LEFT], right=keys[pygame.K_RIGHT])

        # Drawing
        screen.fill((0, 0, 0))
        pygame.draw.rect(screen, (255, 255, 255), (game.paddle_x, SCREEN_HEIGHT - PADDLE_HEIGHT, PADDLE_WIDTH, PADDLE_HEIGHT))
        pygame.draw.rect(screen, (255, 255, 255), (game.ball_x, game.ball_y, BALL_SIZE, BALL_SIZE))
        
        font = pygame.font.Font(None, 36)
        score_text = font.render(f"Score: {game.score}", True, (255, 255, 255))
        screen.blit(score_text, (10, 10))
        sleep(1/60)
        pygame.display.flip()
    if log:
        state_timeline.to_csv("playerOutput.csv")
    pygame.quit()

def runGameAgent(agent):
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Single Player Pong")

    game = Game()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = agent.getOutput(game)
        print(keys)
        game.update(left=keys[0] < 0.5, right=keys[0] >= 0.5)

        # Drawing
        screen.fill((0, 0, 0))
        pygame.draw.rect(screen, (255, 255, 255), (game.paddle_x, SCREEN_HEIGHT - PADDLE_HEIGHT, PADDLE_WIDTH, PADDLE_HEIGHT))
        pygame.draw.rect(screen, (255, 255, 255), (game.ball_x, game.ball_y, BALL_SIZE, BALL_SIZE))
        
        font = pygame.font.Font(None, 36)
        score_text = font.render(f"Score: {game.score}", True, (255, 255, 255))
        screen.blit(score_text, (10, 10))

        pygame.display.flip()

    pygame.quit()

def runGameManualAI(agent):
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Single Player Pong")

    game = Game()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if game.paddle_x < game.ball_x:
            game.update(left=False, right=True)
        else:
            game.update(left=True, right=False)

        # Drawing
        screen.fill((0, 0, 0))
        pygame.draw.rect(screen, (255, 255, 255), (game.paddle_x, SCREEN_HEIGHT - PADDLE_HEIGHT, PADDLE_WIDTH, PADDLE_HEIGHT))
        pygame.draw.rect(screen, (255, 255, 255), (game.ball_x, game.ball_y, BALL_SIZE, BALL_SIZE))
        
        font = pygame.font.Font(None, 36)
        score_text = font.render(f"Score: {game.score}", True, (255, 255, 255))
        screen.blit(score_text, (10, 10))

        pygame.display.flip()

    pygame.quit()

def runLogicAgent(agent,frames=1000):
    game = Game()
    oldBallY = SCREEN_HEIGHT
    for _ in range(frames):
        keys = agent.getOutput(game)
        game.update(left=keys[0] < 0.5, right=keys[0] >= 0.5)
        ballY = game.ball_y
        if ballY > oldBallY+100:
            return game
    return game



def main():
    runGameHuman()

if __name__ == "__main__":
    main()