import pygame
from time import sleep
import copy
import DQN2 
from sys import exit

class GameRendering:
    def __init__(self, game, agent, Config):
        pygame.init()
        pygame.font.init()
        
        self.side_length = 100

        self.game = copy.deepcopy(game)
        self.done = False
        self.height = game.num_rows
        self.width = game.num_columns
        self.imagerect = (0, 0)
        self.black = (0,0,0)
        self.white = (255, 255, 255)
        self.red = (255, 0, 0)
        self.green = (0, 255, 0)
        self.open = (125, 125, 125)
        self.blue = (0,0,125)
        self.screen = pygame.display.set_mode([self.side_length*2 + self.width*100, self.side_length + 100* self.height])
        self.text = pygame.font.SysFont('Comic Sans MS', 30)
        self.mark = 1

        while True:
            self.mouse_pos = pygame.mouse.get_pos()
            self.render(self.white, self.black, self.red, self.green, self.white)
            for event in pygame.event.get():
                if event.type == pygame.MOUSEBUTTONUP:
                    pos = pygame.mouse.get_pos()
                    observations , valid, done, _ = self.game.step(pos[0] // 100 - 1, self.mark)
                    
                    if done:
                        self.done = True
                        self.render(self.white, self.black, self.red, self.green, self.white)
                        sleep(4)
                        pygame.quit()
                    elif not valid:
                        pygame.quit()
                    else:
                        observations = self.game.flip()
                        print(observations)
                        self.mark = self.mark % 2 + 1
                        action = agent.get_action(observations, 0)
                        observations , valid, done, _ = self.game.step(action, self.mark)
                        
                        if done:
                            self.done = True
                            self.render(self.white, self.black, self.red, self.green, self.white)
                            sleep(4)
                            pygame.quit()
                        elif not valid:
                            pygame.quit()
                        self.mark = self.mark % 2 + 1
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()

    def render(self, background, line, p1_color, p2_color, open_color):
        self.screen.fill(background)
        w = 0
        h = 0
        mark = self.mark
        if not self.done:
            textsurface = self.text.render("Spiller: " + str(mark), True, (0, 0, 0))
            self.screen.blit(textsurface,(400,0))
        else:
            textsurface = self.text.render("Spiller:" + str(mark) + " Vant", True, (0, 0, 0))
            self.screen.blit(textsurface,(400,0))
        #vertical lines
        for line in range(self.width+1):
            pygame.draw.line(self.screen, self.black, [line*100 + self.side_length, 50], [line*100 + self.side_length, self.height*100 + 50],3)
        for line in range(self.height+1):
            pygame.draw.line(self.screen, self.black, [self.side_length, line*100 +50], [self.side_length+self.width * 100, line*100 + 50],3)
        for space in range(self.width * self.height):
            if self.game.board[space] == 0:
                if space < 7 and self.mouse_pos[0] > self.side_length + w*100 and self.mouse_pos[0] < self.side_length + w*100 + 100:
                    pygame.draw.circle(self.screen, self.green, [self.side_length + w*100 + 50, h * 100 + 100], 45)
                else:
                    pygame.draw.circle(self.screen, self.open, [self.side_length + w*100 + 50, h * 100 + 100], 45)
            if self.game.board[space] == 1:
                pygame.draw.circle(self.screen, self.red, [self.side_length + w*100 + 50, h * 100 + 100], 45)
            if self.game.board[space] == 2:
                pygame.draw.circle(self.screen, self.blue, [self.side_length + w*100 + 50, h * 100 + 100], 45)
            w += 1
            if w == self.width:
                w = 0
                h+= 1
        pygame.event.pump()
        pygame.display.flip()

gamma = 0.99
copy_step = 25
max_exp = 100000
min_exp = 100
batch_size = 32
learning_rate = 0.00146
epsilon = 0.05
decay = 0.999
min_epsilon = 0.01
episodes = 200000


template_gym = DQN2.environment.ConnectXGym()
Opponent = DQN2.DQN2(template_gym.positions.n, template_gym.actions.n, gamma, max_exp, min_exp, batch_size, learning_rate)
Opponent.load_weights('lookahead_vs_verticalbot2')
game = DQN2.ConnectXEnvironment(7,6,4)
GameRendering(game,Opponent,0)
