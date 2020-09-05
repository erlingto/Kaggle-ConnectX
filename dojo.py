import gym
import game_environment
import agents
import numpy as np


class ConnectXGym2(gym.Env):
    def __init__(self):
        self.env = game_environment.ConnectXEnvironment(7, 6, 4)
    
        self.trainer = 0
        
      
        self.columns = self.env.num_columns
        self.rows = self.env.num_rows
        self.actions = gym.spaces.Discrete(self.columns)
        self.positions = gym.spaces.Discrete(self.columns * self.rows)
        self.list_of_trainers = ["variety14.0" ,"variety5.0", "variety3.0", "variety4.0","variety8.0", "variety9.0", "variety19.0", "lookahead_vs_verticalbot1","variety20.0", "variety12.0", "variety13.0"]
        self.score_list = {i : 0 for i in self.list_of_trainers}
        self.games_list = {i : 0 for i in self.list_of_trainers}
        self.change_trainer_at_random()
    
    def reset_scores(self):
        self.score_list = {i : 0 for i in self.list_of_trainers}
        self.games_list = {i : 0 for i in self.list_of_trainers}
    def print_scores(self):
        for i in self.score_list:
            print(i, self.score_list[i], end = "")
            print("/", self.games_list[i])
    
    def change_trainer(self, trainer):
        self.trainer = trainer

    def change_trainer_at_random(self):
        choice = np.random.choice(self.list_of_trainers)
        if choice == "verticalbot":
            trainer = agents.verticalBot(7,6)
            self.change_trainer(trainer)
        else:
            trainer = OpponentDQN(42,7)
            trainer.load_weights(choice)
            self.change_trainer(trainer)
        

    def step(self, action, mark):
        return self.env.step(action, mark)

    def render(self, **kwargs):
        return self.env.render(**kwargs)

    def generate_data(self, TrainNet, TargetNet, epsilon, copy_step):
        rewards = 0
        iter = 0
        opp_action = 0
        done = False

        env = self.env
        
        trainee_mark, observations = env.reset()
        
        if trainee_mark == 1:
            TrainNet.mark = 1
            self.trainer.mark = 2
            opp_mark = 2
                 
            while not done:
                '''trainee makes a move '''
                action = TrainNet.get_action(observations, epsilon)

                prev_observations = np.array(observations)
        
                observations, valid, done, reward = env.step(action, trainee_mark)

                reward = reward[trainee_mark-1]
                if not done:
                    ''' opponent makes a move '''
                    ''' flip_the board '''
                    observations = env.flip()
                    opp_action = self.trainer.get_action(observations, 0.03)

                    observations, valid, done, reward = env.step(opp_action, opp_mark)
                    reward = reward[trainee_mark-1]

                if done:
                    rewards += reward

                exp = {'prev_obs': prev_observations, 'a' : action, 'r': reward, 'obs': observations, 'done' : done }
                TrainNet.add_experience(exp)

                loss = TrainNet.train(TargetNet)
                iter += 1
                if iter % copy_step == 0:
                    TargetNet.copy_weights(TrainNet)
            return rewards, loss
        else:
            TrainNet.mark = 2
            self.trainer.mark = 1
            opp_mark = 1     
            ''' opponent makes a move '''
            opp_action = self.trainer.get_action(observations, 0.03)

            observations, valid, done, reward = env.step(opp_action, opp_mark)
            reward = reward[trainee_mark-1]

            while not done:
                observations = env.flip()
                action = TrainNet.get_action(observations, epsilon)

                prev_observations = np.array(observations)
        
                observations, valid, done, reward = env.step(action, trainee_mark)
                reward = reward[trainee_mark-1]
                if not done:
                    ''' opponent makes a move '''
                    opp_action = self.trainer.get_action(observations, 0.03)

                    observations, valid, done, reward = env.step(opp_action, opp_mark)
                    reward = reward[trainee_mark-1]

                if done:
                    rewards += reward

                exp = {'prev_obs': prev_observations, 'a' : action, 'r': reward, 'obs': env.flip(), 'done' : done }
                TrainNet.add_experience(exp)

                loss = TrainNet.train(TargetNet)
                iter += 1
                if iter % copy_step == 0:
                    TargetNet.copy_weights(TrainNet)
            return rewards, loss

def dojo(games, gym, TrainNet, TargetNet, min_epsilon, epsilon, copy_step):
    total_loss = 0
    even_match = 0
    decay = 0.999
    for i in range(games):
        rewards, loss = gym.generate_data(TrainNet, TargetNet, epsilon, copy_step)
        if rewards == 0:
            even_match += 1
        gym.render()
        print("motstander", gym.trainer.mark)
        print("SCORE:", rewards)
        gym.score_list[gym.trainer.name] += rewards
        gym.games_list[gym.trainer.name] += 20
        total_loss += loss
        print(i)
        if i%10 == 0 and i != 0:
            gym.change_trainer_at_random()
        if i%2 == 0 and i !=  0:
            epsilon = max(min_epsilon, epsilon*decay)
        if i%100 == 0 and i != 0:
            print('Total Loss:', total_loss)
            print('Even matches:', even_match)
            gym.print_scores()
            gym.reset_scores()
            even_match = 0
            total_loss = 0
            print("games", i)
            print("epsilon", epsilon)
        if i%50000 == 0 and i != 0:
            plt = plot_grad_flow(TrainNet.model.named_parameters())
            path = "plot" + str(i)+ ".png"
            plt.savefig(path)
