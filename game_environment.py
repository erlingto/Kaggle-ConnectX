import numpy as np

class ConnectXEnvironment:
    def __init__(self, num_columns, num_rows, connect):
        self.connect = connect
        self.num_columns = num_columns
        self.num_rows = num_rows
        self.size = num_rows * num_columns
        self.board = np.zeros(num_columns* num_rows, dtype = int)
        self.marks = [1, 2]
        self.done = False
    
    def flip(self):
        flipped_board = np.array(self.board)
        for i in range(len(self.board)):
            if flipped_board[i] == self.marks[0]:
                flipped_board[i] = self.marks[1]
            elif flipped_board[i] == self.marks[1]:
                flipped_board[i] = self.marks[0]

        return flipped_board
    def check(self, position):
        mark = self.board[position]
        reward = [0, 0]
        done = False

        j = 0
        for i in range(self.num_columns):
            if not self.board[i] == 0:
                j +=1
                if j == self.num_columns:
                    done = True
                    reward = [0, 0]
                    return done, reward

        column = position % self.num_columns 
        row = int((position - column) / self.num_columns)
        
        inverse_row = self.num_rows-1-row
        
        diagonal = column + row
        inverse_diagonal = inverse_row + column
        
       
        if diagonal < self.num_rows:
            diagonal_bottom = diagonal * self.num_columns 
        else:
            diagonal_bottom =  self.num_columns * (self.num_rows-1) + (diagonal- self.num_rows) + 1 
        
        if inverse_diagonal < self.num_rows:
            inverse_diagonal_row = inverse_diagonal
            inverse_diagonal_column = 0
            diagonal_top = self.num_columns * (self.num_rows - inverse_diagonal_row-1)
        else:
            
            inverse_diagonal_column = inverse_diagonal - self.num_rows + 1
            inverse_diagonal_row = self.num_rows - 1
            diagonal_top = inverse_diagonal - self.num_rows + 1

        diagonal_column = diagonal_bottom % self.num_columns 
        diagonal_row = int((diagonal_bottom - diagonal_column) / self.num_columns)


        ''' positions '''
        position_diagonal_ur = diagonal_bottom 
        
        position_diagonal_dr = diagonal_top

        
        position_right = row * self.num_columns
        position_vertical = column

        ''' range '''
        range_ur = min(self.num_columns-diagonal_column, diagonal_row+1)
        range_dr = min(self.num_columns-inverse_diagonal_column, inverse_diagonal_row+1)
        

       
        range_horizontal = self.num_columns
        range_vertical = self.num_rows

        ''' conditions / counters '''
        win_condition_dr = 0
        win_condition_ur = 0
        win_condition_r = 0
        win_condition_down = 0

        for i in range(max(range_ur, range_dr, range_horizontal, range_vertical)):
            if range_ur >= i+1 and range_ur >= self.connect:
                if self.board[position_diagonal_ur] == mark:
                    win_condition_ur +=1
                else:
                    win_condition_ur = 0
                if win_condition_ur == self.connect:
                    reward[mark - 1] = 1 
                    reward[mark % 2 + 1 - 1] = -1
                  
                    done = True
                    return done, reward
                
                position_diagonal_ur = position_diagonal_ur - 1 * self.num_columns + 1
                
            if range_dr >= i+1 and range_dr >= self.connect:
                if self.board[position_diagonal_dr] == mark:
                    win_condition_dr +=1
                    
                else:
                    win_condition_dr = 0
                if win_condition_dr == self.connect:
                    reward[mark - 1] = 1 
                    reward[mark % 2 + 1 - 1] = -1 
                    
                    done = True
                    return done, reward
                
                
                position_diagonal_dr = position_diagonal_dr + 1 * self.num_columns + 1
                
                
            if range_horizontal >= i+1 and range_horizontal >= self.connect:
                if self.board[position_right] == mark:
                    win_condition_r +=1
                else:
                    win_condition_r = 0
                if win_condition_r == self.connect:
                    reward[mark - 1] = 1 
                    reward[mark % 2 + 1 - 1] = -1 
                    
                    done = True
                    return done, reward
                position_right = position_right + 1
                
            if range_vertical >= i+1 and range_vertical >= self.connect:
                if self.board[position_vertical] == mark:
                    win_condition_down +=1
                else:
                    win_condition_down = 0
                if win_condition_down == self.connect:
                    reward[mark - 1] = 1 
                    reward[mark % 2 + 1 - 1] = -1 
                    
                    done = True
                    return done, reward
                position_vertical = position_vertical + 1  * self.num_columns
                
                
        return done, reward

    def step(self, action, mark):
        done = False
        valid = True
        if action < self.num_columns + 1:
            if self.board[action] == 0:
                k = 0
                while self.board[action + self.num_columns * k] == 0:
                        k += 1
                        if k == self.num_rows:
                            break
                self.board[action + self.num_columns * (k-1)] = mark
                done, reward = self.check(action + self.num_columns* (k-1))
                observations = np.array(self.board)
                
                return observations, valid, done, reward
            else:
                print("action is full")
                valid = False
                observations = np.array(self.board)
                self.render()
                print(action)
                return observations, valid, done, reward
        else:
            print("action : ", action, end = '')
            print("is out of bounds")
            valid = False
            observations = np.array(self.board)
            return observations, valid, done, reward
    

    def render(self):
        for k in range(self.num_columns * 4 +1):
                if k%4 == 0:
                    print('+', end = '')
                else:
                    print('-', end = '')
        print('\n', end = '')
        for i in range(self.num_rows):
            for j in range(self.num_columns):
                print('|', ' ', sep = '', end = '')
                print(self.board[i*self.num_columns + j] ,' ', end = '', sep= '')
            print('| \n', end = '')
            for k in range(self.num_columns * 4 +1):
                if k%4 == 0:
                    print('+', end = '')
                else:
                    print('-', end = '')
            print('\n', end = '')
            
    def reset(self):
        self.board = np.zeros(self.num_columns* self.num_rows, dtype = int)
        self.marks = [1, 2]
        self.done = False

        coinflip = np.random.random()
        
        if coinflip < 0.5:
            trainee_mark = 1

        else:
            trainee_mark = 2
    

        observations = np.array(self.board)

        return trainee_mark, observations
    
    def copy_board(self, board):
        self.board = np.array(board)
