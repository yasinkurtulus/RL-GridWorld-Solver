"""
Grid World Environment
Custom Environment for the GridWorld Problem as in the book Deep Reinforcement Learning.
"""

class GridWorldEnv():
    """Grid-World Environment Class"""
    
    def __init__(self, gridsize=7, startState='00', terminalStates=['64'], ditches=['52'], 
                 ditchPenalty=-10, turnPenalty=-1, winReward=100, mode='prod'):
        
        self.mode = mode
        self.gridsize = min(gridsize, 9)
        self.create_statespace()
        self.actionspace = [0, 1, 2, 3]
        self.actionDict = {0: 'UP', 1: 'DOWN', 2: 'LEFT', 3: 'RIGHT'}
        self.startState = startState
        self.terminalStates = terminalStates
        self.ditches = ditches
        self.winReward = winReward
        self.ditchPenalty = ditchPenalty
        self.turnPenalty = turnPenalty
        self.stateCount = self.get_statespace_len()
        self.actionCount = self.get_actionspace_len()
        self.stateDict = {k: v for k, v in zip(self.statespace, range(self.stateCount))}
        self.currentState = self.startState
        self.isGameEnd = False
        self.totalTurns = 0
        
        if self.mode == 'debug':
            print("State Space:", self.statespace)
            print("State Dict:", self.stateDict)
            print("Action Space:", self.actionspace)
            print("Action Dict:", self.actionDict)
            print("Start State:", self.startState)
            print("Terminal States:", self.terminalStates)
            print("Ditches:", self.ditches)
            print("WinReward:{}, TurnPenalty:{}, DitchPenalty:{}".format(
                self.winReward, self.turnPenalty, self.ditchPenalty))

    def create_statespace(self):
        """gridsize parametresine göre tüm geçerli koordinatları oluşturur."""
        self.statespace = []
        for row in range(self.gridsize):
            for col in range(self.gridsize):
                self.statespace.append(str(row) + str(col))

    def set_mode(self, mode):
        self.mode = mode

    def get_statespace(self):
        return self.statespace

    def get_actionspace(self):
        return self.actionspace

    def get_actiondict(self):
        return self.actionDict

    def get_statespace_len(self):
        return len(self.statespace)

    def get_actionspace_len(self):
        return len(self.actionspace)

    def next_state(self, current_state, action):
        """Oyun kurallarına göre bir sonraki durumu hesaplar."""
        s_row = int(current_state[0])
        s_col = int(current_state[1])
        next_row = s_row
        next_col = s_col
        
        if action == 0:
            next_row = max(0, s_row - 1)
        if action == 1:
            next_row = min(self.gridsize - 1, s_row + 1)
        if action == 2:
            next_col = max(0, s_col - 1)
        if action == 3:
            next_col = min(self.gridsize - 1, s_col + 1)
            
        new_state = str(next_row) + str(next_col)
        
        if new_state in self.statespace:
            if new_state in self.terminalStates:
                self.isGameEnd = True
            if self.mode == 'debug':
                print("CurrentState:{}, Action:{}, NextState:{}".format(current_state, action, new_state))
            return new_state
        else:
            return current_state

    def compute_reward(self, state):
        """Çukurlara ve hedef noktasına göre anlık ödülü hesaplar."""
        reward = 0
        reward += self.turnPenalty
        if state in self.ditches:
            reward += self.ditchPenalty
        if state in self.terminalStates:
            reward += self.winReward
        return reward

    def reset(self):
        """Simülasyonu sıfırlar ve başlangıç durumunu döndürür."""
        self.isGameEnd = False
        self.totalAccumulatedReward = 0
        self.totalTurns = 0
        self.currentState = self.startState
        return self.currentState

    def step(self, action):
        """Seçilen eylemi uygular; yeni durumu ve ödülü döndürür."""
        if self.isGameEnd:
            raise Exception('Game is Over Exception')
        if action not in self.actionspace:
            raise Exception('Invalid Action Exception')
            
        self.currentState = self.next_state(self.currentState, action)
        obs = self.currentState
        reward = self.compute_reward(obs)
        done = self.isGameEnd
        self.totalTurns += 1
        
        if self.mode == 'debug':
            print("Obs:{}, Reward:{}, Done:{}, TotalTurns:{}".format(obs, reward, done, self.totalTurns))
            
        return obs, reward, done, self.totalTurns


if __name__ == '__main__':
    """Kodu test etmek için ana fonksiyon bloğu."""
    env = GridWorldEnv(mode='debug')
    
    print("\nReseting Env...")
    env.reset()
    
    print("\nGo DOWN...")
    env.step(1)
    
    print("\nGo RIGHT...")
    env.step(3)
    
    print("\nGo LEFT...")
    env.step(2)
    
    print("\nGo UP...")
    env.step(0)
    
    # print("\nInvalid ACTION...")
    # env.step(4)