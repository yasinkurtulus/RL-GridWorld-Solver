"""
Policy Iteration Algorithm
Code to demonstrate the Policy Iteration method for solving Grid World
"""

import numpy as np
from envs.grid_world import GridWorldEnv

class PolicyIteration:
    """Politika İterasyonu Algoritması Sınıfı"""
    
    def __init__(self, env=GridWorldEnv(), discountingFactor=0.9, convergenceThreshold=1e-4, 
                 iterationThresholdValue=1000, iterationThresholdPolicy=100, mode='prod'):
        """
        PolicyIteration Sınıfını Başlatır.
        
        Parametreler:
        - env: GridWorldEnv ortamının bir örneği
        - discountingFactor (gamma): Gelecekteki ödüllerin iskonto faktörü
        - convergenceThreshold (theta): Değerlerin yakınsama eşiği
        - iterationThresholdValue: Değer değerlendirmesi (Value Iteration) için maksimum döngü sınırı
        - iterationThresholdPolicy: Politika güncellemeleri için maksimum döngü sınırı
        - mode: Çalışma modu ('prod' veya 'debug')
        """
        self.env = env
        self.gamma = discountingFactor
        self.th = convergenceThreshold
        self.maxIterValue = iterationThresholdValue
        self.maxIterPolicy = iterationThresholdPolicy
        
        self.stateCount = self.env.get_statespace_len()
        self.actionCount = self.env.get_actionspace_len()
        self.uniformActionProbability = 1.0 / self.actionCount
        
        self.stateDict = self.env.stateDict
        self.actionDict = self.env.actionDict
        self.mode = mode
        
        self.V = np.zeros(self.stateCount)
        self.Q = [np.zeros(self.actionCount) for s in range(self.stateCount)]
        self.Policy = np.zeros(self.stateCount, dtype=int)
        
        self.totalReward = 0
        self.totalSteps = 0

    def reset_episode(self):
        """Bölüm (episode) sayaçlarını sıfırlar."""
        self.totalReward = 0
        self.totalSteps = 0

    def compute_value_under_policy(self):
        """
        Mevcut politikaya göre durumların değerini (V) hesaplar.
        (Politika Değerlendirme Aşaması)
        """
        self.V = np.zeros(self.stateCount)
        
        for i in range(self.maxIterValue):
            last_V = np.copy(self.V)
            
            for state_index in range(self.stateCount):
                current_state = self.env.statespace[state_index]
                
                for action in self.env.actionspace:
                    next_state = self.env.next_state(current_state, action)
                    reward = self.env.compute_reward(next_state)
                    next_state_index = self.env.stateDict[next_state]
                    
                    self.Q[state_index][action] = reward + self.gamma * last_V[next_state_index]
                
                if self.mode == 'debug':
                    print("Q(s={}):{}".format(current_state, self.Q[state_index]))
                
                self.V[state_index] = max(self.Q[state_index])
                
            if np.sum(np.fabs(last_V - self.V)) <= self.th:
                if self.mode == 'debug':
                    print(f"Yakınsama {i}. iterasyonda başarıldı. Değer döngüsü kırılıyor!")
                break

    def iterate_policy(self):
        """
        Politika İterasyonunun kalbi. Rastgele bir kural setiyle başlar,
        değerlendirir ve kural setini kopyalayıp günceller.
        (Politika Geliştirme Aşaması)
        """
        # 1. Adım: Başlangıçta tamamen RASTGELE bir politika (kural seti) uydur
        self.Policy = [np.random.choice(self.actionCount) for s in range(self.stateCount)]
        
        for i in range(self.maxIterPolicy):
            # 2. Adım: Mevcut rastgele/eski kuralların haritadaki değerini hesapla
            self.compute_value_under_policy()
            
            # Eski kuralları bir kenara not et
            old_policy = list(self.Policy)
            
            # 3. Adım: Değerlere bakarak kuralları daha iyi hale getir
            self.improve_policy()
            new_policy = list(self.Policy)
            
            # 4. Adım: Eski kurallarla yeni kurallar tamamen aynıysa algoritmayı durdur
            if np.all(np.array(old_policy) == np.array(new_policy)):
                print(f'Politika Yakınsaması (Policy Convergence) {i}. adımda başarıldı!')
                break
                
            self.Policy = new_policy
            
        return self.Policy

    def improve_policy(self):
        """
        Hesaplanan Q değerlerine bakarak politikayı iyileştirir (Günceller).
        """
        self.Policy = np.argmax(self.Q, axis=1)
        if self.mode == 'debug':
            print("Optimal Policy:", self.Policy)

    def run_episode(self):
        """Güncel politika ile oyunu bir el oynar."""
        self.reset_episode()
        obs = self.env.reset()
        
        while True:
            action = self.Policy[self.env.stateDict[obs]]
            new_obs, reward, done, _ = self.env.step(action)
            
            if self.mode == 'debug':
                print("PrevObs:{}, Action:{}, Obs:{}, Reward:{}, Done:{}".format(
                    obs, action, new_obs, reward, done))
                
            self.totalReward += reward
            self.totalSteps += 1
            
            if done:
                break
            else:
                obs = new_obs
                
        return self.totalReward

    def evaluate_policy(self, n_episodes=100):
        """Politikayı n_episodes kadar oynatıp test eder ve skoru döner."""
        episode_scores = []
        if self.mode == 'debug':
            print(f"{n_episodes} bölüm çalıştırılıyor!")
            
        for e in range(n_episodes):
            score = self.run_episode()
            episode_scores.append(score)
            
            if self.mode == 'debug':
                print("Bölüm {} Skoru = {}".format(e, score))
                
        return np.mean(episode_scores)

    def solve_mdp(self, n_episode=10):
        """Tüm süreci orkestra eden ana yönetici fonksiyon."""
        if self.mode == 'debug':
            print("Değişkenler başlatılıyor ve ortam kuruluyor...")
            
        self.iterate_policy()
        
        if self.mode == 'debug':
            print("Politika Puanlanıyor (Scoring Policy)...")
            
        return self.evaluate_policy(n_episode)


if __name__ == '__main__':
    """Kodu test etmek için ana fonksiyon bloğu."""
    print("Değişkenler başlatılıyor ve ortam kuruluyor...")
    
    # Sınıfı çağır
    policyIteration = PolicyIteration(env=GridWorldEnv(mode='prod'), mode='prod')
    
    # Çözüm sürecini başlat ve sonucu yazdır
    print('Politika Değerlendirme Skoru = ', policyIteration.solve_mdp())