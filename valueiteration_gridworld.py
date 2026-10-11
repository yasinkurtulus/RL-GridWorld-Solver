"""
Value Iteration Algorithm
Code to demonstrate the Value Iteration method for solving Grid World
"""

import numpy as np
from envs.grid_world import GridWorldEnv  # Bir önceki adımda yazdığımız ortam

class ValueIteration:
    """Değer İterasyonu Algoritması Sınıfı"""
    
    def __init__(self, env=GridWorldEnv(), discountingFactor=0.9, convergenceThreshold=1e-4, iterationThreshold=1000, mode='prod'):
        """
        ValueIteration Sınıfını Başlatır.
        
        Parametreler:
        - env: GridWorldEnv ortamının bir örneği
        - discountingFactor (gamma): Gelecekteki ödüllerin bugünkü değerini düşüren iskonto faktörü
        - convergenceThreshold (theta): Değerlerin sabitlendiğini (yakınsama) anlamak için eşik değeri
        - iterationThreshold: Yakınsama kontrolü için maksimum döngü sınırı
        - mode: Çalışma modu ('prod' veya 'debug')
        """
        self.env = env
        self.gamma = discountingFactor
        self.th = convergenceThreshold
        self.maxIter = iterationThreshold
        self.mode = mode
        
        self.stateCount = self.env.get_statespace_len()
        self.actionCount = self.env.get_actionspace_len()
        self.uniformActionProbability = 1.0 / self.actionCount
        self.stateDict = self.env.stateDict
        self.actionDict = self.env.actionDict
        
        # Değer Fonksiyonu (V), Q-Tablosu (Q) ve Politika (Policy) dizilerini sıfırlarla başlat
        self.V = np.zeros(self.stateCount)
        self.Q = [np.zeros(self.actionCount) for s in range(self.stateCount)]
        self.Policy = np.zeros(self.stateCount, dtype=int)
        
        self.totalReward = 0
        self.totalSteps = 0

    def reset_episode(self):
        """Bölüm (episode) sayaçlarını sıfırlar."""
        self.totalReward = 0
        self.totalSteps = 0

    def iterate_value(self):
        """
        Bellman Optimizasyon Denklemini kullanarak değerleri (V) iteratif olarak günceller 
        ve yakınsamayı (convergence) kontrol eder.
        """
        self.V = np.zeros(self.stateCount)
        
        for i in range(self.maxIter):
            # V dizisinin bir önceki halini kopyala
            last_V = np.copy(self.V)
            
            for state_index in range(self.stateCount):
                current_state = self.env.statespace[state_index]
                
                # Her bir olası hamle için Q değerini hesapla
                for action in self.env.actionspace:
                    next_state = self.env.next_state(current_state, action)
                    reward = self.env.compute_reward(next_state)
                    next_state_index = self.env.stateDict[next_state]
                    
                    # Q(s,a) = R(s') + gamma * V(s')
                    self.Q[state_index][action] = reward + self.gamma * last_V[next_state_index]
                
                if self.mode == 'debug':
                    print("Q(s={}):{}".format(current_state, self.Q[state_index]))
                
                # V(s) = max_a Q(s,a) (En yüksek Q değerini o karenin V değeri yap)
                self.V[state_index] = max(self.Q[state_index])
                
            # Yakınsama Kontrolü: Eski V ile yeni V arasındaki fark eşik değerinden küçük mü?
            if np.sum(np.fabs(last_V - self.V)) <= self.th:
                print(f"Yakınsama {i}. iterasyonda başarıldı. Döngü kırılıyor (Breaking V_Iteration loop)!")
                break

    def extract_optimal_policy(self):
        """
        Hesaplanan Q değerlerine bakarak her kare için en iyi eylemi (Policy) seçer.
        """
        # Her durum için en yüksek Q değerine sahip olan eylemin indeksini (argmax) politikaya kaydet
        self.Policy = np.argmax(self.Q, axis=1)
        if self.mode == 'debug':
            print("Optimal Policy:", self.Policy)

    def run_episode(self):
        """
        Çıkarılan optimum politikayı kullanarak oyunu bir el (episode) oynar.
        """
        self.reset_episode()
        obs = self.env.reset()
        
        while True:
            # Mevcut duruma göre politikadan eylemi seç
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
        """
        Bulunan politikanın başarısını ölçmek için oyunu n_episodes kez oynatır 
        ve ortalama skoru döndürür.
        """
        episode_scores = []
        if self.mode == 'debug':
            print(f"{n_episodes} bölüm çalıştırılıyor!")
            
        for e in range(n_episodes):
            score = self.run_episode()
            episode_scores.append(score)
            if self.mode == 'debug':
                print(f"Bölüm {e} Skoru = {score}")
                
        return np.mean(episode_scores)

    def solve_mdp(self, n_episode=100):
        """
        Tüm MDP (Markov Karar Süreci) çözüm sürecini sırasıyla yönetir.
        """
        if self.mode == 'debug':
            print("Değerler İtere Ediliyor (Iterating Values)...")
        self.iterate_value()
        
        if self.mode == 'debug':
            print("Optimum Politika Çıkarılıyor (Extracting Optimal Policy)...")
        self.extract_optimal_policy()
        
        if self.mode == 'debug':
            print("Politika Puanlanıyor (Scoring Policy)...")
        return self.evaluate_policy(n_episode)


if __name__ == '__main__':
    """Kodu test etmek için ana fonksiyon bloğu."""
    print("Değişkenler başlatılıyor ve ortam kuruluyor...")
    
    # ValueIteration sınıfını çağır
    valueIteration = ValueIteration(env=GridWorldEnv(mode='prod'), mode='prod')
    
    # Süreci başlat ve sonucu yazdır
    print('Politika Değerlendirme Ortalaması (Policy Evaluation Score) = ', valueIteration.solve_mdp())