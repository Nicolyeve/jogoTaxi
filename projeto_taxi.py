import gymnasium as gym  
import numpy as np  
import matplotlib.pyplot as plt  
import pickle  

def run(episodes, is_training=True, render=False):
    env = gym.make('Taxi-v3', render_mode='human' if render else None)
    
    
    q = np.zeros((env.observation_space.n, env.action_space.n)) if is_training else pickle.load(open('taxi.pkl', 'rb'))
    
    learning_rate = 0.9  
    discount_factor = 0.9  
    epsilon = 1.0  
    epsilon_decay = 0.0001  
    rng = np.random.default_rng()  

    rewards_per_episode = []  

    for episode in range(episodes):
        state, _ = env.reset() 
        terminated, truncated = False, False  
        total_rewards = 0 

        while not (terminated or truncated):
            action = env.action_space.sample() if is_training and rng.random() < epsilon else np.argmax(q[state])
            
            new_state, reward, terminated, truncated, _ = env.step(action)
            total_rewards += reward  # Soma a recompensa ao total do episódio
            
            if is_training:
                q[state, action] += learning_rate * (reward + discount_factor * np.max(q[new_state]) - q[state, action])
            
            state = new_state
        
        epsilon = max(epsilon - epsilon_decay, 0)
        
        if epsilon == 0:
            learning_rate = 0.0001

        rewards_per_episode.append(total_rewards)  

    env.close()  

    smoothed_rewards = [np.mean(rewards_per_episode[max(0, i-100):i+1]) for i in range(episodes)]
    
    plt.plot(smoothed_rewards)
    plt.savefig('taxi.png')  

    if is_training:
        pickle.dump(q, open('taxi.pkl', 'wb'))

if __name__ == '__main__':
    run(15000)
    
    # Testa o agente treinado com 10 episódios e renderiza o ambiente
    run(10, is_training=False, render=True)