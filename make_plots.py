import matplotlib.pyplot as plt
import seaborn as sns
import csv
import numpy as np

def main():

    sns.set(font_scale=1.3)
    
    def moving_average(X, n=100):
        new_list = []
        for i in range(len(X)):
            lower_bound = i - n
            if lower_bound < 0:
                lower_bound = 0
            new_list.append(np.mean(X[lower_bound:i]))
        return new_list
    
    ######## plot Q-table TRAINING
    dir = "training_results/Task2/5+1"
    rewards = np.loadtxt(f"{dir}/total_rewards.txt")
    steps =  np.loadtxt(f"{dir}/total_rewards_steps.txt")
    steps_per =  np.loadtxt(f"{dir}/steps_per_iteration.txt")

    plt.plot(steps, moving_average(rewards, n=1))
    plt.xlabel("Step")
    plt.ylabel("Food Gathered per Episode")
    plt.title("Food Gathered per Episode (Qtable-5+1)")
    plt.subplots_adjust(bottom=0.15)
    plt.show()

    plt.plot(steps, moving_average(steps_per, n=10))
    plt.xlabel("Step")
    plt.ylabel("Epsiode Length (steps)")
    plt.title("Time to gather all Food (Qtable-5+1)")
    plt.subplots_adjust(bottom=0.15)
    plt.show()

    ######## plot Epsilon TRAINING
    start_epsilon = 0.5
    end_epsilon = 0.05
    end_epsilon_iteration = 20000
    epsilons = []
    for i in range(100000):
        if i < end_epsilon_iteration:
            epsilon = start_epsilon - ((start_epsilon - end_epsilon) * (i/end_epsilon_iteration))
        else:
            epsilon = end_epsilon

        epsilons.append(epsilon)
    plt.plot(range(100000), epsilons)
    plt.xlabel("Step")
    plt.ylabel("Epsilon")
    plt.title("Epsilon decay")
    plt.subplots_adjust(bottom=0.15)
    plt.show()




    ####### Evaluation Boxplots
    dqn1 = [39, 11, 41, 36, 75, 23, 119, 5, 33, 19]
    dqn2 = [172, 9, 15, 15, 48, 70, 30, 64, 31, 53]
    qtable = [552, 587, 1128, 380, 1830, 843, 389, 379, 376, 216]
    random = [[10, 4, 31, 0.0, 8, 35, 8, 16, 4, 32]]

    print(np.mean(dqn1),np.mean(dqn2),np.mean(qtable),np.mean(random))
    ax = sns.boxplot(data=[dqn1, dqn2, qtable, random])
    ax.set_xticklabels(["DQN", "DQN-simple", "Q-table", "Random Baseline"])
    ax.set_ylabel("Foward moves untill Collision")
    ax.set_title("Evaluation of final models")
    plt.show()

    ax = sns.boxplot(data=[qtable])
    ax.set_xticklabels(["DQN", "DQN-simple", "Q-table"])
    ax.set_ylabel("Foward moves untill Collision")
    ax.set_title("Evaluation of final models")
    plt.show()



if __name__ == "__main__":
    main()



