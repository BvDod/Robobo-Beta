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
    qtable3 = [208, 220, 261, 221, 241, 236, 233, 208, 242, 234]
    qtable3_food = [11, 11, 11, 11, 11, 11, 11, 11, 11, 11]

    qtable5 = [261, 228, 215, 189, 193, 202, 183, 215, 209, 182]
    qtable5_food = [11, 11, 11, 11, 11, 11, 11, 11, 11, 11]

    qtable51 = [233, 247, 239, 183, 235, 234, 243, 210, 214, 244]
    qtable51_food =[11, 11, 11, 11, 11, 11, 11, 11, 11, 11]

    qtableRand = [500, 500, 500, 500, 500, 500, 500, 500, 500, 500]
    qtableRand_food = [3, 3, 2, 3, 3, 3, 1, 4, 2, 3]


    print(np.mean(qtable3),np.mean(qtable5),np.mean(qtable51),np.mean(qtableRand))
    print(np.mean(np.array(qtable3)/np.array(qtable3_food)),np.mean(np.array(qtable5)/np.array(qtable5_food)),np.mean(np.array(qtable51)/np.array(qtable51_food)),np.mean(np.array(qtableRand)/np.array(qtableRand_food)))

    ax = sns.boxplot(data=[np.array(qtable3)/np.array(qtable3_food), np.array(qtable5)/np.array(qtable5_food), np.array(qtable51)/np.array(qtable51_food)])
    ax.set_xticklabels(["Q-table-3", "Q-table-5", "Q-table-5+1"])
    ax.set_ylabel("Moves")
    ax.set_title("Evaluation: avg. moves untill food gathered")
    plt.show()

    ax = sns.boxplot(data=[qtable])
    ax.set_xticklabels(["DQN", "DQN-simple", "Q-table"])
    ax.set_ylabel("Foward moves untill Collision")
    ax.set_title("Evaluation of final models")
    plt.show()



if __name__ == "__main__":
    main()



