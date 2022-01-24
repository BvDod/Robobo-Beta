import matplotlib.pyplot as plt
import seaborn as sns
import csv
import numpy as np

def main():

    sns.set()
    
    def moving_average(X, n=100):
        new_list = []
        for i in range(len(X)):
            lower_bound = i - n
            if lower_bound < 0:
                lower_bound = 0
            new_list.append(np.mean(X[lower_bound:i]))
        return new_list
    
    ######## plot Q-table TRAINING

    rewards = np.loadtxt("rewards.txt")
    steps =  np.loadtxt("steps.txt")

    plt.plot(steps, moving_average(rewards, n=30))
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.title("Q-table Training Reward Curve (smoothed)")
    plt.show()


    ######## plot DQN TRAINING
    steps = []
    rewards = []
    with open('5,5.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if row[1] == "Step":
                continue
            steps.append(int(row[1]))
            rewards.append(float(row[2]))
    steps= np.array(steps[1:20000])
    rewards = np.array(rewards[1:20000])
    rewards = rewards[steps <= 20000]
    steps = steps[steps <= 20000]
    yhat = moving_average(rewards, n=30)

    plt.plot(steps, yhat)
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.title("DQN Training Reward Curve (smoothed)")
    plt.show()

    steps = []
    rewards = []
    with open('5.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if row[1] == "Step":
                continue
            steps.append(int(row[1]))
            rewards.append(float(row[2]))
    steps= np.array(steps[1:20000])
    rewards = np.array(rewards[1:20000])
    rewards = rewards[steps <= 20000]
    steps = steps[steps <= 20000]
    yhat = moving_average(rewards, n=30)

    plt.plot(steps, yhat)
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.title("DQN-simple Training Reward Curve (smoothed)")
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



def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    import numpy as np
    from math import factorial
    
  
    window_size = np.abs(np.int(window_size))
    order = np.abs(np.int(order))
    
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')


if __name__ == "__main__":
    main()



