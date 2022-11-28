import matplotlib.pyplot as plt
import pandas as pd

from hiive.mdptoolbox import mdp, example
from gym.envs.toy_text.frozen_lake import generate_random_map

def plot_rewards(results):
    for result in results:
        run_stats_df = pd.DataFrame(result['run_stats'])
        plt.plot(run_stats_df['Iteration'], run_stats_df['Mean V'], label = "Mean Reward")
        plt.plot(run_stats_df['Iteration'], run_stats_df['Max V'], label = "Max Reward")
        plt.xlabel('Iteration')
        plt.ylabel('Reward')
        plt.title(result['exp_name'])
        plt.legend()
        plt.show()

def plot_time(results):
    for result in results:
        run_stats_df = pd.DataFrame(result['run_stats'])
        plt.plot(run_stats_df['Iteration'], run_stats_df['Time'], label = "time")
        plt.xlabel('Iteration')
        plt.ylabel('time')
        plt.title(result['exp_name'])
        plt.legend()
        plt.show()

def run_forest(solver, states, discounts, solver_name, probability=0.1, max_iter=5000):
    experiments = [] #num states, probability, discount, time, iterations, policy
    for d in discounts:
        for s in states:
            current_name = f"{solver_name}_iter_{s}_state_{d}_discount"
            record = {}
            P, R = example.forest(S=s, p=probability)
            args = {"transitions":P, "reward":R, "gamma":d, "max_iter":max_iter}
            mdp = solver(args)
            mdp.run()
            record['exp_name'] = current_name
            record['policy'] = mdp.policy
            record['num_states'] = s
            record['discount'] = d
            record['iterations'] = mdp.iter
            record['run_stats'] = mdp.run_stats
            experiments.append(record)
    plot_rewards(experiments)
    plot_time(experiments)
    return experiments

def visualize_direction(policy, size):
    pol = list(policy)
    sublists = [pol[x:x+size] for x in range(0, len(pol), size)]
    for lst in sublists:
        for i in range(len(lst)):
            if lst[i] == 0:
                lst[i] = '←'
            elif lst[i] == 1:
                lst[i] = '↓'
            elif lst[i] == 2:
                lst[i] = '→'  
            elif lst[i] == 3:
                lst[i] = '↑'
        print(lst)
    return sublists


def run_lake(solver, sizes, discounts, solver_name, max_iter=500):
    experiments = [] #num states, probability, discount, time, iterations, policy
    for d in discounts:
        for s in sizes:
            print(d, s)
            current_name = f"{solver_name}_iter_{s}_state_{d}_discount"
            record = {}
            random_map = generate_random_map(size=s)
            P, R = example.openai("FrozenLake-v1", desc=random_map, is_slippery=True)
            args = {"transitions":P, "reward":R, "gamma":d, "max_iter":max_iter}
            mdp = solver(args)
            mdp.run()
            record['exp_name'] = current_name
            record['policy'] = mdp.policy
            record['size_of_grid'] = s
            record['discount'] = d
            record['iterations'] = mdp.iter
            record['run_stats'] = mdp.run_stats
            experiments.append(record)
    plot_rewards(experiments)
    plot_time(experiments)
    return experiments


def compare_policy(results_1, results_2, term):
    diff = {}
    for i in range(len(results_1)):
        assert results_1[i][term] == results_2[i][term]
        assert results_1[i]['discount'] == results_2[i]['discount']
        
        diff[f"{results_1[i][term]}_{results_1[i]['discount']}"] = results_2[i]['policy'] == results_1[i]['policy']
    return diff


def q_learning_forest(solver, states, discounts, n_iters, solver_name, probability=0.1):
    experiments = [] #num states, probability, discount, time, iterations, policy
    for d in discounts:
        for s in states:
            for n_iter in n_iters:
                current_name = f"{solver_name}_{n_iter}_iter_{s}_state_{d}_discount"
                record = {}
                P, R = example.forest(S=s, p=probability)
                args = {"transitions":P, "reward":R, "gamma":d, "n_iter":n_iter}
                mdp = solver(args)
                mdp.run()
                record['exp_name'] = current_name
                record['policy'] = mdp.policy
                record['num_states'] = s
                record['discount'] = d
                record['iterations'] = n_iter
                record['run_stats'] = mdp.run_stats
                experiments.append(record)
    plot_rewards(experiments)
    plot_time(experiments)
    return experiments


def q_learning_lake(solver, sizes, discounts, n_iterssolver_name, , max_iter=500):
    experiments = [] #num states, probability, discount, time, iterations, policy
    for d in discounts:
        for s in sizes:
            for n_iter in n_iters:
                current_name = f"{solver_name}_{n_iter}_iter_{s}_state_{d}_discount"
                record = {}
                random_map = generate_random_map(size=s)
                P, R = example.openai("FrozenLake-v1", desc=random_map, is_slippery=True)
                args = {"transitions":P, "reward":R, "gamma":d, "n_iter": n_iter}
                mdp = solver(args)
                mdp.run()
                record['exp_name'] = current_name
                record['policy'] = mdp.policy
                record['size_of_grid'] = s
                record['discount'] = d
                # record['iterations'] = mdp.iter
                record['run_stats'] = mdp.run_stats
                experiments.append(record)
    plot_rewards(experiments)
    plot_time(experiments)
    return experiments