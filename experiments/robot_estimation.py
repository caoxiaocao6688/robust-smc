import time

import numpy as np
import random
from robust_smc.nonlinearmhe_robot import NonlinearMheRobot
from robust_smc.ukf_robot import UKFRobot
from robust_smc.robustnonlinearmhe_robot import RobustifiedNonlinearMheRobot
from data_processing.data_processing import load_data
from tqdm import trange

from sklearn.metrics import mean_squared_error
from experiment_utilities import pickle_save

import matplotlib.pyplot as plt
import matplotlib as mpl
# Experiment Settings
SIMULATOR_SEED = 1992
RNG_SEED = 24
# Sampler Settings
NUM_LATENT = 3
NUM_SAMPLES = 1000


class Robot:
    def __init__(self, contamination, seed, filepath_list, min_len):
        self.process_std = np.array([0.0034, 0.0056, 0.0041])
        self.observation_std = np.array([0.0238, 0.0284, 0.0259, 0.0107, 0.0094, 0.0118])
        # self.observation_std = np.array([0.0238,  0.0107])
        self.transition_matrix = None
        self.filepath_list = filepath_list
        self.min_len = min_len
        self.seed = seed
        self.contamination = contamination
        state = []
        action = []
        obs = []
        for file_path in self.filepath_list:
            state_, action_, obs_ = load_data(file_path, min_len)
            state += state_
            action += action_
            obs += obs_
        self.NUM_RUNS = len(state)
        # self.NUM_RUNS = 1 # TODO
        self.X_list = []
        self.Y_list = []
        self.U_list = []
        for i in range(self.NUM_RUNS):
            self.X_list.append(np.array(state[i]))
            # self.Y_list.append(np.array(obs[i])[:, [0, 3]])
            self.Y_list.append(np.array(obs[i]))
            self.U_list.append(np.array(action[i]))
        X_arr = np.array(self.X_list)
        self.zero_states = X_arr[:, 0, :]
        self.m_0 = np.mean(X_arr[:, 0, :], axis=0)
        self.rng = np.random.RandomState(self.seed)

    def noise_model(self, Y):
        p_u = self.rng.rand(Y.shape[0])
        noise = np.zeros_like(Y)
        loc = (p_u <= self.contamination)
        noise[loc] = 10 * self.rng.rand(loc.sum(), 1)
        return self.observation_std * noise

    # def renoise(self, run):
    #     return self.Y_list[run] + self.noise_model(self.Y_list[run])  # TODO

    def get_y(self, run):
        return self.Y_list[run]

    def renoise_(self, run):
        Y = self.Y_list[run]
        p_u = self.rng.rand(Y.shape[0])
        loc = (p_u <= self.contamination)
        random_number = random.randint(0, 2)
        Y[loc, random_number] = 20
        return Y


def experiment_step(simulator, run):
    Y = simulator.renoise_(run)
    transition_cov = np.diag(simulator.process_std ** 2)
    observation_cov = np.diag(simulator.observation_std ** 2)
    prior_std = np.array([0.0001, 0.0001, 0.0001])

    # UKF
    ukf = UKFRobot(
        data=Y,
        transition_matrix=simulator.transition_matrix,
        transition_cov=transition_cov,
        observation_cov=observation_cov,
        m_0=simulator.zero_states[run, :],
        P_0=np.diag(prior_std) ** 2,
        U=simulator.U_list[run],
        X=simulator.X_list[run]
    )
    # ukf.compare_state_sequences()
    # ukf.compare_observations()
    time_a = time.time()
    ukf.filter()
    print('UKF_time', time.time() - time_a)

    # MHE
    mhe = NonlinearMheRobot(
        data=Y,
        transition_matrix=simulator.transition_matrix,
        transition_cov=transition_cov,
        observation_cov=observation_cov,
        m_0=simulator.zero_states[run, :],
        P_0=np.diag(prior_std) ** 2,
        U=simulator.U_list[run]
    )
    time_a = time.time()
    mhe.filter()
    print('MHE_time', time.time() - time_a)

    # beta-MHE
    robust_mhes = []
    for b in BETA:
        robust_mhe = RobustifiedNonlinearMheRobot(
            data=Y,
            beta=b,
            transition_matrix=simulator.transition_matrix,
            transition_cov=transition_cov,
            observation_cov=observation_cov,
            m_0=simulator.zero_states[run, :],
            P_0=np.diag(prior_std) ** 2,
            U=simulator.U_list[run]
        )
        time_a = time.time()
        robust_mhe.filter()
        print('Robust_MHE_time', time.time() - time_a)
        robust_mhes.append(robust_mhe)

    return simulator, ukf, mhe, robust_mhes


def compute_mse_and_coverage(simulator, sampler, run):
    if isinstance(sampler, UKFRobot):
        filter_means = np.stack(sampler.filter_means)[:, :, 0]
        filter_vars = np.diagonal(np.stack(sampler.filter_covs), axis1=1, axis2=2)

        scores = []
        for var in range(NUM_LATENT):
            mean = filter_means[:, var]
            std = np.sqrt(filter_vars[:, var])

            mse = mean_squared_error(simulator.X_list[run][:, var], mean)
            upper = simulator.X_list[run][:, var] <= mean + 1.64 * std
            lower = simulator.X_list[run][:, var] >= mean - 1.64 * std
            coverage = np.sum(upper * lower) / simulator.X_list[run].shape[0]
            scores.append([mse, coverage])
    elif isinstance(sampler, NonlinearMheRobot):
        filter_means = np.stack(sampler.filter_means)[:, :, 0]
        filter_vars = np.diagonal(np.stack(sampler.filter_covs), axis1=1, axis2=2)

        scores = []
        for var in range(NUM_LATENT):
            mean = filter_means[:, var]
            std = np.sqrt(filter_vars[:, var])

            mse = mean_squared_error(simulator.X_list[run][:, var], mean)
            upper = simulator.X_list[run][:, var] <= mean + 1.64 * std
            lower = simulator.X_list[run][:, var] >= mean - 1.64 * std
            coverage = np.sum(upper * lower) / simulator.X_list[run].shape[0]
            scores.append([mse, coverage])
    elif isinstance(sampler, RobustifiedNonlinearMheRobot):
        filter_means = np.stack(sampler.filter_means)[:, :, 0]
        filter_vars = np.diagonal(np.stack(sampler.filter_covs), axis1=1, axis2=2)

        scores = []
        for var in range(NUM_LATENT):
            mean = filter_means[:, var]
            std = np.sqrt(filter_vars[:, var])

            mse = mean_squared_error(simulator.X_list[run][:, var], mean)
            upper = simulator.X_list[run][:, var] <= mean + 1.64 * std
            lower = simulator.X_list[run][:, var] >= mean - 1.64 * std
            coverage = np.sum(upper * lower) / simulator.X_list[run].shape[0]
            scores.append([mse, coverage])
    else:
        trajectories = np.stack(sampler.X_trajectories)
        mean = trajectories.mean(axis=1)
        quantiles = np.quantile(trajectories, q=[0.05, 0.95], axis=1)
        scores = []
        for var in range(NUM_LATENT):
            mse = mean_squared_error(simulator.X[:, var], mean[:, var])
            upper = simulator.X_list[run][:, var] <= quantiles[1, :, var]
            lower = simulator.X_list[run][:, var] >= quantiles[0, :, var]
            coverage = np.sum(upper * lower) / simulator.X_list[run].shape[0]
            scores.append([mse, coverage])
    return scores


def run(contamination):
    simulator = Robot(
        contamination=contamination,
        seed=SIMULATOR_SEED,
        filepath_list=['../data_processing/20230216-140452.npz',
                       '../data_processing/20230216-141321.npz', '../data_processing/20230216-141616.npz',
                       '../data_processing/20230216-142042.npz'],
        min_len=101
    )
    ukf_data, mhe_data, robust_mhe_data = [], [], []
    for run in trange(simulator.NUM_RUNS):
        simulator, ukf, mhe, robust_mhes = experiment_step(simulator, run)
        ukf_data.append(compute_mse_and_coverage(simulator, ukf, run))
        mhe_data.append(compute_mse_and_coverage(simulator, mhe, run))
        robust_mhe_data.append([compute_mse_and_coverage(simulator, robust_mhe, run) for robust_mhe in robust_mhes])
    mhe_np, robust_mhe_np = np.array(mhe_data), np.array(robust_mhe_data)
    mhe_np = robust_mhe_np[:, 0, :, :]
    robust_mhe_np = np.delete(robust_mhe_np, 0, axis=1)
    return np.array(ukf_data), mhe_np, robust_mhe_np


def run2(contamination):
    simulator = Robot(
        contamination=contamination,
        seed=SIMULATOR_SEED,
        filepath_list=['../data_processing/20230216-140452.npz',
                       '../data_processing/20230216-141321.npz', '../data_processing/20230216-141616.npz',
                       '../data_processing/20230216-142042.npz'],
        min_len=101
    )
    robust_mhe_error_list, ukf_error_list, mhe_error_list = [], [], []
    for run in trange(simulator.NUM_RUNS):
        simulator, ukf, mhe, robust_mhes = experiment_step(simulator, run)
        ukf_error = simulator.X_list[run] - np.squeeze(np.array(ukf.filter_means), axis=2)
        mhe_error = simulator.X_list[run] - np.squeeze(np.array(mhe.filter_means), axis=2)
        robust_mhes_error = simulator.X_list[run] - [np.squeeze(np.array(robust_mhe.filter_means), axis=2) for
                                                     robust_mhe in
                                                     robust_mhes]
        ukf_error_list.append(ukf_error)
        robust_mhe_error_list.append(robust_mhes_error)
        mhe_error_list.append(mhe_error)
    mhe_np, robust_mhe_np = np.array(mhe_error_list), np.array(robust_mhe_error_list)
    mhe_np = robust_mhe_np[:, 0, :, :]
    robust_mhe_np = np.delete(robust_mhe_np, 0, axis=1)
    return np.array(ukf_error_list), mhe_np, robust_mhe_np


def run3(contamination):
    simulator = Robot(
        contamination=contamination,
        seed=SIMULATOR_SEED,
        filepath_list=['../data_processing/20230216-140452.npz',
                       '../data_processing/20230216-141321.npz', '../data_processing/20230216-141616.npz',
                       '../data_processing/20230216-142042.npz'],
        min_len=101
    )
    robust_mhe_list, ukf_list, mhe_list = [], [], []
    for run in trange(simulator.NUM_RUNS):
        simulator, ukf, mhe, robust_mhes = experiment_step(simulator, run)
        ukf_list.append(np.squeeze(np.array(ukf.filter_means), axis=2))
        mhe_list.append(np.squeeze(np.array(mhe.filter_means), axis=2))
        robust_mhe_list.append([np.squeeze(np.array(robust_mhe.filter_means), axis=2) for
                                robust_mhe in
                                robust_mhes])
    mhe_np, robust_mhe_np = np.array(mhe_list), np.array(robust_mhe_list)
    mhe_np = robust_mhe_np[:, 0, :, :]
    robust_mhe_np = np.delete(robust_mhe_np, 0, axis=1)
    robust_mhe_np = np.squeeze(robust_mhe_np)
    return np.array(simulator.X_list), np.array(ukf_list), mhe_np, robust_mhe_np


if __name__ == '__main__':
    mode = 3
    if mode == 1:
        BETA = [1e-5, 1e-3, 1e-2, 0.05, 0.1, 0.2]
        CONTAMINATION = [0.01]
        for contamination in CONTAMINATION:
            print('CONTAMINATION=', contamination)
            results = run(contamination)
            pickle_save(
                f'../results/robot_estimation/error_{contamination}.pk',
                results)
    elif mode == 2:
        BETA = [1e-5, 0.1]
        CONTAMINATION = [0.01]
        for contamination in CONTAMINATION:
            print('CONTAMINATION=', contamination)
            results = run2(contamination)
            pickle_save(
                f'../results/robot_estimation/original_{contamination}.pk',
                results)
    elif mode == 3:
        BETA = [1e-5, 0.1]
        CONTAMINATION = [0.01]
        for contamination in CONTAMINATION:
            print('CONTAMINATION=', contamination)
            results = run3(contamination)
            pickle_save(
                f'../results/robot_estimation/traj_{contamination}.pk',
                results)
