import time

import numpy as np
from sklearn.metrics import mean_squared_error
from tqdm import trange

from experiment_utilities import pickle_save
from robust_smc.data import ReversibleReaction
from robust_smc.nonlinearmhe import NonlinearMhe
from robust_smc.robustnonlinearmhe import RobustifiedNonlinearMhe
from robust_smc.ukf import UKF

# Experiment Settings
SIMULATOR_SEED = 1992
RNG_SEED = 24
NUM_RUNS = 40
NUM_LATENT = 2
NUM_SAMPLES = 1000
NOISE_STD = 0.1
FINAL_TIME = 10
TIME_STEP = 0.1

# RNG
RNG = np.random.RandomState(RNG_SEED)


def experiment_step(simulator):
    Y = simulator.renoise()
    seed = RNG.randint(0, 1000000)

    transition_cov = np.diag(simulator.process_std ** 2)
    observation_cov = simulator.observation_std ** 2

    # BPF sampler
    prior_std = np.array([1, 1])

    # UKF
    ukf = UKF(
        data=Y,
        transition_matrix=simulator.transition_matrix,
        transition_cov=transition_cov,
        observation_cov=observation_cov,
        m_0=np.array([0.1, 4.5]),
        P_0=np.diag(prior_std) ** 2
    )
    a = time.time()
    ukf.filter()
    print('UKF_time:', time.time() - a)

    # MHE
    mhe = NonlinearMhe(
        data=Y,
        transition_matrix=simulator.transition_matrix,
        transition_cov=transition_cov,
        observation_cov=observation_cov,
        m_0=np.array([0.1, 4.5]),
        P_0=np.diag(prior_std) ** 2
    )
    a = time.time()
    mhe.filter()
    print('MHE_time:', time.time() - a)

    # beta-MHE
    robust_mhes = []
    for b in BETA:
        robust_mhe = RobustifiedNonlinearMhe(
            data=Y,
            beta=b,
            transition_matrix=simulator.transition_matrix,
            transition_cov=transition_cov,
            observation_cov=observation_cov,
            m_0=np.array([0.1, 4.5]),
            P_0=np.diag(prior_std) ** 2
        )
        a = time.time()
        robust_mhe.filter()
        print('Robust_MHE_time:', time.time() - a)
        robust_mhes.append(robust_mhe)

    return simulator, ukf, mhe, robust_mhes


def compute_mse_and_coverage(simulator, sampler):
    if isinstance(sampler, UKF):
        filter_means = np.stack(sampler.filter_means)[:, :, 0]
        filter_vars = np.diagonal(np.stack(sampler.filter_covs), axis1=1, axis2=2)

        scores = []
        for var in range(NUM_LATENT):
            mean = filter_means[:, var]
            std = np.sqrt(filter_vars[:, var])

            mse = mean_squared_error(simulator.X[:, var], mean)
            upper = simulator.X[:, var] <= mean + 1.64 * std
            lower = simulator.X[:, var] >= mean - 1.64 * std
            coverage = np.sum(upper * lower) / simulator.X.shape[0]
            scores.append([mse, coverage])
    elif isinstance(sampler, NonlinearMhe):
        filter_means = np.stack(sampler.filter_means)[:, :, 0]
        filter_vars = np.diagonal(np.stack(sampler.filter_covs), axis1=1, axis2=2)

        scores = []
        for var in range(NUM_LATENT):
            mean = filter_means[:, var]
            std = np.sqrt(filter_vars[:, var])

            mse = mean_squared_error(simulator.X[:, var], mean)
            upper = simulator.X[:, var] <= mean + 1.64 * std
            lower = simulator.X[:, var] >= mean - 1.64 * std
            coverage = np.sum(upper * lower) / simulator.X.shape[0]
            scores.append([mse, coverage])
    elif isinstance(sampler, RobustifiedNonlinearMhe):
        filter_means = np.stack(sampler.filter_means)[:, :, 0]
        filter_vars = np.diagonal(np.stack(sampler.filter_covs), axis1=1, axis2=2)

        scores = []
        for var in range(NUM_LATENT):
            mean = filter_means[:, var]
            std = np.sqrt(filter_vars[:, var])

            mse = mean_squared_error(simulator.X[:, var], mean)
            upper = simulator.X[:, var] <= mean + 1.64 * std
            lower = simulator.X[:, var] >= mean - 1.64 * std
            coverage = np.sum(upper * lower) / simulator.X.shape[0]
            scores.append([mse, coverage])
    return scores


def run(runs, contamination):
    process_std = None

    simulator = ReversibleReaction(
        final_time=FINAL_TIME,
        time_step=TIME_STEP,
        observation_std=NOISE_STD,
        process_std=process_std,
        contamination_probability=contamination,
        seed=SIMULATOR_SEED
    )
    ukf_data, mhe_data, robust_mhe_data = [], [], []
    for _ in trange(runs):
        simulator, ukf, mhe, robust_mhes = experiment_step(simulator)
        ukf_data.append(compute_mse_and_coverage(simulator, ukf))
        mhe_data.append(compute_mse_and_coverage(simulator, mhe))
        robust_mhe_data.append([compute_mse_and_coverage(simulator, robust_mhe) for robust_mhe in robust_mhes])
    return np.array(ukf_data), np.array(mhe_data), np.array(robust_mhe_data)


def run2(runs, contamination, simulator=None):
    process_std = None
    simulator = ReversibleReaction(
        final_time=FINAL_TIME,
        time_step=TIME_STEP,
        observation_std=NOISE_STD,
        process_std=process_std,
        contamination_probability=contamination,
        seed=SIMULATOR_SEED
    )
    robust_mhe_error_list, ukf_error_list, mhe_error_list = [], [], []
    for _ in trange(runs):
        simulator, ukf, mhe, robust_mhes = experiment_step(simulator)
        ukf_error = simulator.X - np.squeeze(np.array(ukf.filter_means), axis=2)
        mhe_error = simulator.X - np.squeeze(np.array(mhe.filter_means), axis=2)
        robust_mhes_error = simulator.X - [np.squeeze(np.array(robust_mhe.filter_means), axis=2) for robust_mhe in
                                           robust_mhes]
        ukf_error_list.append(ukf_error)
        robust_mhe_error_list.append(robust_mhes_error)
        mhe_error_list.append(mhe_error)
    return np.array(ukf_error_list), np.array(mhe_error_list), np.array(robust_mhe_error_list)


if __name__ == '__main__':
    mode = 2
    if mode == 1:
        CONTAMINATION = [0, 0.05, 0.1, 0.15, 0.2]
        BETA = [1e-4, 2e-4]
        for contamination in CONTAMINATION:
            print('CONTAMINATION=', contamination)
            results = run(NUM_RUNS, contamination)
            pickle_save(
                f'../results/reversible_reaction/error_{contamination}.pk',
                results)
    elif mode == 2:
        contamination = 0.2
        BETA = [1e-4]
        results = run2(NUM_RUNS, contamination)
        pickle_save(
            f'../results/reversible_reaction/original_{contamination}.pk',
            results)