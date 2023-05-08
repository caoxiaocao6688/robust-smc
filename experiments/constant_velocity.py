import numpy as np
from sklearn.metrics import mean_squared_error
from tqdm import trange

from experiment_utilities import pickle_save
from robust_smc.data import ConstantVelocityModel
from robust_smc.kalman import Kalman
from robust_smc.mhe import Mhe
from robust_smc.robustmhe import RobustifiedMhe

# Experiment Settings
SIMULATOR_SEED = 1400
RNG_SEED = 1218
NUM_RUNS = 100

CONTAMINATION = [0.2] # Sampler Settings
NUM_LATENT = 4
NUM_SAMPLES = 100
NOISE_VAR = 1.0
FINAL_TIME = 20
TIME_STEP = 0.1
EXPLOSION_SCALE = 100.0

# RNG
RNG = np.random.RandomState(RNG_SEED)  # 随机数


def experiment_step(simulator):
    Y = simulator.renoise()
    # Kalman
    kalman = Kalman(
        data=Y,
        transition_matrix=simulator.transition_matrix,
        observation_matrix=simulator.observation_matrix,
        transition_cov=simulator.process_cov,
        observation_cov=simulator.observation_cov,
        m_0=np.zeros((NUM_LATENT, 1)),
        P_0=simulator.initial_cov
    )
    # a = time.time()
    kalman.filter()
    # print('KF time', (time.time()-a)/1000)
    # MHE
    mhe = Mhe(
        data=Y,
        transition_matrix=simulator.transition_matrix,
        observation_matrix=simulator.observation_matrix,
        transition_cov=simulator.process_cov,
        observation_cov=simulator.observation_cov,
        m_0=np.zeros((NUM_LATENT, 1)),
        P_0=simulator.initial_cov
    )
    # a = time.time()
    mhe.filter()
    # print('MHE time', (time.time()-a)/1000)

    # beta-MHE
    robust_mhes = []
    for b in BETA:
        robust_mhe = RobustifiedMhe(
            data=Y,
            beta=b,
            transition_matrix=simulator.transition_matrix,
            observation_matrix=simulator.observation_matrix,
            transition_cov=simulator.process_cov,
            observation_cov=simulator.observation_cov,
            m_0=np.zeros((NUM_LATENT, 1)),
            P_0=simulator.initial_cov
        )
        # a = time.time()
        robust_mhe.filter()
        # print('ROBUST MHE time', (time.time() - a) / 1000)
        robust_mhes.append(robust_mhe)
    return simulator, kalman, mhe, robust_mhes


def compute_mse_and_coverage(simulator, sampler):
    if isinstance(sampler, Kalman):
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
    elif isinstance(sampler, Mhe):
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
    elif isinstance(sampler, RobustifiedMhe):
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
    observation_cov = NOISE_VAR * np.eye(2)
    simulator = ConstantVelocityModel(
        final_time=FINAL_TIME,
        time_step=TIME_STEP,
        observation_cov=observation_cov,
        explosion_scale=EXPLOSION_SCALE,
        contamination_probability=contamination,
        seed=SIMULATOR_SEED
    )  # simulator代表真实系统
    robust_mhe_data, kalman_data, mhe_data = [], [], []
    for _ in trange(runs):
        simulator, kalman, mhe, robust_mhes = experiment_step(simulator)
        kalman_data.append(compute_mse_and_coverage(simulator, kalman))
        mhe_data.append(compute_mse_and_coverage(simulator, mhe))
        robust_mhe_data.append([compute_mse_and_coverage(simulator, robust_mhe) for robust_mhe in robust_mhes])

    return np.array(kalman_data), np.array(mhe_data), np.array(robust_mhe_data)


def run2(runs, contamination, simulator=None):
    observation_cov = NOISE_VAR * np.eye(2)
    simulator = ConstantVelocityModel(
        final_time=FINAL_TIME,
        time_step=TIME_STEP,
        observation_cov=observation_cov,
        explosion_scale=EXPLOSION_SCALE,
        contamination_probability=contamination,
        seed=SIMULATOR_SEED
    )  # simulator代表真实系统
    robust_mhe_error_list, kf_error_list, mhe_error_list = [], [], []
    for _ in trange(runs):
        simulator, kalman, mhe, robust_mhes = experiment_step(simulator)
        kf_error = simulator.X - np.squeeze(np.array(kalman.filter_means), axis=2)
        mhe_error = simulator.X - np.squeeze(np.array(mhe.filter_means), axis=2)
        robust_mhes_error = simulator.X - [np.squeeze(np.array(robust_mhe.filter_means), axis=2) for robust_mhe in robust_mhes]
        kf_error_list.append(kf_error)
        robust_mhe_error_list.append(robust_mhes_error)
        mhe_error_list.append(mhe_error)
    return np.array(kf_error_list), np.array(mhe_error_list), np.array(robust_mhe_error_list)


if __name__ == '__main__':
    mode = 2
    for contamination in CONTAMINATION:
        if mode == 1:
            BETA = [0.00001, 0.00002, 0.00004, 0.00006, 0.00008, 0.0001, 0.0002]
            results = run(NUM_RUNS, contamination)
            pickle_save(
                f'../results/constant_velocity/error_{contamination}.pk',
                results)
        elif mode == 2:
            BETA = [0.0001]
            results = run2(NUM_RUNS, contamination)
            pickle_save(
                f'../results/constant_velocity/original_{contamination}.pk',
                results)
        else:
            raise ValueError