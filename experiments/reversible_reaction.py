import numpy as np

from robust_smc.data import ReversibleReaction
from robust_smc.nonlinearmhe import NonlinearMhe
from robust_smc.ukf import UKF
from robust_smc.robustnonlinearmhe import RobustifiedNonlinearMhe
from robust_smc.sampler import LinearGaussianBPF

from tqdm import trange

from sklearn.metrics import mean_squared_error, median_absolute_error
from experiment_utilities import pickle_save

# Experiment Settings
SIMULATOR_SEED = 1992
RNG_SEED = 24
NUM_RUNS = 100
BETA = [1e-5, 2e-5, 4e-5, 6e-5, 8e-5, 1e-4]
# CONTAMINATION = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
CONTAMINATION = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
# Sampler Settings
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
    # X_init = np.array([[0.1, 4.5]]) + prior_std[None, :] * RNG.randn(NUM_SAMPLES, NUM_LATENT)
    # X_init = X_init.squeeze()
    # vanilla_bpf = LinearGaussianBPF(
    #     data=Y,
    #     transition_matrix=simulator.transition_matrix,
    #     observation_model=simulator.observation_model,
    #     transition_cov=transition_cov,
    #     observation_cov=observation_cov,
    #     X_init=X_init,
    #     num_samples=NUM_SAMPLES,
    #     seed=seed
    # )
    # vanilla_bpf.sample()

    # UKF
    ukf = UKF(
        data=Y,
        transition_matrix=simulator.transition_matrix,
        transition_cov=transition_cov,
        observation_cov=observation_cov,
        m_0=np.array([0.1, 4.5]),
        P_0=np.diag(prior_std)**2
    )
    ukf.filter()

    # MHE
    mhe = NonlinearMhe(
        data=Y,
        transition_matrix=simulator.transition_matrix,
        transition_cov=transition_cov,
        observation_cov=observation_cov,
        m_0=np.array([0.1, 4.5]),
        P_0=np.diag(prior_std)**2
    )
    mhe.filter()

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
            P_0=np.diag(prior_std)**2
        )
        robust_mhe.filter()
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
    else:
        trajectories = np.stack(sampler.X_trajectories)
        mean = trajectories.mean(axis=1)
        quantiles = np.quantile(trajectories, q=[0.05, 0.95], axis=1)
        scores = []
        for var in range(NUM_LATENT):
            mse = mean_squared_error(simulator.X[:, var], mean[:, var])
            upper = simulator.X[:, var] <= quantiles[1, :, var]
            lower = simulator.X[:, var] >= quantiles[0, :, var]
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


if __name__ == '__main__':
    for contamination in CONTAMINATION:
        print('CONTAMINATION=', contamination)
        results = run(NUM_RUNS, contamination)
        pickle_save(f'../results/reversible_reaction/impulsive_noise_with_student_t/beta-sweep-contamination-{contamination}.pk', results)
