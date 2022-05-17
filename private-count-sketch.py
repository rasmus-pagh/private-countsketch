# Run this script to produce all figures for Improved Utility Analysis of Private CountSketch

import numpy as np
from scipy.special import gammaln
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()
trials = 1000000
epsilon = 1
delta = 1e-6
noise_scaling_factor = np.sqrt(2 * np.log(1/delta)) / epsilon

repetition_list = [3,7,15,31,63]

# Standard Gaussian
standard_gaussian_noise = np.random.normal(size=trials, scale=1)


# Median of k normals with mean zero and variance k

def quantile_gaussian_noise(trials, repetitions, quantile=0.5):
    gaussian_noise = np.random.normal(size=(trials, repetitions), scale=np.sqrt(repetitions))
    return np.quantile(gaussian_noise, quantile, axis = 1) #, method='closest_observation')

for repetitions in repetition_list:
    sns.kdeplot(quantile_gaussian_noise(trials, repetitions), label=f"k={repetitions}")

plt.title('Median of k Gaussians with mean zero and variance k')
plt.xlim(-5,5)
plt.legend()
if trials >= 1000000:
    plt.savefig('median-of-normals.pdf')
plt.show(block=False)


# Additional error over Countsketch in a zero variance setting

plt.figure()
for repetitions in repetition_list:
    sns.ecdfplot(np.abs(quantile_gaussian_noise(trials, repetitions)), label=f"k={repetitions}")

sns.ecdfplot(np.abs(standard_gaussian_noise), label=f"Std. Gaussian", linestyle="dotted")

plt.title('Additional error over Countsketch in a zero variance setting')
plt.xlim(0,5)
plt.legend()
if trials >= 1000000:
    plt.savefig('zero-variance-setting.pdf')
plt.show(block=False)


# Table with average absolute value for median of k normals with mean zero and variance k

def average_abs_median_gaussian(trials, repetitions):
    return np.linalg.norm(quantile_gaussian_noise(trials, repetitions), ord=1) / trials

for repetitions in range(1,7,2):
    print(f"k={repetitions}: {average_abs_median_gaussian(trials, repetitions)}")


# Added noise for median of k normals with mean zero and variance k

count_sketch_error_probability = 1 # Probability of error
count_sketch_error_value = 10000 # Value of error, otherwise zero

plt.figure()
for repetitions in repetition_list:
    cs_errors = np.random.choice([-count_sketch_error_value, 0, count_sketch_error_value], p=[count_sketch_error_probability / 2, 1 - count_sketch_error_probability, count_sketch_error_probability / 2], size=(trials, repetitions))
    gaussian_noise = np.random.normal(size=(trials, repetitions), scale=np.sqrt(repetitions))
    non_private_errors = np.median(cs_errors, axis = 1)
    private_errors = np.median(cs_errors + gaussian_noise, axis = 1)
    error_difference = np.abs(private_errors) - count_sketch_error_value
    sns.kdeplot(error_difference, label=f"k={repetitions}")

sns.kdeplot(standard_gaussian_noise, label=f"Std. Gaussian", linestyle="dotted")

plt.title(f'Additional error over Countsketch in a high variance setting')
plt.xlim(-20,5)
plt.legend()
if trials >= 1000000:
    plt.savefig('high-variance-setting.pdf')
plt.show(block=False)



# Failure probability of CountSketch with k repetitions

def binomial(k, i):
    return np.exp(gammaln(k+1) - gammaln(i+1) - gammaln(k-i+1))

def count_sketch_failure_probability(k, fp):
    return 2 * sum([ (fp**i) * binomial(k, i) * ((1-fp)**(k-i)) for i in range((k+1)//2, k+1) ])

detailed_repetition_list = range(1, 64, 2)
beta_values = [1/4, 1/2, 3/4]

plt.figure()
for beta in beta_values:
    failure_probabilities = [ count_sketch_failure_probability(k, (1-beta)/2) for k in detailed_repetition_list ]
    splot = sns.lineplot(x = detailed_repetition_list, y = failure_probabilities, label=f"$\\beta={beta}$")

splot.set(yscale="log")
plt.title(f'Failure probability of Countsketch')
splot.set_xlabel("Number of repetitions (k)")
plt.savefig('countsketch-failure-probability.pdf')
plt.show(block=False)


# Sparse vectors
load_factor = 1. # Expected number of nonzero entries in each bucket
entry_size = 10

plt.figure()
for repetitions in [1, 3, 7, 15, 31, 63]: #, 127, 255]:
    cs_errors = np.random.poisson(lam=load_factor/2, size=(trials, repetitions)) - np.random.poisson(lam=load_factor/2, size=(trials, repetitions))
    cs_errors *= entry_size
    gaussian_noise = np.random.normal(size=(trials, repetitions), scale=np.sqrt(repetitions))
    pcs_errors = np.median(cs_errors + gaussian_noise, axis = 1)
    sns.ecdfplot(pcs_errors, label=f"k={repetitions}")

sns.ecdfplot(standard_gaussian_noise, label=f"Std. Gaussian", linestyle="dotted")
plt.xlim(-1.5 * entry_size, 1.5 * entry_size)
plt.title('Cumulative error distribution for sparse vectors, entries in $\{0,' + str(entry_size) + '\}$')
plt.legend()
if trials >= 1000000:
    plt.savefig(f'sparse-error-load-{load_factor}.pdf')
plt.show(block=False)


# Real-world examples

filename = "world_cities_population_simplemaps.txt"
f = open(filename, "r")
population_counts = []
for line in f:
    if line.strip() != '':
        population_counts.append(int(line.strip()))

d = len(population_counts)
x = np.array(population_counts)

def city_sketch_figure(cities_trials, kb_pairs, xlimit, noise_scale):
    plt.figure()
    for (k,b) in kb_pairs:
        errors = []
        for _ in range(cities_trials):
            A = np.random.choice([-1,0,+1], p=[1/(2*b),1-1/b,1/(2*b)], size = (k, d))
            errors.append(np.median(np.dot(A, x) + np.random.normal(scale = np.sqrt(k) * noise_scale, size = k)))
        sns.kdeplot(errors, label=f"k={k}, b={b}")
    ymin, ymax = plt.gca().get_ylim()
    sns.kdeplot(standard_gaussian_noise * noise_scale, label=f"Std. Gaussian", linestyle="dotted")
    plt.title(f'World cities sketch error with DP noise of magnitude {"{:.0e}".format(noise_scale)}')
    plt.xlim(-xlimit, xlimit)
    plt.ylim(ymin, ymax)
    plt.legend()
    if cities_trials > 1000:
        plt.savefig(f'world-cities-noise-{noise_scale}.pdf')
    plt.show(block=False)

cities_trials = 10000

xlimit = 3000000
noise_scale = 100000
kb_pairs = [ (1,1000), (3,1000), (5,1000), (9,1000), (19,1000) ]
city_sketch_figure(cities_trials, kb_pairs, xlimit, noise_scale)

noise_scale = 10000
xlimit = 300000
kb_pairs = [ (1,10000), (3,10000), (5,10000), (9,10000), (19,10000) ]
city_sketch_figure(cities_trials, kb_pairs, xlimit, noise_scale)

