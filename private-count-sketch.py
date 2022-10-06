# Run this script to produce all figures for Improved Utility Analysis of Private CountSketch

import numpy as np
from scipy.special import gammaln
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import gzip
import os

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
    return np.quantile(gaussian_noise, quantile, axis = 1)

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


# Sparse vectors (event-level privacy)

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


# Error distribution on world cities vectors

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


# Cumulative noise distribution on two market basked datasets

# Download data if needed:
for filename in ['kosarak.dat.gz', 'retail.dat.gz']:
    if not os.path.isfile(filename):
        response = requests.get("http://fimi.uantwerpen.be/data/" + filename)
        open(filename, "wb").write(response.content)

repetitions = 5

for (filename, countsketch_table_size, max_basket_size) in [("retail",500,30),("kosarak",1000, 100)]:
    plt.figure()
    skipped = 0
    f = gzip.open(filename+".dat.gz", "rt")
    item_counts = {}
    for line in f:
        if line.strip() != '':
            basket = line.strip().split(" ")
            if len(basket) > max_basket_size: #clip to maximum basket size
                skipped += len(basket) - max_basket_size
                basket = basket[:max_basket_size]
            for x in basket:
                item_counts[x] = item_counts.get(x, 0) + 1
    
    print(f'{filename}: {len(item_counts)} distinct elements, {sum([item_counts[x] for x in item_counts])} total elements, {skipped} skipped')
    item_counts = np.array(sorted([ item_counts[x] for x in item_counts ]))
    input_size = len(item_counts)
    cp = 1 / countsketch_table_size # collision probability
    
    cs_errors = []
    for _ in range(repetitions):
        data = np.random.choice([-1, 1], p=[1/2, 1/2], size=input_size)
        row = np.random.randint(0, countsketch_table_size, size=input_size)
        col = np.array(range(input_size))
        cs_matrix = coo_matrix((data, (row, col)), shape=(countsketch_table_size, input_size))
        cs_errors.append(cs_matrix @ item_counts)
    
    cs_errors = np.array(cs_errors)
    non_private_errors = np.median(cs_errors, axis = 0)
    gaussian_noise = np.random.normal(size=(repetitions, countsketch_table_size), scale=max_basket_size*noise_scaling_factor*np.sqrt(repetitions))
    # Noise scaled with maximum basket size
    private_errors = np.median(cs_errors + gaussian_noise, axis = 0)
    error_difference = np.abs(private_errors) - np.abs(non_private_errors)
    sns.ecdfplot(private_errors, label=f"private CS, $\epsilon$={epsilon}, $\delta=${delta}")
    splot = sns.ecdfplot(non_private_errors, label=f"standard CS")
    splot.set_xlabel("Estimation error (up to)")

    plt.title(f'Estimation error CDF {filename} (max basket size {max_basket_size}), k={repetitions}, b={countsketch_table_size}')
    plt.xlim(-np.percentile(private_errors,98),np.percentile(private_errors,98))
    plt.legend()
    plt.savefig(f'cdf-{filename}-{repetitions}-{countsketch_table_size}.pdf')
    plt.show(block=False)
