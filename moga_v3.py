"""
DEV NOTES:
        1. PARAMETERS ARE ALWAYS SUBJECT TO CHANGE ONCE REAL-WORLD DATA IS AVAILABLE
        2. EVALUATE FUNCTION STILL LACKS SOIL RETENTION AND RAINFALL VARIABLES
        3. STILL ON THE LOOKOUT FOR A BETTER MUTATE FUNCTION
"""

import numpy as np
import pandas as pd
import random
from deap import creator, tools, base, algorithms
import time
from statsmodels.tsa.arima.model import ARIMA


"""USING PYTHON'S DEAP LIBRARY FOR A MULTI-OBJECTIVE GENETIC ALGORITHM (MOGA) THAT AIMS TO PROVIDE
AN OPTIMAL SOLUTION TO WATER ALLOCATION (WEEKLY) AMONG FARMS IN NUEVA ECIJA DURING RICE SEASON."""


# WATER MEASUREMENTS ARE IN CUBIC METER
# THESE ARE CURRENTLY ALL DUMMY DATA
NUM_FARM = 10
FARM_SIZE = 1       # HECTARES(HA)
NUM_PERIODS = 4     # WEEKS

# CROP COEFFICIENT (Kc) VALUES FOR RICE GROWTH STAGE (FA0-56 RECOMMENDED VALUES) FOR GROWING SEASON
RICE_KC_VALUES = np.array([1.13, 1.28, 1.40, 1.03])  # [INITIAL, DEVELOPMENT, MID-SEASON, LATE SEASON], AVERAGE KC VALUES PER STAGE
KC_VALUES = np.zeros(NUM_PERIODS)
KC_VALUES.fill(RICE_KC_VALUES[0])  # ASSUME RICE IS STILL AT INITIAL STAGE

# REFERENCE ET₀ VALUES (MM/DAY)
ET0_VALUES = np.array([5.0, 4.8, 4.5, 4.2]) # REPLACE WITH REAL-WORLD VALUES

# COMPUTE WEEKLY ETc AND CONVERTS TO CUBIC METERS PER HECTARE
ETC_VALUES = (ET0_VALUES * KC_VALUES) * 10  # 1 MM OF WATER PER HECTARE = 10 M^3

WATER_DEMAND = ETC_VALUES * 7    # COMPUTE WATER DEMAND BASED ON ETc AND MULTIPLY BY 7 DAYS/WEEK

WEEKLY_WATER_DEMAND = WATER_DEMAND * NUM_FARM

# WATER SUPPLY FROM DAM
TOTAL_DAM_RELEASE = 518e6

# ADDS VARIABILITY SO THAT DAMN RELEASE IS NOT SAME FOR ALL RELEASE PERIOD -- REPLACE WITH REAL-WORLD DATA ON DAM RELEASES
TREND_FACTOR = np.array([0.8, 0.9, 1.1, 1.0])
NORMALIZED_TREND_FACTOR = TREND_FACTOR / TREND_FACTOR.sum()

# WEEKLY WATER SUPPLY -- CHANGE TO REAL WOLRD-DATA
WEEKLY_WATER_SUPPLY = TOTAL_DAM_RELEASE * NORMALIZED_TREND_FACTOR

# CONVERTS RAINFALL TO M^3
def rainfall_to_volume(rainfall_mm, conversion_factor=10):
    return rainfall_mm * conversion_factor


# ADJUSTABLE EFFIECENCY BASED ON HOW MUCH RAIN IS USABLE
def compute_effective_rainfall(volume, efficiency=0.75):
    return volume * efficiency


# ASSUME RETENTION RATE OF 20% OF ETC AS DUMMY
def compute_soil_storage(etc, retention_coeff=0.2):
    soil_storage = np.zeros_like(etc)
    for t in range(1, etc.shape[1]):
        soil_storage[:, t] = etc[:, t - 1] * retention_coeff

    return soil_storage

# DUMMY DATA FOR NOW
rainfall = np.random.gamma(shape=2.0, scale=10.0, size=52)

rainfall_series = pd.Series(rainfall)

model = ARIMA(rainfall_series, order=(1, 1, 1))
model_fit = model.fit()

forecast = model_fit.forecast(steps=NUM_PERIODS)

forecasted_rainfall = np.maximum(forecast, 0)

print("Forecasted Rainfall (mm):", forecasted_rainfall)

RAINFALL_VALUES = forecasted_rainfall.to_numpy()


creator.create('FitnessMulti', base.Fitness, weights=(1.0, 1.0, 1.0))
creator.create('Individual', list, fitness=creator.FitnessMulti)


def create_individual():
    individual = np.zeros((NUM_FARM, NUM_PERIODS))

    for i in range(NUM_PERIODS):
        available_water = WEEKLY_WATER_SUPPLY[i]

        # ADDS VARIABILITY TO WATER DISTRIBUTION IN INDIVIDUAL
        rand_f = np.random.uniform(0.9, 1.1, size=NUM_FARM) # WITHIN ±10 RANGE
        water_alloc = WEEKLY_WATER_DEMAND[i] * rand_f

        # ENSURES TOTAL ALLOCATED WATER DOES NOT EXCEED AVAILABLE WATER SUPPLY
        total_alloc = water_alloc.sum()
        excess = total_alloc - available_water
        if excess > 0:
            water_alloc -= (water_alloc / total_alloc) * excess  # PROPRTIONAL REDUCTION FOR OVER CAPPING

        individual[:, i] = water_alloc

    return creator.Individual(individual.flatten().tolist())


toolbox = base.Toolbox()
toolbox.register('individual', tools.initIterate, creator.Individual, create_individual)
toolbox.register('population', tools.initRepeat, list, toolbox.individual)


# EVALUATES EACH INIDVIDUAL BASE ON EQUITY, DEMAND FULFILLMENT, AND SUSTAINABILITY
# MIGHT NEED REFINEMENT ONCE REAL-WORLD DATA IS AVAILABLE
def evaluate(individual):   # FITNESS FUNCTION
    alloc_matrix = np.array(individual).reshape((NUM_FARM, NUM_PERIODS))

    # REFERENCES FOR THE FORMULAS USED IN THIS EVALUATE FUNCTION
    # Jain, R., Chiu, D. M., & Hawe, W. R. (1984). A Quantitative Measure of Fairness and Discrimination for Resource Allocation in Shared Computer Systems.
    # Loucks, D. P., & van Beek, E. (2017). Water Resource Systems Planning and Management.
    # FAO (Food and Agriculture Organization) Reports on Water Allocation and Sustainability.


    # THE CLOSER THE SCORES ARE TO 1 THE BETTER THE SCORES
    epsilon = 1e-6  # PREVENTS DIVISION BY ZERO IN EXTREME CASES

    etc = ETC_VALUES[np.newaxis, :]

    rainfall_volume = rainfall_to_volume(RAINFALL_VALUES)
    effective_rainfall = compute_effective_rainfall(rainfall_volume)
    effective_rain = effective_rainfall[np.newaxis, :]

    soil_storage = compute_soil_storage(etc)

    adjusted_demand = np.maximum(1, etc - effective_rain - soil_storage)
    total_demand = adjusted_demand.sum()

    # EQUITY SCORE = 1 - (STD(WATER ALLOCATION) / MEAN(WATER ALLOCATION))
    equity_score = 1 - (np.std(alloc_matrix) / (np.mean(alloc_matrix) + epsilon))
    equity_score = np.clip(equity_score, 0, 1) # NORMALIZE VALUES BETWEEN 0 TO 1 IF WATER ALLOCATION IS NEARLY UNIFORM

    # DEMAND FULFILLMENT SCORE = SUM(MIN(WATER ALLOCATION, WATER REQUIREMENT)) / SUM(WATER REQUIREMENT)
    fulfilled = np.sum(np.minimum(alloc_matrix, adjusted_demand))
    demand_fulfillment_score = fulfilled / (total_demand + epsilon)
    demand_fulfillment_score = np.clip(demand_fulfillment_score, 0, 1) # PREVENT SCORE FROM GOING OVER 1

    # SUSTAINABILITY SCORE = 1 - (SUM(WATER ALLOCATION) / AVAILABLE WATER SUPPLY)
    sustainability_score = 1 - (np.sum(alloc_matrix) / (np.sum(WEEKLY_WATER_SUPPLY) + epsilon))

    return float(np.round(equity_score, 4)), float(np.round(demand_fulfillment_score, 4)), float(np.round(sustainability_score, 4))


toolbox.register('evaluate', evaluate)


# STILL FNDING A BETTER ALGORITHM TO MUTATE
# GENERALLY SPEAKING, MUTATE INTRODUCES RANDOM CHANGES TO EACH INDIVIDUAL IN THE POPULATION
# THIS HELPS THE ALGORITHM TO HAVE A GENETIC DIVERSITY AND TO PREVENT IT FROM GETTING STUCK IN LOCAL OPTIMA
def mutate(individual, indpb=0.5, swap_prob=0.2, noise_factor=0.1):
    individual = np.array(individual).reshape((NUM_FARM, NUM_PERIODS))

    for i in range(NUM_PERIODS):
        for j in range(NUM_FARM):
            if random.random() < indpb:
                max_water = min(WEEKLY_WATER_DEMAND[i], WEEKLY_WATER_SUPPLY[i])
                range_limit = 0.3 * max_water   # DYNAMIC MUTATION RANGE
                min_bound = max(0, individual[j, i] - range_limit)
                max_bound = max(max_water, individual[j, i] + range_limit)

                # GAUSSIAN PERTURBATION
                delta = np.random.normal(0, range_limit * 0.2)
                new_value = individual[j, i] + delta

                # OCCASIONALLY SWAP ALLOCATIONS BETWEEN FARMS
                if random.random() < swap_prob:
                    swap_farm = random.randint(0, NUM_FARM - 1)
                    if individual[j, i] + individual[swap_farm, i] <= max_water:    # ENSURES FEASIBILITY
                        individual[j, i], individual[swap_farm, i] = individual[swap_farm, i], individual[j, i]

                # ADDS SLIGHT RANDOM NOISE TO HELP AVOID LOCAL MINIMA
                new_value += random.uniform(-noise_factor * max_water, noise_factor * max_water)

                individual[j, i] = np.clip(new_value, min_bound, max_bound)

    return creator.Individual(individual.flatten().tolist()),


toolbox.register("mutate", mutate)


# SIMULATED BINARY CROSSOVER (SBX) CREATES TWO OFFSPRING FROM TWO PARENTS BY GENERATING VALUES THAT LIE BETWEEN THE PARENT'S VALUES
# ALLOWS CONTROLLED EXPLORATION AND EXPLOITATION
# COMMON IN MOGA
def sbx_crossover(parent1, parent2, eta=5, adaptive=True):
    size = len(parent1)
    offspring1, offspring2 = parent1[:], parent2[:]
    
    for i in range(size):
        if random.random() < 0.9:
            u = random.random()
            if u <= 0.5:
                beta = (2 * u) ** (1 / (eta + 1))
            else:
                beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))

            if adaptive:
                beta *= 1.2 
            
            offspring1[i] = 0.5 * ((1 + beta) * parent1[i] + (1 - beta) * parent2[i])
            offspring2[i] = 0.5 * ((1 - beta) * parent1[i] + (1 + beta) * parent2[i])

    return creator.Individual(offspring1), creator.Individual(offspring2)


toolbox.register("mate", sbx_crossover)
toolbox.register("select", tools.selNSGA2, nd='standard')


def run_moga(pop_size=100, ngen=1000, cxpb=0.6, mutpb=0.4, stall_generations=300, min_improvement=1e-3):
    population = toolbox.population(n=pop_size)
    hof = tools.ParetoFront()

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register('min', np.min, axis=0)
    stats.register("avg", np.mean, axis=0)

    prev_best = None
    stall_count = 0

    all_individuals = []

    # eaMuPlusLambda USES AN ELITIST STRATEGY THAT HELPS MAINTAIN DIVERSITY AND ENSURES STEADY IMPROVEMENT IN SOLUTIONS
    for gen in range(ngen):
        # RUN ONE GENERATION AT A TIME
        population, logbook = algorithms.eaMuPlusLambda(
            population, toolbox,
            mu=pop_size, lambda_=pop_size,
            cxpb=cxpb, mutpb=mutpb,
            ngen=1, stats=stats,
            halloffame=hof, verbose=False
        )

        if len(hof) > 0:
            for ind in hof:
                if not ind.fitness.valid:
                    ind.fitness.values = toolbox.evaluate(ind)

            fitness_array = np.array([ind.fitness.values for ind in hof])
            best_fitness = np.max(fitness_array, axis=0)
        else:
            best_fitness = None

        # EARLY STOPPING THE ALGORITHM ONCE STALL COUNT IS REACHED
        if best_fitness is not None:
            if prev_best is not None:
                improvement = np.abs(prev_best - best_fitness).max()
                if improvement < min_improvement:
                    stall_count += 1
                else:
                    stall_count = 0

            prev_best = best_fitness

            print(f"Gen {gen + 1}/{ngen}: Best Fitness = {best_fitness} | Stalled: {stall_count}/{stall_generations}")

        if stall_count >= stall_generations:
            print(f"\nEarly stopping at generation {gen + 1} due to stagnation.")
            break

        all_individuals.extend(population)
        
    return population, hof, logbook, all_individuals


if __name__ == '__main__':
    start = time.time()
    final_population, pareto_front, logs, all_individuals = run_moga()

    if len(pareto_front) > 0:
        # FIND BEST SOLUTION FOR EACH OBJECTIVE
        best_solution = min(pareto_front, key=lambda ind: abs(ind.fitness.values[0] - 1)) # SELECTS THE VALUE CLOSER TO 1

        allocation_matrix = np.array(best_solution).reshape((NUM_FARM, NUM_PERIODS))
        fitness_scores = best_solution.fitness.values
            
        print(f"\nBest Solution for Monthly Water Allocation:\n")
        print(f'Weekly Water Allocation Matrix:\n(m³ allocated per farm across periods)\n {np.round(allocation_matrix, 2)}')
            
        print("\nObjective Scores:")
        print(f"Equity Score: {fitness_scores[0]:.4f}")
        print(f"Demand Fulfillment Score: {fitness_scores[1]:.4f}")
        print(f"Sustainability Score: {fitness_scores[2]:.4f}")
        print("-" * 60)

    else:
        print("No valid solutions found.")

    end = time.time()
    print(f'Time: {end - start:.6f}s')

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plotting
import numpy as np

# Extract fitness values of all individuals
fitnesses = [ind.fitness.values for ind in all_individuals if ind.fitness.valid]
fitnesses = np.array(fitnesses)

# 3D Scatter Plot of Objectives
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(fitnesses[:, 0], fitnesses[:, 1], fitnesses[:, 2], c='blue', marker='o', alpha=0.6)

ax.set_xlabel('Equity Score')
ax.set_ylabel('Demand Fulfillment Score')
ax.set_zlabel('Sustainability Score')
ax.set_title('Objective Trade-offs (All Individuals)')

plt.tight_layout()
plt.show()
