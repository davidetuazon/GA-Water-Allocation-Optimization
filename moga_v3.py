"""
DEV NOTES:
        1. PARAMETERS ARE ALWAYS SUBJECT TO CHANGE ONCE REAL-WORLD DATA IS AVAILABLE
        2. EVALUATE FUNCTION STILL LACKS SOIL RETENTION AND RAINFALL VARIABLES
        3. STILL ON THE LOOKOUT FOR A BETTER MUTATE FUNCTION
"""

import numpy as np
import random
from deap import creator, tools, base, algorithms
import time

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

# REPLACE WITH DATA FROM WEATHER FORECAST API
RAINFALL_VALUES = np.array([20, 30, 15, 10])
RAINFALL_VOLUME = RAINFALL_VALUES * 10 # CONVERT TO MM TO M^3

EFFECTIVE_RAINFALL = RAINFALL_VOLUME * 0.75 #ADJUSTABLE EFFICIENCY BASED ON HOW MUCH RAINFALL IS USABLE

# ASSUME RETENTION CAPACITY OF 20% OF ETC AS DUMMY
SOIL_RETENTION_COEFF = 0.2


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

    # EQUITY SCORE = 1 - (STD(WATER ALLOCATION) / MEAN(WATER ALLOCATION))
    equity_score = 1 - (np.std(alloc_matrix) / (np.mean(alloc_matrix) + epsilon))
    equity_score = np.clip(equity_score, 0, 1) # NORMALIZE VALUES BETWEEN 0 TO 1 IF WATER ALLOCATION IS NEARLY UNIFORM

    # ADD RAINFALL AND SOIL RETENTION VARIABLES
    effective_rainfall = EFFECTIVE_RAINFALL[np.newaxis, :]
    etc = ETC_VALUES[np.newaxis, :]

    # SIMULATE RETAINED WATER FOR PREVIOUS WEEK
    soil_storage = np.zeros((NUM_FARM, NUM_PERIODS))
    for t in range(1, NUM_PERIODS):
        soil_storage[:, t] = (etc[:, t - 1] * SOIL_RETENTION_COEFF)

    adjusted_demand = np.maximum(1, etc - effective_rainfall - soil_storage)
    total_demand = adjusted_demand.sum()

    # DEMAND FULFILLMENT SCORE = SUM(MIN(WATER ALLOCATION, WATER REQUIREMENT)) / SUM(WATER REQUIREMENT)
    demand_fulfillment_score = np.sum(np.minimum(alloc_matrix, adjusted_demand)) / (total_demand + epsilon)

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


def run_moga(pop_size=200, ngen=2000, cxpb=0.6, mutpb=0.4, stall_generations=500, min_improvement=1e-3):
    population = toolbox.population(n=pop_size)
    hof = tools.ParetoFront()

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register('min', np.min, axis=0)
    stats.register("avg", np.mean, axis=0)

    prev_best = None
    stall_count = 0

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
            best_fitness = np.array([ind.fitness.values for ind in hof]).max(axis=0)
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
        
    return population, hof, logbook


if __name__ == '__main__':
    start = time.time()
    final_population, pareto_front, logs = run_moga()

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