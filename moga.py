import time
import random
import numpy as np
from deap import base, creator, tools, algorithms

"""ALL CONSTANTS ARE SUBJECT TO CHANGE WITH REAL-WORLD DATA"""

# all water units in cubic meter
NUM_FARMS = 10      # land
FARM_SIZE = 1       # hectares
NUM_PERIODS = 4     # weeks
WATER_DEMAND_PER_HA = 15000

# demand distribution for rice throughout growing season
WATER_DEMAND_DISTRIBUTION = np.array([0.30, 0.25, 0.20, 0.15, 0.10])

# total water demand for all farm throughout growing season
TOTAL_FARM_WATER_DEMAND = NUM_FARMS * FARM_SIZE * WATER_DEMAND_PER_HA

# total crop water requirement per period
TOTAL_CROP_WATER_DEMAND = TOTAL_FARM_WATER_DEMAND * WATER_DEMAND_DISTRIBUTION

# water supply
TOTAL_DAM_RELEASE = 467e6
BASE_DAM_RELEASE = TOTAL_DAM_RELEASE / NUM_PERIODS

WATER_SUPPLY = np.array([
    BASE_DAM_RELEASE * 0.8,  # 80% of base release in month 1
    BASE_DAM_RELEASE * 0.9,  # 90% in month 2
    BASE_DAM_RELEASE,        # 100% in month 3
    BASE_DAM_RELEASE * 1.1,  # 110% in month 4
    BASE_DAM_RELEASE * 0.7   # 70% in month 5
])

# print(TOTAL_FARM_WATER_DEMAND)
# print(TOTAL_CROP_WATER_DEMAND)
# print(WATER_SUPPLY)


creator.create('FitnessMulti', base.Fitness, weights=(-2.0, -2.0, -2.0))
creator.create('Individual', list, fitness=creator.FitnessMulti)

"""CHANGES WITH REAL-WORLD DATA"""
def create_individual():
    individual = np.zeros((NUM_FARMS, NUM_PERIODS))

    for i in range(NUM_PERIODS):
        available_water = WATER_SUPPLY[i]

        # distributes total crop water demand through 10 farms
        expected_per_farm = TOTAL_CROP_WATER_DEMAND[i] / NUM_FARMS

        rand_f = np.random.uniform(0.8, 1.2, size=NUM_FARMS)
        
        # applies variation
        water_alloc = expected_per_farm * rand_f

        if water_alloc.sum() > available_water:
            water_alloc *= (available_water / water_alloc.sum()) 

        individual[:, i] = water_alloc

    return creator.Individual(individual.flatten().tolist())


toolbox = base.Toolbox()
toolbox.register('individual', tools.initIterate, creator.Individual, create_individual)
toolbox.register('population', tools.initRepeat, list, toolbox.individual)


"""CALCULATIONS NEEDS DOUBLE CHECKING"""
def evaluate(individual):
    alloc_matrix = np.array(individual).reshape((NUM_FARMS, NUM_PERIODS))

    # expected water alloc per farm throughout all periods
    # compute the normalized std per period
    # compute total eq score
    expected_per_farm = TOTAL_CROP_WATER_DEMAND / NUM_FARMS
    std_per_period = np.std(alloc_matrix, axis=0)
    equity_score = np.mean(std_per_period / expected_per_farm)

    # compute unmet demands for allocation
    unmet_demand = np.maximum(TOTAL_CROP_WATER_DEMAND - alloc_matrix.sum(axis=0), 0)
    demand_score = unmet_demand.sum() / TOTAL_CROP_WATER_DEMAND.sum()

    # compute overuse
    overuse = np.maximum(alloc_matrix.sum(axis=0) - TOTAL_CROP_WATER_DEMAND, 0)
    overuse_score = overuse.sum() / TOTAL_CROP_WATER_DEMAND.sum()

    return float(np.round(equity_score, 6)), float(np.round(demand_score, 6)), float(np.round(overuse_score, 6))

toolbox.register('evaluate', evaluate)


"""MIGHT CHANGE OR JUST TWEAK/TUNE FEW PARAMETERS"""
def mutate(individual, eta=10, indpb=0.7, adaptive=True):
    individual = np.array(individual).reshape((NUM_FARMS, NUM_PERIODS))

    for i in range(NUM_PERIODS):
        for j in range(NUM_FARMS):
            if random.random() < indpb:
                max_water = min(TOTAL_CROP_WATER_DEMAND[i], WATER_SUPPLY[i])

                range_factor = 0.3 if adaptive else 0.15  
                range_limit = range_factor * max_water

                min_bound = max(0, individual[j, i] -  range_limit)
                max_bound = min(max_water, individual[j, i] + range_limit)

                # Polynomial mutation
                delta = (random.random() ** (1 / (eta + 1))) - 0.5
                new_value = individual[j, i] + delta * (max_bound - min_bound)

                individual[j, i] = np.clip(new_value, min_bound, max_bound)

    return creator.Individual(individual.flatten().tolist()),

toolbox.register("mutate", mutate)


"""SUBJECT TO CHANGE ---> MIGHT LOOK FOR A BETTER CROSSOVER METHOD"""
def sbx_crossover(parent1, parent2, eta=15, adaptive=True):
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


"""STAGNATES TOO EARLY, WILL THIS ISSUE SOON"""
def run_moga(pop_size=200, ngen=2000, cxpb=0.65, mutpb=0.35, stall_generations=250, min_improvement=1e-3):
    population = toolbox.population(n=pop_size)
    hof = tools.ParetoFront()

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register('min', np.min, axis=0)
    stats.register("avg", np.mean, axis=0)

    prev_best = None
    stall_count = 0

    for gen in range(ngen):
        population, logbook = algorithms.eaMuPlusLambda(
            population, toolbox,
            mu=pop_size, lambda_=pop_size,
            cxpb=cxpb, mutpb=mutpb,
            ngen=1, stats=stats,
            halloffame=hof, verbose=False
        )

        best_fitness = np.min([ind.fitness.values for ind in hof], axis=0) if len(hof) > 0 else None

        if best_fitness is not None:
            if prev_best is not None:
                improvement = np.abs(prev_best - best_fitness).max()
                if improvement < min_improvement:
                    stall_count += 1
                    if stall_count % 100 == 0:
                        # Increase mutation to encourage exploration
                        mutpb = min(0.8, mutpb + 0.05)
                        cxpb = max(0.5, cxpb - 0.05)
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
        # find best solution for each objective separately
        best_solutions = [
            min(pareto_front, key=lambda ind: ind.fitness.values[i]) 
            for i in range(len(pareto_front[0].fitness.values))
        ]

        objective_names = ["EQUITY", "DEMAND FULLFILMENT", "SUSTAINABILITY"]
        
        for i, best_solution in enumerate(best_solutions):
            allocation_matrix = np.array(best_solution).reshape((NUM_FARMS, NUM_PERIODS))
            fitness_scores = best_solution.fitness.values
            
            print(f"\nBest Solution for {objective_names[i]}:\n")
            print(f'Allocation Matrix:\n {np.round(allocation_matrix, 4)}')
            
            print("\nObjective Scores:")
            print(f"Equity Score: {fitness_scores[0]:.4f}")
            print(f"Demand Fulfillment Score: {fitness_scores[1]:.4f}")
            print(f"Sustainability Score: {fitness_scores[2]:.4f}")
            print("-" * 60)

    else:
        print("No valid solutions found.")

    end = time.time()
    print(f'Time: {end - start:.6f}s')
