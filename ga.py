import math
import numpy as np
import matplotlib.pyplot as plt


# Rastrigin function
def fitness_rastrigin(position):
    fitness_val = 0.0
    for i in range(len(position)):
        xi = position[i]
        fitness_val += (xi * xi) - (10 * math.cos(2 * math.pi * xi)) + 10
    return fitness_val


# Genetic Algorithm implementation
def genetic_algorithm(population_size, chromosome_length, num_generations):
    population = np.random.uniform(low=-5.0, high=5.0, size=(population_size, chromosome_length))

    best_fitness_values = []
    best_individuals = []

    for generation in range(num_generations):
        fitness_values = np.array([fitness_rastrigin(individual) for individual in population])
        parents = np.empty_like(population)
        for i in range(population_size):
            random_indices = np.random.choice(population_size, size=2, replace=False)
            parent1 = population[random_indices[0]]
            parent2 = population[random_indices[1]]
            if fitness_values[random_indices[0]] < fitness_values[random_indices[1]]:
                parents[i] = parent1
            else:
                parents[i] = parent2
        crossover_point = np.random.randint(1, chromosome_length)
        offspring = np.empty_like(population)
        for i in range(0, population_size, 2):
            parent1 = parents[i]
            parent2 = parents[i + 1]
            offspring[i, :] = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
            offspring[i + 1, :] = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
        mutation_rate = 0.05
        mutation_mask = np.random.uniform(size=offspring.shape) < mutation_rate
        mutation_values = np.random.uniform(low=-0.1, high=0.1, size=offspring.shape)
        offspring = np.where(mutation_mask, offspring + mutation_values, offspring)
        population = offspring
        best_fitness = np.min(fitness_values)
        best_individual = population[np.argmin(fitness_values)]
        best_fitness_values.append(best_fitness)
        best_individuals.append(best_individual)
        print("Generation:", generation + 1, "Best Fitness:", best_fitness)

    return best_fitness_values, best_individuals


population_size = 100
chromosome_length = 10
num_generations = 100

best_fitness_values, best_individuals = genetic_algorithm(population_size, chromosome_length, num_generations)

X = np.linspace(-5.0, 5.0, 100)
Y = np.linspace(-5.0, 5.0, 100)
X, Y = np.meshgrid(X, Y)
Z = (X ** 2 - 10 * np.cos(2 * np.pi * X) + 10) + (Y ** 2 - 10 * np.cos(2 * np.pi * Y) + 10)
# Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot Rastrigin function surface
ax.plot_surface(X, Y, Z, cmap='jet', alpha=0.5)

# Plot children of each population
for i in range(len(best_individuals)):
    child = best_individuals[i]
    child_fitness = best_fitness_values[i]

    ax.scatter(child[0], child[1], child_fitness, c='red', marker='o')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Fitness')

plt.title('Genetic Algorithm: Rastrigin Function')
plt.show()

# Plotting fitness values
plt.plot(range(num_generations), best_fitness_values)
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.title('Fitness Values over Generations')
plt.show()

#mariia kyrychenko 192425