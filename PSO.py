import random
import math
import copy
import matplotlib.pyplot as plt

# Rastrigin function
def fitness_rastrigin(position):
    fitness_val = 0.0
    for i in range(len(position)):
        xi = position[i]
        fitness_val += (xi * xi) - (10 * math.cos(2 * math.pi * xi)) + 10
    return fitness_val


# Particle class
class Particle:
    def __init__(self, position, velocity, fitness, best_position, best_fitness):
        self.position = position
        self.velocity = velocity
        self.fitness = fitness
        self.best_position = best_position
        self.best_fitness = best_fitness


# Particle Swarm Optimization function
def pso(fitness_func, dim, num_particles, max_iter, minx, maxx):

    inertia = 0.7
    congitiveW = 1.2
    socialW = 1.2

    particles = []
    best_swarm_position = [0.0] * dim
    best_swarm_fitness = float('inf')
    best_positions = []

    for _ in range(num_particles):
        position = [random.uniform(minx, maxx) for _ in range(dim)]
        velocity = [random.uniform(minx, maxx) for _ in range(dim)]
        fitness = fitness_func(position)
        particles.append(Particle(position, velocity, fitness, position, fitness))
        if fitness < best_swarm_fitness:
            best_swarm_fitness = fitness
            best_swarm_position = copy.deepcopy(position)

    fitness_values = []
    for _ in range(max_iter):
        for particle in particles:
            for k in range(dim):
                r1 = random.random()
                r2 = random.random()
                particle.velocity[k] = (inertia * particle.velocity[k] +
                                        congitiveW * r1 * (particle.best_position[k] - particle.position[k]) +
                                        socialW * r2 * (best_swarm_position[k] - particle.position[k]))
                particle.velocity[k] = max(particle.velocity[k], minx)
                particle.velocity[k] = min(particle.velocity[k], maxx)

            for k in range(dim):
                particle.position[k] += particle.velocity[k]
            particle.fitness = fitness_func(particle.position)
            if particle.fitness < particle.best_fitness:
                particle.best_fitness = particle.fitness
                particle.best_position = copy.deepcopy(particle.position)
                particle.best_position = copy.deepcopy(particle.position)
                if particle.fitness < best_swarm_fitness:
                    best_swarm_fitness = particle.fitness
                best_swarm_position = copy.deepcopy(particle.position)
                fitness_values.append(best_swarm_fitness)
                best_positions.append(copy.deepcopy(best_swarm_position))
    return best_positions, fitness_values


dim = 3
num_particles = 100
max_iterations = 100
best_positions, fitness_values = pso(fitness_rastrigin, dim, num_particles, max_iterations, -10.0, 10.0)


# Plot particles in 3D
def plot_particles(particles):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for particle in particles:
        fitness = particle.fitness
        marker_size = 10.0 * (1.0 - fitness / 50.0)
        ax.scatter(particle.position[0], particle.position[1], particle.position[2], c='b', marker='o', s=marker_size)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()


particles = [Particle(position, [], fitness, [], fitness) for position, fitness in zip(best_positions, fitness_values)]
plot_particles(particles)

# Plot fitness values
plt.plot(fitness_values)
plt.title('Fitness Values for PSO')
plt.show()

#mariia kyrychenko 192425