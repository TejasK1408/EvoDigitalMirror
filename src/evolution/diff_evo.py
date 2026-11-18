
import random


# ============================================================
# 1. BOUNDARY HANDLING
# ============================================================

def keep_inside_bounds(value, epsilon=1e-6):
    """
    Keep every gene strictly inside (0, 1).

    Reflection boundary handling:
    - Lower bound = epsilon
    - Upper bound = 1 - epsilon
    - Does not simply clip out-of-range values.
    """

    low = epsilon
    high = 1.0 - epsilon
    width = high - low

    value = (value - low) % (2 * width)

    if value > width:
        value = 2 * width - value

    return low + value


# ============================================================
# 2. DIFFERENTIAL MUTATION: DE/rand/1
# ============================================================

def differential_mutation(population, target_index, F=0.5):
    """
    Canonical DE/rand/1 mutation.

    V = X1 + F * (X2 - X3)

    Three distinct individuals are randomly selected,
    excluding the target individual.

    Boundary handling is performed using reflection.
    """

    if len(population) < 4:
        raise ValueError(
            "DE/rand/1 requires at least 4 individuals."
        )

    if not 0 <= target_index < len(population):
        raise IndexError("Invalid target index.")

    # Select three distinct donors, excluding the target.
    candidates = [
        i for i in range(len(population))
        if i != target_index
    ]

    r1, r2, r3 = random.sample(candidates, 3)

    X1 = population[r1]
    X2 = population[r2]
    X3 = population[r3]

    # Dictionary-based genotype.
    if isinstance(X1, dict):

        if not (
            X1.keys() == X2.keys()
            and X1.keys() == X3.keys()
        ):
            raise ValueError(
                "All genotypes must have identical keys."
            )

        mutant = {}

        for key in X1:

            value = (
                X1[key]
                + F * (X2[key] - X3[key])
            )

            mutant[key] = keep_inside_bounds(value)

        return mutant

    # List-based genotype.
    if not (
        len(X1) == len(X2) == len(X3)
    ):
        raise ValueError(
            "All genotypes must have equal lengths."
        )

    mutant = []

    for gene1, gene2, gene3 in zip(X1, X2, X3):

        value = gene1 + F * (gene2 - gene3)

        mutant.append(
            keep_inside_bounds(value)
        )

    return mutant


# ============================================================
# 3. BINOMIAL CROSSOVER: DE/bin
# ============================================================

def binomial_crossover(target, mutant, CR=0.7):
    """
    Binomial crossover.

    CR = probability of inheriting a gene from the mutant.

    At least one mutant gene is guaranteed through j_rand.
    """

    if isinstance(target, dict):

        keys = list(target.keys())

        if target.keys() != mutant.keys():
            raise ValueError(
                "Target and mutant gene keys must match."
            )

        if not keys:
            raise ValueError("Genotype cannot be empty.")

        j_rand = random.randrange(len(keys))

        offspring = {}

        for j, key in enumerate(keys):

            if random.random() < CR or j == j_rand:
                offspring[key] = mutant[key]

            else:
                offspring[key] = target[key]

        return offspring

    # List-based genotype.
    if len(target) != len(mutant):
        raise ValueError(
            "Target and mutant genotypes must have equal lengths."
        )

    if not target:
        raise ValueError("Genotype cannot be empty.")

    j_rand = random.randrange(len(target))

    offspring = []

    for j in range(len(target)):

        if random.random() < CR or j == j_rand:
            offspring.append(mutant[j])

        else:
            offspring.append(target[j])

    return offspring


# ============================================================
# 4. GENERATE ONE OFFSPRING
# ============================================================

def generate_offspring(population, target_index, F=0.5, CR=0.7):
    """
    Generate one trial individual for a given target.

    Steps:
        1. Select three random donor individuals.
        2. Apply DE/rand/1 mutation.
        3. Apply binomial crossover.
    """

    target = population[target_index]

    mutant = differential_mutation(
        population,
        target_index,
        F
    )

    offspring = binomial_crossover(
        target,
        mutant,
        CR
    )

    return mutant, offspring


# ============================================================
# 5. GENERATE AN ENTIRE OFFSPRING POPULATION
# ============================================================

def evolve_population(population, F=0.5, CR=0.7):
    """
    Generate one trial individual for every target individual.

    Population size remains unchanged.
    """

    if len(population) < 4:
        raise ValueError(
            "Population must contain at least 4 individuals."
        )

    offspring_population = []

    for i in range(len(population)):

        _, offspring = generate_offspring(
            population,
            i,
            F,
            CR
        )

        offspring_population.append(offspring)

    return offspring_population


# ============================================================
# 6. GREEDY SELECTION
# ============================================================

def selection(population, offspring_population,
              fitness_population, fitness_offspring):
    """
    Target-versus-trial greedy selection.

    The individual with the better fitness survives.

    Assumption:
        Higher fitness is better.
    """

    if not (
        len(population)
        == len(offspring_population)
        == len(fitness_population)
        == len(fitness_offspring)
    ):
        raise ValueError(
            "Population and fitness lengths must match."
        )

    next_population = []
    next_fitness = []

    for i in range(len(population)):

        if fitness_offspring[i] >= fitness_population[i]:

            next_population.append(
                offspring_population[i]
            )

            next_fitness.append(
                fitness_offspring[i]
            )

        else:

            next_population.append(
                population[i]
            )

            next_fitness.append(
                fitness_population[i]
            )

    return next_population, next_fitness


# ============================================================
# 7. TESTING
# ============================================================

if __name__ == "__main__":

    # Ten individuals, six genes per individual.
    population = [
        [0.2, 0.4, 0.6, 0.3, 0.7, 0.5],
        [0.3, 0.5, 0.2, 0.6, 0.4, 0.8],
        [0.7, 0.2, 0.5, 0.4, 0.6, 0.3],
        [0.5, 0.8, 0.3, 0.2, 0.9, 0.4],
        [0.6, 0.3, 0.8, 0.5, 0.2, 0.7],
        [0.4, 0.6, 0.7, 0.8, 0.3, 0.2],
        [0.8, 0.5, 0.4, 0.7, 0.6, 0.3],
        [0.3, 0.7, 0.6, 0.4, 0.5, 0.9],
        [0.9, 0.4, 0.3, 0.5, 0.7, 0.6],
        [0.5, 0.2, 0.9, 0.3, 0.4, 0.8]
    ]

    F = 0.5
    CR = 0.7

    print("\nINITIAL POPULATION")
    for i, individual in enumerate(population):
        print(f"X{i}: {individual}")

    offspring_population = evolve_population(
        population,
        F=F,
        CR=CR
    )

    print("\nGENERATED OFFSPRING")
    for i, individual in enumerate(offspring_population):
        print(f"U{i}: {individual}")

    # Verify boundary conditions.
    for individual in offspring_population:
        for gene in individual:
            assert 0 < gene < 1

    print("\nBoundary verification passed.")
    print("Initial population size:", len(population))
    print("Offspring population size:", len(offspring_population))
