import math
import random
import numpy as np
import copy

smp = 5    # seeking memory pool
srd = 2     # seeking range of the selected dimension
cdc = 10    # counts of dimension to change
spc = True  # self-position considering
mr = 0.30   # mixture ratio
cats_num = 50 # number of cats
operations_num = 100
machines_num = 10
iterations = 1000

class CatSwarmOptmization:
    n = 0
    smp = 0
    spc = False
    srd = 0
    cdc = 0

    def __init__(self,
        n: int, 
        smp: int,
        spc: bool,
        cdc: int
    ) -> None:
        self.n = n
        self.smp = smp
        self.spc = spc
        self.srd = random.randint(0, n-1)
        self.cdc = cdc
    
    def seekingMode(self, position: list(int)) -> list(int):
        candidates = []
        look_around = 0

        # If SPC is True, consider the current position as a candidate
        # Then look around for at last SMP positions
        if self.spc:
            fitness = self.getFitness(position)
            candidates = [(position, fitness)]
            look_around = self.smp-1
        else:
            look_around = self.smp

        # Look around, keeping the max and min fitness found
        max_fitness = 0
        min_fitness = math.inf
        for i in range(look_around):
            mutation = self.mutate(position)

            # Generate random SRD
            self.srd = random.randint(0, self.n-1)

            fitness = self.getFitness(mutation)

            if fitness > max_fitness:
                max_fitness = fitness
            if fitness < min_fitness:
                min_fitness = fitness

            candidates.append((mutation, fitness))
        
        # If the max and min fitness found are different, select the position with max probability
        if max_fitness != min_fitness:
            probabilityFunction = self.getCandidateProbabilityFunction(max_fitness, min_fitness)
            candidates.sort(key=probabilityFunction, reverse=True)
            return candidates[0][0]
        
        # If the max and min fitness found are equal, select the current position
        return position

    def getCandidateProbabilityFunction(self, max_fitness: float, min_fitness) -> function:
        def probabilityFunction(candidate: tuple(list(int), float)) -> float:
            return abs(candidate[1]-min_fitness)/(max_fitness-min_fitness)
        
        return probabilityFunction       
    
    def mutate(self, position: list(int)) -> list(int):
        borders = []
        borders.append(self.srd)
        borders.append((self.cdc + self.srd) % self.n)
        borders.sort()

        mid = (borders[0]+borders[1])/2

        # Switch values between borders
        for i in range(mid):
            position[borders[0]+i] += position[borders[1]-i]
            position[borders[1]-i] = position[borders[0]+i]
            position[borders[0]+i] -= position[borders[1]-i]

        return position
    
    def tracingMode(self, cat, best_cat, vel_range):
        w = 0.7
        r = random.random(0,1)
        c = 2.05
        velocity = w * cat[vel] + r * c * (best_cat[pos] - cat[pos])
        velocity = np.where(velocity > vel_range, vel_range, velocity)
        cat[pos] += velocity
        
        return cat



    def getFitness(self, position: list(int)) -> float:
        pass


class Cat:

    def __init__(self, operations):
        self.position = operations

    def fitness(self):
        pass

    def apply_mode(self):
        if self.sm:
            self.__apply_sm()
        else:
            self.__apply_tm()

    def __apply_sm(self):
        cat_copies = []
        j = smp
        if spc:
            j = smp - 1
            cat_copies.append(self)

        for i in range(0, j):
            cat_copies.append(copy.deepcopy(self))

        for cat in cat_copies:
            srd = random.randrange(0, operations_num)

            if (srd+cdc > operations_num):
                mutation = cat.position[srd - (len(cat.position) - cdc):srd+1]
                mutation = mutation[::-1]
                new_position = cat.position[0:srd - (len(cat.position) - cdc)] + mutation + cat.position[srd+1:]
                assert len(new_position) == operations_num
                cat.position = new_position
            else:
                mutation = cat.position[srd+1:srd+cdc]
                mutation = mutation[::-1]
                new_position = cat.position[0:srd+1] + mutation + cat.position[srd+cdc:]
                assert len(new_position) == operations_num
                cat.position = new_position

        self_fitness = self.fitness()
        for cat in cat_copies:
            new_fitness = cat.fitness()
            if new_fitness < self_fitness:
                self.position = cat.position
                break


    def __apply_tm(self):        
        pass

class Machine:
    def __init__(self):
        self.time = 0
        self.current_operation = None

class Operation:
    def __init__(self, job, machine, time):
        self.job = job
        self.machine = machine
        self.time = time

    def __repr__(self):
        return "<Operation job:%s, machine:%s, time:%s>" % (self.job, self.machine, self.time)
        
def read_input():
    times = np.genfromtxt("./times.csv", dtype=int, delimiter=",")
    machines = np.genfromtxt("./machines.csv", dtype=int, delimiter=",")
    return times, machines

def parse_input(times, machines):
    jobs = times.shape[0] 
    operations = []
    for x, y in np.ndindex(times.shape):
        operation = Operation(x % jobs + 1, machines[x, y], times[x, y])
        operations.append(operation)
    return operations

def main():
    times,machines = read_input()
    operations = parse_input(times, machines)
    cats = []
    for i in range(0, cats_num):
        operations = copy.deepcopy(operations)
        random.shuffle(operations)
        cats.append(Cat(operations))
main()