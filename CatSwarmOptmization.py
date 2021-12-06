import math
import random
import numpy as np

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

main()