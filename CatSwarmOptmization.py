import math
import random
import numpy as np

class Cat:
    sm = True
    fitness = 0
    velocity = 1

    def __init__(self, position: list(int), sm: bool, fitness: float):
        self.position = position
        self.sm = sm
        self.fitness = fitness

class CatSwarmOptmization:
    n = 0
    smp = 0
    spc = False
    srd = 0
    cdc = 0
    mr = 0
    swarm = []
    max_velocity = 0

    def __init__(
        self, 
        operations,
        n: int, 
        smp: int,
        spc: bool,
        cdc: int,
        mr: float
    ) -> None:
        self.operations = operations
        self.n = n
        self.smp = smp
        self.spc = spc
        self.srd = random.randint(0, n-1)
        self.cdc = cdc
        self.mr = mr

    def mixtureStates(self):
        for cat in self.swarm:
            sm = random.random() > self.mr
            cat.sm = sm
            

    def initializePopulation(self):
        for i in range(self.n):
            position = [x for x in range(1, len(self.operations))]
            random.shuffle(position)
            fitness = self.getFitness(position)
        
            cat = Cat(position=position, fitness=fitness)
            self.swarm.append(cat)

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
    
    def tracingMode(self, cat, best_cat):
        '''w = 0.7
        r = random.random(0,1)
        c = 2.05
        velocity = w * cat.velocity + r * c * (best_cat[pos] - cat[pos])
        velocity = np.where(velocity > self.max_velocity, self.max_velocity, velocity)
        cat[pos] += velocity
        
        return cat'''
        return cat.position

    def rank(self):
        rank = [cat for cat in self.swarm]
        rank.sort(key=lambda cat: cat.fitness)
        
        return rank
        
    def applyMode(self, cat: Cat):
        new_position = []
        if cat.sm:
            new_position = self.seekingMode(cat.position)
        else:
            rank = self.rank()
            new_position= self.tracingMode(cat, rank[0])
        
        cat.position = new_position
        cat.fitness = self.getFitness(new_position)
   
    def getFitness(self, position: list(int)) -> float:
        schedule = Schedule(position, self.operations)
        end_time = schedule.getEndTime()

        return math.exp(-end_time)

class Machine:
    def __init__(self):
        self.time = 0
        self.current_operation = None

class Operation:
    def __init__(self, number, job, machine, time):
        self.number = number 
        self.job = job
        self.machine = machine
        self.time = time

    def __repr__(self):
        return "<Operation number %s: job:%s, machine:%s, time:%s>" % (self.number, self.job, self.machine, self.time)

class Schedule:
    sequence = []
    operations = []

    jobs_next_free_time = {}
    machines_schedule = {}

    def __init__(self, sequence: list(int), operations: list(Operation)):
        self.sequence = sequence
        self.operations = operations

    def buildMachinesSchedule(self):
        for op_index in self.sequence:
            # Find job and machine free times
            op = [op for op in self.operations if op.number == op_index][0]
            job_next_free_time = self.jobs_next_free_time[op.job]
            machine_next_free_time = self.findMachineAvailableTime(op.machine, job_next_free_time, op.duration)

            # Reconcile op start time 
            op_start_time = max(job_next_free_time, machine_next_free_time)

            self.jobs_next_free_time[op.job] = op_start_time

            # Add op event to machine schedule
            self.machines_schedule[op.machine].append((op_start_time, op_start_time+op.duration))
            self.machines_schedule[op.machine].sort(key=lambda event: event[0])

    def findMachineAvailableTime(self, machine:int, min_start: int, duration: int) -> int:
        schedule = self.machines_schedule[machine]

        # If nothing scheduled, return 0
        if len(schedule) == 0:
            return 0

        for i, event in enumerate(schedule):
            next_event = schedule[i+1]

            # If it's the last event, return the event end time
            if next_event == None:
                return event[1]

            # If the incoming event fits, return the machine event end time
            if event[1] >= min_start and next_event[0] <= min_start+duration:
                return event[1]
    
    def getEndTime(self):
        max_end_time = 0
        for machine in self.machines_schedule:
            schedule = self.machines_schedule[machine]

            if len(schedule) == 0:
                continue

            last_event = schedule[-1]
            if last_event[1] > max_end_time:
                max_end_time = last_event[1]

        return max_end_time

def read_input():
    times = np.genfromtxt("./times.csv", dtype=int, delimiter=",")
    machines = np.genfromtxt("./machines.csv", dtype=int, delimiter=",")
    return times, machines

def parse_input(times, machines):
    jobs = times.shape[0] 
    operations = []
    op_number = 1
    for x, y in np.ndindex(times.shape):
        operation = Operation(op_number, x % jobs + 1, machines[x, y], times[x, y])
        operations.append(operation)
        op_number+=1
    return operations

def main():
    smp = 5 # seeking memory pool
    cdc = 10 # counts of dimension to change
    spc = True # self-position considering
    mr = 0.30 # mixture ratio
    cats_num = 50 # number of cats
    iterations = 1000

    times,machines = read_input()
    operations = parse_input(times, machines)

    cat_swarm_optmization = CatSwarmOptmization(operations, cats_num, smp, spc, cdc, mr)

    cat_swarm_optmization.initializePopulation()
    
    for i in range(0, iterations):
        cat_swarm_optmization.mixtureStates()
        for cat in cat_swarm_optmization.swarm:
            cat_swarm_optmization.applyMode(cat)
    
    rank = cat_swarm_optmization.rank()
    best_cat = rank[0]

    print(f"Best cat fitness: {best_cat.fitness}")

if __name__ == '__main__':
    main()