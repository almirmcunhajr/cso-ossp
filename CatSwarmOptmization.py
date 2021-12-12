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

    def __init__(self, operations,
        n: int, 
        smp: int,
        spc: bool,
        cdc: int
    ) -> None:
        self.position = operations
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
        
    def apply_mode(self):
        if self.sm:
            self.seekingMode()
        else:
            self.tracingMode
   
    def getFitness(self, position: list(int)) -> float:
        schedule = Schedule(position, self.operations)
        end_time = schedule.getEndTime()

        return math.exp(-end_time)

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
            op = self.operations[op_index]
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
    for x, y in np.ndindex(times.shape):
        operation = Operation(x % jobs + 1, machines[x, y], times[x, y])
        operations.append(operation)
    return operations

def main():
    times,machines = read_input()
    operations = parse_input(times, machines)
    best_fitness = None
    cats = []
    for i in range(0, cats_num):
        operations = copy.deepcopy(operations)
        random.shuffle(operations)
        cats.append(CatSwarmOptmization(operations))
    #iterações
    for i in range(0,1000):
        for cat in cats:
            sm = random.random() > mr
            cat.sm = sm
            new_fitness = cat.getFitness()
            if best_fitness == None or new_fitness < best_fitness:
                best_fitness = new_fitness
            
        for cat in cats:
            cat.apply_mode()

main()