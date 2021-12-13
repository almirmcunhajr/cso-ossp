import math
import random
import numpy as np

class Cat:
    def __init__(self, position: 'list(int)', fitness: float):
        self.position = position
        self.fitness = fitness
        self.velocity = 1
        self.sm = True

class CatSwarmOptmization:
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
        self.solution_size = len(operations)
        self.n = n
        self.smp = smp
        self.spc = spc
        self.srd = random.randint(0, self.solution_size-1)
        self.cdc = cdc
        self.mr = mr
        self.swarm = []
        self.max_velocity = 1

    def mixtureStates(self):
        for cat in self.swarm:
            sm = random.random() > self.mr
            cat.sm = sm
            
    def initializePopulation(self):
        for i in range(self.n):
            position = [x for x in range(1, self.solution_size+1)]
            random.shuffle(position)
            fitness = self.getFitness(position)
        
            cat = Cat(position=position, fitness=fitness)
            self.swarm.append(cat)

    def seekingMode(self, position: 'list(int)') -> 'list(int)':
        candidates = []
        look_around = 0

        max_fitness = 0
        min_fitness = math.inf

        # If SPC is True, consider the current position as a candidate
        # Then look around for at last SMP positions
        if self.spc:
            fitness = self.getFitness(position)
            candidates = [(position, fitness)]
            look_around = self.smp-1
            max_fitness = min_fitness = fitness            
        else:
            look_around = self.smp

        # Look around, keeping the max and min fitness found
        for i in range(look_around):
            mutation = self.mutate(position)

            # Generate random SRD
            self.srd = random.randint(0, self.solution_size-1)

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

    def getCandidateProbabilityFunction(self, max_fitness: float, min_fitness):
        def probabilityFunction(candidate: 'tuple(list(int), float)') -> float:
            return abs(candidate[1]-max_fitness)/(max_fitness-min_fitness)
        
        return probabilityFunction       
    
    def mutate(self, position: 'list(int)') -> 'list(int)':
        borders = []
        borders.append(self.srd)
        borders.append((self.cdc + self.srd) % self.solution_size)
        borders.sort()

        mid = int((borders[1]-borders[0])/2)

        # Switch values between borders
        for i in range(mid+1):
            tmp = position[borders[0]+i]
            position[borders[0]+i] = position[borders[1]-i]
            position[borders[1]-i] = tmp

        return position
    
    def tracingMode(self, cat, best_cat):
        w = 0.7
        r = random.random()
        c = 2.05

        for pos in range(self.solution_size):
            velocity = w * cat.velocity + r * c * (best_cat.position[pos] - cat.position[pos])
            velocity = np.where(velocity > self.max_velocity, velocity, self.max_velocity)
            cat.position[pos] = (cat.position[pos] + int(velocity)) % self.solution_size
            if cat.position[pos] == 0:
                cat.position[pos] = 1
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
   
    def getFitness(self, position: 'list(int)') -> float:
        schedule = Schedule(position, self.operations)
        end_time = schedule.getEndTime()

        return end_time

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
    def __init__(self, sequence: 'list(int)', operations: 'list(Operation)'):
        self.sequence = sequence
        self.operations = operations
        self.jobs_next_free_time = {}
        self.machines_schedule = {}

        self.buildMachinesSchedule()

    def buildMachinesSchedule(self):
        for op_index in self.sequence:
            # Find job and machine free times
            op = [op for op in self.operations if op.number == op_index][0]
            
            job_next_free_time = 0
            if op.job in self.jobs_next_free_time:
                job_next_free_time = self.jobs_next_free_time[op.job]
                
            machine_next_free_time = self.findMachineAvailableTime(op.machine, job_next_free_time, op.time)

            # Reconcile op start time 
            op_start_time = max(job_next_free_time, machine_next_free_time)

            self.jobs_next_free_time[op.job] = op_start_time

            # Add op event to machine schedule
            if op.machine not in self.machines_schedule:
                self.machines_schedule[op.machine] = []

            self.machines_schedule[op.machine].append((op_start_time, op_start_time+op.time))
            self.machines_schedule[op.machine].sort(key=lambda event: event[0])

    def findMachineAvailableTime(self, machine:int, min_start: int, duration: int) -> int:
        # If nothing scheduled, return 0
        if machine not in self.machines_schedule:
            return 0

        schedule = self.machines_schedule[machine]

        for i, event in enumerate(schedule):
            # If it's the last event, return the event end time
            if i+1 == len(schedule):
                return event[1]

            next_event = schedule[i+1]

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
    
    best_cat_fitness = math.inf
    best_cat_fitness_history = []
    for i in range(0, iterations):
        cat_swarm_optmization.mixtureStates()
        for cat in cat_swarm_optmization.swarm:
            cat_swarm_optmization.applyMode(cat)
    
        rank = cat_swarm_optmization.rank()

        if rank[0].fitness < best_cat_fitness:
            best_cat_fitness = rank[0].fitness
            best_cat_fitness_history.append(best_cat_fitness)

        print(f"Best cat fitness of iteration {i}: {best_cat_fitness}")

if __name__ == '__main__':
    main()