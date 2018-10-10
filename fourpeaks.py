import sys
import os
import time

import java.io.FileReader as FileReader
import java.io.File as File
import java.lang.String as String
import java.lang.StringBuffer as StringBuffer
import java.lang.Boolean as Boolean
import java.util.Random as Random

import dist.DiscreteDependencyTree as DiscreteDependencyTree
import dist.DiscreteUniformDistribution as DiscreteUniformDistribution
import dist.Distribution as Distribution
import opt.DiscreteChangeOneNeighbor as DiscreteChangeOneNeighbor
import opt.EvaluationFunction as EvaluationFunction
import opt.GenericHillClimbingProblem as GenericHillClimbingProblem
import opt.HillClimbingProblem as HillClimbingProblem
import opt.NeighborFunction as NeighborFunction
import opt.RandomizedHillClimbing as RandomizedHillClimbing
import opt.SimulatedAnnealing as SimulatedAnnealing
import opt.example.FourPeaksEvaluationFunction as FourPeaksEvaluationFunction
import opt.ga.CrossoverFunction as CrossoverFunction
import opt.ga.SingleCrossOver as SingleCrossOver
import opt.ga.DiscreteChangeOneMutation as DiscreteChangeOneMutation
import opt.ga.GenericGeneticAlgorithmProblem as GenericGeneticAlgorithmProblem
import opt.ga.GeneticAlgorithmProblem as GeneticAlgorithmProblem
import opt.ga.MutationFunction as MutationFunction
import opt.ga.StandardGeneticAlgorithm as StandardGeneticAlgorithm
import opt.ga.UniformCrossOver as UniformCrossOver
import opt.prob.GenericProbabilisticOptimizationProblem as GenericProbabilisticOptimizationProblem
import opt.prob.MIMIC as MIMIC
import opt.prob.ProbabilisticOptimizationProblem as ProbabilisticOptimizationProblem
import shared.FixedIterationTrainer as FixedIterationTrainer
import shared.ConvergenceTrainer as ConvergenceTrainer

from array import array


N=70
T=N/10
fill = [2] * N
ranges = array('i', fill)

ef = FourPeaksEvaluationFunction(T)
odd = DiscreteUniformDistribution(ranges)
nf = DiscreteChangeOneNeighbor(ranges)
mf = DiscreteChangeOneMutation(ranges)
cf = SingleCrossOver()
df = DiscreteDependencyTree(.1, ranges)
hcp = GenericHillClimbingProblem(ef, odd, nf)
gap = GenericGeneticAlgorithmProblem(ef, odd, mf, cf)
pop = GenericProbabilisticOptimizationProblem(ef, odd, df)


def tune_MIMIC():
        for samples in range(20,100,20):
                mimic = MIMIC(samples, 10, pop)
                fit = FixedIterationTrainer(mimic, 2000)
                fit.train()
                print "MIMIC_tune: " + str(ef.value(mimic.getOptimal()))

#tune_MIMIC()

def test_iter(algorithm, arguments, no_loops, no_iter):
        results = []
        for loop in range(no_loops):
                algo_init = algorithm(*arguments)
                fit = FixedIterationTrainer(algo_init, no_iter)
                fit.train()
                results += [ef.value(algo_init.getOptimal())]
        return results

def convergence():
        converge_output = ["Number of Iterations,Randomized Hill Climbing,Simulated Annealing,Genetic Algorithm,MIMIC"]
        for no_iter in range(500,3000,500):
                line = [str(no_iter)]
                algorithms_arguments = [(RandomizedHillClimbing, (hcp,)), 
                (SimulatedAnnealing, (1E11, .95, hcp)),
                (StandardGeneticAlgorithm, (200,100,10,gap)),
                (MIMIC, (100,10,pop))]
                for algos, args in algorithms_arguments:
                        algo_results = test_iter(algos, args, 10, no_iter)
                        print(algo_results)
                        avg_algo_results = sum(algo_results) / len(algo_results)*1.0
                        line += [str(avg_algo_results)]
                converge_output.append(",".join(line))
        return converge_output


converge_list = convergence()


with open('fourpeaks_converge.csv', 'wb') as file:
        for line in converge_list:
                file.write(line)
                file.write('\n')