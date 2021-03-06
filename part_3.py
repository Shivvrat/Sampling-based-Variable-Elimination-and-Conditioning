import sys
import warnings
from math import log10
import random
import time
import numpy
import get_data_uai
import get_evidence_data
import get_pr
import helper
import sampling_VE
import sampling_VE_adaptive_proposal_distribution

warnings.filterwarnings("ignore")
arguments = list(sys.argv)

try:
    # Please give the actual directory of the files
    uai_file_name = str(arguments[1])
    evidence_file_name = str(arguments[2])
    pr_file_name = str(arguments[3])
    type_of_algorithm = str(arguments[4])
    w_cutset = int(arguments[5])
    num_samples = int(arguments[6])
except:
    print("You have given less arguments")
    print("The code to run the algorithm :-")
    print(
        "python main.py <uai_file_directory_and_name> <evidence_file_directory_and_name> <pr_file_directory_and_name> <type_of_algorithm> <w_cutset_size> <number_of_samples>")
    print("Example :-")
    print("python main.py 1.uai 1.uai.evid 1.uai.PR -vec 3 50")
    print("python main.py 1.uai 1.uai.evid 1.uai.PR -avec 3 50")

def main():
    """
    This is the main function which is used to run all the algorithms
    :return:
    """
    log_actual_pr = log10(get_pr.get_pr(pr_file_name))
    if type_of_algorithm == "-vec":
        errors = []
        timer = []
        for each in range(10):
            random.seed(each)
            start_time_normal = time.time()
            num_of_var, cardinalities, num_of_cliques, num_of_var_in_clique, var_in_clique, distribution_array = get_data_uai.get_uai_data(
                uai_file_name)
            evidence = get_evidence_data.get_evidence(evidence_file_name)
            var_in_clique, distribution_array = helper.instantiate(num_of_var_in_clique, evidence, cardinalities,
                                                                   var_in_clique, distribution_array)
            estimate = sampling_VE.sampling_VE(num_of_var, cardinalities, num_of_cliques, num_of_var_in_clique,
                                               var_in_clique, distribution_array, w_cutset, num_samples)
            time_normal = (time.time() - start_time_normal)
            log_predicted_pr = log10(estimate)
            actual = (log_actual_pr - log_predicted_pr) / log_actual_pr
            errors.append(actual)
            timer.append(time_normal)
        mean_error = numpy.mean(errors)
        std_error = numpy.var(errors)
        mean_time = numpy.mean(timer)
        std_time = numpy.var(timer)
        print("The error is", mean_error, "±", numpy.sqrt(std_error))
        print("The time is", mean_time, "±", numpy.sqrt(std_time))

    elif type_of_algorithm == "-avec":
        errors = []
        timer = []
        for each in range(10):
            random.seed(each)
            start_time_normal = time.time()
            num_of_var, cardinalities, num_of_cliques, num_of_var_in_clique, var_in_clique, distribution_array = get_data_uai.get_uai_data(
                uai_file_name)
            evidence = get_evidence_data.get_evidence(evidence_file_name)
            var_in_clique, distribution_array = helper.instantiate(num_of_var_in_clique, evidence, cardinalities,
                                                                   var_in_clique, distribution_array)
            estimate_2 = sampling_VE_adaptive_proposal_distribution.sampling_VE(num_of_var, cardinalities,
                                                                                num_of_cliques, num_of_var_in_clique,
                                                                                var_in_clique, distribution_array,
                                                                                w_cutset, num_samples)
            time_normal = (time.time() - start_time_normal)
            log_predicted_pr = log10(estimate_2)
            actual = (log_actual_pr - log_predicted_pr) / log_actual_pr
            errors.append(actual)
            timer.append(time_normal)
        mean_error = numpy.mean(errors)
        std_error = numpy.var(errors)
        mean_time = numpy.mean(timer)
        std_time = numpy.var(timer)
        print("The error is", mean_error, "±", numpy.sqrt(std_error))
        print("The time is", mean_time, "±", numpy.sqrt(std_time))
    else:
        print("Please give correct algorithm name")


if __name__ == "__main__":
    main()
