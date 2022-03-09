import PyNomad
import sys


def bb(x):
    """
    TODO
    :param x:
    :return:
    """
    try:
        f = x.get_coord(0)**2 + x.get_coord(1)**2 + x.get_coord(2)**2
        x.setBBO(str(f).encode("UTF-8"))
    except:
        print("Unexpected eval error", sys.exc_info()[0])
        return 0
    return 1 # 1: success 0: failed evaluation


# Initial point x0, lower bound (lb) and upper bound(ub)
x0 = [0.71, 0.51, int(10)]
lb = [-1, -1, int(-1)]
ub = [1, 1, int(10)]

# Formatting the parameters for PyNomad
input_type = "BB_INPUT_TYPE (R R I)"  # R=real (float) and I=integer
dimension = "DIMENSION 3"
max_nb_of_evaluations = "MAX_BB_EVAL 100"

params = [max_nb_of_evaluations, dimension, input_type,
          "DISPLAY_DEGREE 2", "BB_OUTPUT_TYPE OBJ", "DISPLAY_ALL_EVAL FALSE", "DISPLAY_STATS BBE OBJ (SOL)"]

# Important : PyNomad strictly minimizes the bb function
PyNomad.optimize(bb, x0, lb, ub, params)

