import PyNomad
import sys


def bb(x1, x2, x3):
    """
    "Black-box" function to minimize f(x1,x2,x3)=x1^2 + x2^2 + x3^2
    :param x1: continuous variable x1 \in [-1, 1]
    :param x2: continuous variable x2 \in [-1, 1]
    :param x3: integer variable x3 \in {1, 10}
    :return: f(x1,x2,x3)=x1^2 + x2^2 + x3^^
    """
    return x1 ** 2 + x2 ** 2 + x3 ** 2


def bb_pynomad(x):
    """
    "Black-box" function formatted to software PyNomad
    :param x: a vector of 3 component, x=(x1,x2,x3) where x1,x2 are continuous and x3 is an integer.
    :param bb: function to optimize
     NOTE :  x.get_coord(i) retrives the i-th component of the vector x
    :return: for an evaluation, 0 if PyNomad failed, 1 if PyNomad succeed
    """
    try:
        f = bb(x.get_coord(0), x.get_coord(1), x.get_coord(2))
        x.setBBO(str(f).encode("UTF-8"))
    except:
        print("Unexpected eval error", sys.exc_info()[0])
        return 0
    return 1  # 1: success 0: failed evaluation


###############################################
#  First instance x0 = [0.71, 0.51, int(10)]  #
###############################################

# Initial point x0, lower bound (lb) and upper bound(ub)
x0 = [0.71, 0.51, int(10)]
lb = [-1, -1, int(-1)]
ub = [1, 1, int(10)]

# Formatting the parameters for PyNomad
input_type = "BB_INPUT_TYPE (R R I)"  # R=real (float) and I=integer
dimension = "DIMENSION 3"
max_nb_of_evaluations = "MAX_BB_EVAL 5"

params = [max_nb_of_evaluations, dimension, input_type,
          "DISPLAY_DEGREE 2", "BB_OUTPUT_TYPE OBJ", "DISPLAY_ALL_EVAL FALSE", "DISPLAY_STATS BBE OBJ (SOL)"]

# Important : PyNomad strictly minimizes the bb function
PyNomad.optimize(bb_pynomad, x0, lb, ub, params)

###############################################
#  Second instance x0 = [0.71, 0.51, int(10)]  #
###############################################

# Initial point x0, lower bound (lb) and upper bound(ub)
x02 = [0.2, 0.2, int(1)]

# We must redefined new PyNomad parameters for the second run
params2 = [max_nb_of_evaluations, dimension, input_type,
           "DISPLAY_DEGREE 2", "BB_OUTPUT_TYPE OBJ", "DISPLAY_ALL_EVAL FALSE", "DISPLAY_STATS BBE OBJ (SOL)"]

PyNomad.optimize(bb_pynomad, x0, lb, ub, params2)
