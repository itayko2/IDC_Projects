import numpy as np
import timeit

NUM_OF_VECTORS = 20000
DIM = 20
np.random.seed(42)

arr = np.random.rand(NUM_OF_VECTORS, DIM)
gram_matrix_1 = np.zeros(shape=(NUM_OF_VECTORS, NUM_OF_VECTORS))
gram_matrix_2 = np.zeros(shape=(NUM_OF_VECTORS, NUM_OF_VECTORS))

start = timeit.default_timer()


def kernel(x, arrx):
    return np.square(np.sum(x * arrx, axis=1) + 1)


for i in range(NUM_OF_VECTORS):
        gram_matrix_1[i] = kernel(arr[i], arr)

stop = timeit.default_timer()

print("Time for Gram Matrix using kernel: ", stop - start)


def mapping(x):
    result = np.array([1])
    for j in range(DIM):
        result = np.append(result, (np.sqrt(2) * x[j]))

    for k in range(DIM):
        result = np.append(result, np.square(x[k]))

    for n in range(DIM - 1):
        for m in range(n + 1, DIM):
            result = np.append(result, (np.sqrt(2) * (x[n] * x[m])))

    return result


def inner_product(x, arrx):
    return np.sum(x * arrx, axis=1)


start = timeit.default_timer()

arr = np.apply_along_axis(mapping, 1, arr)

for i in range(NUM_OF_VECTORS):
    gram_matrix_2[i] = inner_product(arr[i], arr)


stop = timeit.default_timer()

print(np.isclose(gram_matrix_1, gram_matrix_2))

print("Time for Gram Matrix using mapping: ", stop - start)
