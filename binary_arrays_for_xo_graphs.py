def store_decimal_value_in_binary_array(array, val, reverse=False):
    if reverse:
        start = 0
        stop = len(array)
        step = 1
    else:
        start = len(array) - 1
        stop = -1
        step = -1

    for i in range(start, stop, step):
        if val % 2 == 0:
            array[i] = 0
        else:
            array[i] = 1
        val //= 2

def merge_left_right(left_subarray,right_subarray, array):
    for i in range(len(left_subarray)):
        array[i] = left_subarray[i]
    for j in range(len(right_subarray)):
        array[-1 - j] = right_subarray[-1 - j]

def all_odd_length_binary_arrays_no_mirroring_gen(size):
    """
    size: odd length of array
    """
    halved_size = size // 2
    binary_array = [0]*size
    left_array = [0]*halved_size
    right_array = [0]*halved_size

    for i in range(2**halved_size):
        store_decimal_value_in_binary_array(left_array, i)
        for j in range(i + 1, 2**halved_size):
            store_decimal_value_in_binary_array(right_array, j, reverse=True)
            merge_left_right(left_array, right_array, binary_array)
            binary_array[halved_size] = 0
            yield binary_array
            binary_array[halved_size] = 1
            yield binary_array

def all_even_length_binary_arrays_no_mirroring_gen(size):
    """
    size: even length of array
    """
    halved_size = size // 2
    binary_array = [0]*size
    left_array = [0]*halved_size
    right_array = [0]*halved_size

    for i in range(2**halved_size):
        store_decimal_value_in_binary_array(left_array, i)
        for j in range(i + 1, 2**halved_size):
            store_decimal_value_in_binary_array(right_array, j, reverse=True)
            merge_left_right(left_array, right_array, binary_array)
            yield binary_array

def all_binary_arrays_no_mirroring_gen(size):
    if size % 2 == 0:
        return all_even_length_binary_arrays_no_mirroring_gen(size)
    else:
        return all_odd_length_binary_arrays_no_mirroring_gen(size)
