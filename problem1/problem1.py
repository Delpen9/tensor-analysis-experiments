# Standard Libraries
import os
import numpy as np

# Decomposition
from tensorly import tucker_to_tensor
from tensorly.decomposition import tucker

# Imports
from part1 import part1
from part2 import part2
from part3 import part3

if __name__ == '__main__':
    current_path = os.path.abspath(__file__)
    file_directory = os.path.abspath(os.path.join(current_path, '..', '..', 'output', 'Problem1'))

    _text_1, _text_2, _text_3, _text_4 = part1()
    _text_5 = part2()
    _text_6, _text_7 = part3()

    text_list = [_text_1, _text_2, _text_3, _text_4, _text_5, _text_6, _text_7]
    for i in range(len(text_list)):
        with open(fr'{file_directory}/problem1_{i + 1}.txt', 'w') as filewriter:
            filewriter.write(text_list[i])
