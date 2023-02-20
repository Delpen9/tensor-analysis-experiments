# Standard Libraries
import os
import numpy as np
from datetime import date

# LATEX
from pylatex import *
from pylatex.utils import *

def generate_header(
    doc : any,
    todays_date : str,
    company : str = 'Georgia Institute of Technology'
) -> any:
    '''
    '''
    header = PageStyle('header')

    with header.create(Head('L')):
        header.append('Page date: ')
        header.append(LineBreak())
        header.append(todays_date)
    
    with header.create(Head('C')):
        header.append(company)

    with header.create(Head('R')):
        header.append(simple_page_number())

    doc.preamble.append(header)
    doc.change_document_style('header')

    with doc.create(MiniPage(align = 'c')):
        doc.append(LargeText(bold('HW3: Problem 1')))
        doc.append(LineBreak())
        doc.append(MediumText(bold('Ian Dover')))

    return doc

if __name__ == '__main__':
    geometry_options = {'margin': '0.7in'}
    doc = Document(geometry_options = geometry_options)

    doc = generate_header(doc, date.today())

    current_path = os.path.abspath(__file__)

    # Part 1
    with doc.create(Section('Part 1')):
        data_path = os.path.abspath(os.path.join(current_path, '..', '..', '..', 'output', 'Problem1', 'problem1_1.txt'))
        with doc.create(Subsection('Subsection A')):
            with open(data_path, 'r') as filedata:
                file_contents = filedata.read()
                doc.append(file_contents + '\n\n')

        with doc.create(Subsection('Subsection B')):
            data_path = os.path.abspath(os.path.join(current_path, '..', '..', '..', 'output', 'Problem1', 'problem1_2.txt'))
            with open(data_path, 'r') as filedata:
                file_contents = filedata.read()
                doc.append(file_contents + '\n\n')

            data_path = os.path.abspath(os.path.join(current_path, '..', '..', '..', 'output', 'Problem1', 'problem1_3.txt'))
            with open(data_path, 'r') as filedata:
                file_contents = filedata.read()
                doc.append(file_contents + '\n\n')

            data_path = os.path.abspath(os.path.join(current_path, '..', '..', '..', 'output', 'Problem1', 'problem1_4.txt'))
            with open(data_path, 'r') as filedata:
                file_contents = filedata.read()
                doc.append(file_contents + '\n\n')

    # Part 2
    with doc.create(Section('Part 2')):
        data_path = os.path.abspath(os.path.join(current_path, '..', '..', '..', 'output', 'Problem1', 'problem1_5.txt'))
        with open(data_path, 'r') as filedata:
            file_contents = filedata.read()
            doc.append(file_contents + '\n\n')

    # Part 3
    with doc.create(Section('Part 3')):
        doc.append('''In CP decomposition, the tensor is decomposed into a series of rank-1 tensors whereas in Tucker decompositon,
        the tensor is decomposed into a core tensor and factor matrices. CP decomposition is capable of representing any non-negative tensor
        to any desired accuracy; however, Tucker decomposition is only capable of representing tensors that have a lower rank structure.
        Finally, CP decomposition is less computationally intensive than Tucker decomposition.''')
        data_path = os.path.abspath(os.path.join(current_path, '..', '..', '..', 'output', 'Problem1', 'problem1_6.txt'))
        with open(data_path, 'r') as filedata:
            file_contents = filedata.read()
            doc.append(file_contents + '\n\n')

        data_path = os.path.abspath(os.path.join(current_path, '..', '..', '..', 'output', 'Problem1', 'problem1_7.txt'))
        with open(data_path, 'r') as filedata:
            file_contents = filedata.read()
            doc.append(file_contents + '\n\n')

    doc.generate_pdf('problem1', clean_tex = False)