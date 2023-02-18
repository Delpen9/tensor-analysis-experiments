import numpy as np
import os
from datetime import date

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
                doc.append(file_contents + '\n')

        with doc.create(Subsection('Subsection B')):
            data_path = os.path.abspath(os.path.join(current_path, '..', '..', '..', 'output', 'Problem1', 'problem1_2.txt'))
            with open(data_path, 'r') as filedata:
                file_contents = filedata.read()
                doc.append(file_contents + '\n')

            data_path = os.path.abspath(os.path.join(current_path, '..', '..', '..', 'output', 'Problem1', 'problem1_3.txt'))
            with open(data_path, 'r') as filedata:
                file_contents = filedata.read()
                doc.append(file_contents + '\n')

            data_path = os.path.abspath(os.path.join(current_path, '..', '..', '..', 'output', 'Problem1', 'problem1_4.txt'))
            with open(data_path, 'r') as filedata:
                file_contents = filedata.read()
                doc.append(file_contents + '\n')

    doc.append(NewPage())

    doc.generate_pdf('problem1', clean_tex = False)