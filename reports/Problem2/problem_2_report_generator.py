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
        doc.append(LargeText(bold('HW3: Problem 2')))
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
        image_path = os.path.abspath(os.path.join(current_path, '..', '..', '..', 'output', 'Problem2', 'part_1_vector_image_label_0.png'))
        with doc.create(Figure(position = 'h!')) as figure:
            figure.add_image(image_path, width = '120px')
            figure.add_caption('Vector Image of Label = 0')

        image_path = os.path.abspath(os.path.join(current_path, '..', '..', '..', 'output', 'Problem2', 'part_1_vector_image_label_1.png'))
        with doc.create(Figure(position = 'h!')) as figure:
            figure.add_image(image_path, width = '120px')
            figure.add_caption('Vector Image of Label = 1')

    doc.generate_pdf('problem2', clean_tex = False)