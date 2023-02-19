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
        data_path = os.path.abspath(os.path.join(current_path, '..', '..', '..', 'output', 'Problem3', 'problem3.txt'))
        with open(data_path, 'r') as filedata:
            file_contents = filedata.read()
            doc.append(file_contents + '\n\n')

    # Part 2
    with doc.create(Section('Part 2')):
        doc.append('Below is the relative error for each threshold:\n')
        data_path = os.path.abspath(os.path.join(current_path, '..', '..', '..', 'output', 'Problem3', 'percentiles.csv'))
        with open(data_path, 'r') as filedata:
            file_contents = filedata.read()
            doc.append(file_contents + '\n\n')
        
        # Percentile = 0.1
        image_path = os.path.abspath(os.path.join(current_path, '..', '..', '..', 'output', 'Problem3', 'percentile_01_image_5.png'))
        with doc.create(Figure(position = 'h!')) as figure:
            figure.add_image(image_path, width = '120px')
            figure.add_caption('Percentile = 0.1; Image 5')

        image_path = os.path.abspath(os.path.join(current_path, '..', '..', '..', 'output', 'Problem3', 'percentile_01_image_10.png'))
        with doc.create(Figure(position = 'h!')) as figure:
            figure.add_image(image_path, width = '120px')
            figure.add_caption('Percentile = 0.1; Image 10')

        image_path = os.path.abspath(os.path.join(current_path, '..', '..', '..', 'output', 'Problem3', 'percentile_01_image_15.png'))
        with doc.create(Figure(position = 'h!')) as figure:
            figure.add_image(image_path, width = '120px')
            figure.add_caption('Percentile = 0.1; Image 15')

        # Percentile = 0.2
        image_path = os.path.abspath(os.path.join(current_path, '..', '..', '..', 'output', 'Problem3', 'percentile_02_image_5.png'))
        with doc.create(Figure(position = 'h!')) as figure:
            figure.add_image(image_path, width = '120px')
            figure.add_caption('Percentile = 0.2; Image 5')

        image_path = os.path.abspath(os.path.join(current_path, '..', '..', '..', 'output', 'Problem3', 'percentile_02_image_10.png'))
        with doc.create(Figure(position = 'h!')) as figure:
            figure.add_image(image_path, width = '120px')
            figure.add_caption('Percentile = 0.2; Image 10')

        image_path = os.path.abspath(os.path.join(current_path, '..', '..', '..', 'output', 'Problem3', 'percentile_02_image_15.png'))
        with doc.create(Figure(position = 'h!')) as figure:
            figure.add_image(image_path, width = '120px')
            figure.add_caption('Percentile = 0.2; Image 15')

        # Percentile = 0.3
        image_path = os.path.abspath(os.path.join(current_path, '..', '..', '..', 'output', 'Problem3', 'percentile_03_image_5.png'))
        with doc.create(Figure(position = 'h!')) as figure:
            figure.add_image(image_path, width = '120px')
            figure.add_caption('Percentile = 0.3; Image 5')

        image_path = os.path.abspath(os.path.join(current_path, '..', '..', '..', 'output', 'Problem3', 'percentile_03_image_10.png'))
        with doc.create(Figure(position = 'h!')) as figure:
            figure.add_image(image_path, width = '120px')
            figure.add_caption('Percentile = 0.3; Image 10')

        image_path = os.path.abspath(os.path.join(current_path, '..', '..', '..', 'output', 'Problem3', 'percentile_03_image_15.png'))
        with doc.create(Figure(position = 'h!')) as figure:
            figure.add_image(image_path, width = '120px')
            figure.add_caption('Percentile = 0.3; Image 15')

        # Percentile = 0.4
        image_path = os.path.abspath(os.path.join(current_path, '..', '..', '..', 'output', 'Problem3', 'percentile_04_image_5.png'))
        with doc.create(Figure(position = 'h!')) as figure:
            figure.add_image(image_path, width = '120px')
            figure.add_caption('Percentile = 0.4; Image 5')

        image_path = os.path.abspath(os.path.join(current_path, '..', '..', '..', 'output', 'Problem3', 'percentile_04_image_10.png'))
        with doc.create(Figure(position = 'h!')) as figure:
            figure.add_image(image_path, width = '120px')
            figure.add_caption('Percentile = 0.4; Image 10')

        image_path = os.path.abspath(os.path.join(current_path, '..', '..', '..', 'output', 'Problem3', 'percentile_04_image_15.png'))
        with doc.create(Figure(position = 'h!')) as figure:
            figure.add_image(image_path, width = '120px')
            figure.add_caption('Percentile = 0.4; Image 15')

        # Percentile = 0.5
        image_path = os.path.abspath(os.path.join(current_path, '..', '..', '..', 'output', 'Problem3', 'percentile_05_image_5.png'))
        with doc.create(Figure(position = 'h!')) as figure:
            figure.add_image(image_path, width = '120px')
            figure.add_caption('Percentile = 0.5; Image 5')

        image_path = os.path.abspath(os.path.join(current_path, '..', '..', '..', 'output', 'Problem3', 'percentile_05_image_10.png'))
        with doc.create(Figure(position = 'h!')) as figure:
            figure.add_image(image_path, width = '120px')
            figure.add_caption('Percentile = 0.5; Image 10')

        image_path = os.path.abspath(os.path.join(current_path, '..', '..', '..', 'output', 'Problem3', 'percentile_05_image_15.png'))
        with doc.create(Figure(position = 'h!')) as figure:
            figure.add_image(image_path, width = '120px')
            figure.add_caption('Percentile = 0.5; Image 15')

    doc.generate_pdf('problem3', clean_tex = False)