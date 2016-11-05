#!/usr/bin/env python2

from argparse import ArgumentParser

import matplotlib.pylab as plt
import numpy as np
import pandas as pd

from plotvision import img_to_data
from plotvision.load import *
from plotvision.backend import DEFAULT_CLUSTERING_THRESHOLD,\
                               DEFAULT_LABELING_THRESHOLD


DESCRIPTION = """Recognize line charts on images"""


def build_argparser():
    parser = ArgumentParser(description=DESCRIPTION)
    parser.add_argument('path', type=str, metavar='PATH',
                        help='path to image')
    parser.add_argument('-v', '--verbose', dest='verbose',
                        action='store_true', default=False,
                        help='show debug messages')
    parser.add_argument('-c', type=float, metavar='CTHR',
                        dest='clustering_threshold',
                        default=DEFAULT_CLUSTERING_THRESHOLD,
                        help='color clustering threshold')
    parser.add_argument('-l', type=float, metavar='LTHR',
                        dest='labeling_threshold',
                        default=DEFAULT_LABELING_THRESHOLD,
                        help='color labeling threshold')
    parser.add_argument('-f', dest='frame', default=0, type=int,
                        help='frame index for dicom image')
    parser.add_argument('-o', dest='output', default='output.csv', type=str,
                        help='output file (csv)')
    return parser


if __name__ == '__main__':

    loader = FileLoader(ImageLoader(), DicomLoader())
    parser = build_argparser()
    args = parser.parse_args()

    try:
        source = list(loader.load(args.path))[args.frame]
    except (IOError, IndexError, TypeError):
        parser.error('Error reading file: %s' % args.path)

    data = img_to_data(plot_image=source,
                       clustering_threshold=args.clustering_threshold,
                       labeling_threshold=args.labeling_threshold,
                       verbose=args.verbose)

    df = pd.DataFrame()

    for color, (x, y), (min_x, min_y) in data:

        df = df.append(pd.DataFrame([color]))
        df = df.append(pd.DataFrame(np.array([min_x, min_y]).reshape(1, -1)))
        df = df.append(pd.DataFrame(x.reshape(1, -1)))
        df = df.append(pd.DataFrame(y.reshape(1, -1)))

        plt.plot(x, y, c=color, linewidth=1)
        plt.scatter(min_x, min_y, color=color, s=4)

    plt.show()

    try:
        df.to_csv(args.output, index=False, header=False)
        print 'Saved to', args.output
    except IOError:
        print 'Error occured while saving to', args.output
