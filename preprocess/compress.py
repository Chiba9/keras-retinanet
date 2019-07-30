#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 14:41:30 2019

@author: Chiba
"""

from PIL import Image
import argparse
import os
import sys

def parse_args(args):
    parser = argparse.ArgumentParser(description='Compress image file to lower quality for pedestrain detection.')

    parser.add_argument('folder_in', help='The folder contains images to compress.')
    parser.add_argument('folder_out', help='Path to save the compressed images to.')
    parser.add_argument('--quality', help='The quality of compression.', type = int, default=50)

    return parser.parse_args(args)


def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    counter = 1
    total = len(os.listdir(args.folder_in))
    print('compressing', total, 'files in the raw folder.')
    for filename_in in os.listdir(args.folder_in):
        print('\r',counter, '/', total, end = '')
        image_in = Image.open(os.path.join(args.folder_in, filename_in))
        image_in.save(os.path.join(args.folder_out, filename_in), quality=args.quality)
        counter += 1


if __name__  == '__main__':
    main()