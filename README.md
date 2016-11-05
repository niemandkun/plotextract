# plotextract

Extract plots from raster images

## Description

Researchers often face the problem that important data is stored in
an inappropriate format for their particular task. For example,
series of data may be stored in a raster image as a graph of some function,
while somebody needs a digital representation of that data.
Plotextract solves this problem.

Plotextract is a script written to extract data from images with plots
and save it as a sequence of points.

The script receives a raster image with plots as input,
as output it returns a `.csv` file with coordinates of points from each series,
represented on input image.


## Details

The script uses DBSCAN clustering to detect colors and total
amount of plots on image. Then using Tesseract OCR to recognize dash labels
on the figure axes it transforms detected plots into a right coordinate
system. Final step is to find pixels corresponding to each plot and to fit
approximating curves between those pixels.
Active contour models are used after fitting for better approximation.


## Installation

```
$ git clone https://github.com/niemandkun/plotextract.git
$ cd plotextract
$ python2 setup.py sdist
$ pip install dist/plotextract-0.0.tar.gz
```

## Example

### Input:
![alt tag](https://raw.githubusercontent.com/niemandkun/plotextract/master/samples/input.png)

### Output:
![alt tag](https://raw.githubusercontent.com/niemandkun/plotextract/master/samples/output.png)
