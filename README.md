# plotextract

Extract digital series data from raster image

## Description

Often people face with the problem that important data is stored in
an inappropriate format for their particular task. For example,
series of data may be stored in a raster image as a graph of some function,
althrough people need a digital representation of that data.
Plotextract solves this problem.

Plotextract is a script written to extract data from images with plots
and save it as a sequence of points.

As input script receives a raster image with plots,
as output it returns a `.csv` file with coordinates of points from each series,
represented on input image.


## Installation

```
$ git clone http://github.com/niemandkun/plotextract
$ cd plotextract
$ python2 setup.py sdist
$ pip install dist/plotextract-0.0.tar.gz
```

## Example

### Input:
![alt tag](https://raw.githubusercontent.com/niemandkun/plotextract/master/samples/input.png)

### Output:
![alt tag](https://raw.githubusercontent.com/niemandkun/plotextract/master/samples/output.png)
