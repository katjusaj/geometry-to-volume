# geometry-to-volume

The goal of this seminar was to implement a tool for converting polygonal geometry to volumes and demonstrate its usability on a set of 3D models.

The developed tool uses a process of ray casting to convert the selected region of a model to a volumetric representation. Tool is able to load .obj files, output a raw volumetric data and a 3D visualization of it. Users can define a specific region of space for conversion through coordinate selection, adjust the resolution of the volume and specify material properties for conversion. 

## Instructions

install all imported libraries using terminal

Run program:
$ python obj2vol-raycast.py

input resolution (64, 128, 256)

input coordinates for conversion (minimum and maximum in all 3 dimensions)


## What is included

* main python script obj2vol.py

* example model with material and texture image

* results in RAW and HDF5 file formats

* PDF report







