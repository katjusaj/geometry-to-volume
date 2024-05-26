# geometry-to-volume

The goal of this seminar was to implement a tool for converting polygonal geometry to volumes and demonstrate its usability on a set of 3D models.

The developed tool uses a process of ray casting to convert the selected region of a model to a volumetric representation. Tool is able to load .obj files, output a raw volumetric data and a 3D visualization of it. Users can define a specific region of space for conversion through coordinate selection, adjust the resolution of the volume and specify material properties for conversion. 

## Instructions

python obj2vol.python

input resolution (64, 128, 256)

input coordinates for convertion (minimum and maximum in all 3 dimensions)

