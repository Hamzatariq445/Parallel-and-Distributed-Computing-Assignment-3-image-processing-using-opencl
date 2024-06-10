# Parallel-and-Distributed-Computing-Assignment-3-image-processing-using-opencl

Optimal Image preprocessing on “The ISIC 2020 Challenge Dataset” for Skin Lesion

Analysis Towards Melanoma(Cancer) Detection using OpenCL

You are provided with a dataset of colored images of skin lesions in various JPEG format.
The task is to convert these colored images to grayscale images using OpenCL parallel
computing.
Greyscale conversion involves transforming each pixel of the image from its RGB (Red,
Green, Blue) representation to a single intensity value representing the luminance.
Instructions:
The detailed requirements are mentioned in the sections provided below:
Setup Environment:
Install necessary OpenCL SDK and drivers on your machine. Installation instructions
and tutorial was already provided to you on GCR. Ensure that you have access to a GPU
that supports OpenCL for optimal performance.
Dataset Preparation:
Download the mentioned dataset of colored images of skin lesions. Familiarise
yourself with the dataset structure and image formats. For simplification only the JPEG
format is selected.
OpenCL Implementation:
Write OpenCL kernels to perform the conversion of colored images to grayscale
images. Ensure that your kernels can handle images of varying sizes. Implement efficient
memory allocation and data transfer strategies for optimal performance.
Image Processing:
Load each colored image from the dataset into memory. Apply the OpenCL kernel to
convert the colored image to greyscale. Save the resulting greyscale images to the disk.
Documentation:
Prepare a brief report documenting your implementation approach, including:
● Description of host program, including configuration steps and strategy to handle
complete dataset.
● Description of the greyscale conversion algorithms used and the thresholds selected
for greyscale value assignment. Along with the explanation of the OpenCL kernel
design.
● Also, Include five (5) random images from the original dataset of colored images
along with the generated grayscale images.
