# STEM-haadf-magnification-matching
Google Colab code for aligning HAADF/STEM images by automatically matching magnification and particle position using image registration.
# HAADF Image Magnification and Position Matching

This project provides a Google Colab workflow for aligning two HAADF/STEM microscopy images when one image is zoomed out relative to the other. The code automatically crops and zooms the second image so that its particle positions and magnification match the reference image as closely as possible.

## Purpose

The script is designed for microscopy image comparison, especially when two HAADF images show the same nanoparticle region but have different magnifications or slightly shifted fields of view.

## Features

- Uploads two microscopy images directly in Google Colab
- Uses the first image as the reference
- Automatically estimates the best magnification scaling
- Finds the best x/y crop location
- Ignores labels and scale bars during alignment
- Generates an overlay image for visual comparison
- Saves and downloads the final aligned image

## Method

The code performs particle-based image registration. It normalizes the image contrast, masks regions such as the HAADF label and scale bar, then searches for the scale factor and crop position that give the best match between the reference image and the zoomed-out image.

## Requirements

The code is intended to run in Google Colab. The main Python packages used are:

```python
numpy
matplotlib
Pillow
opencv-python
