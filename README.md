# Panoramic Image Stitching

Image registration has as its purpose to combine two or more images into a single image comprised of the input images which have been geometrically transformed in order to align them. This process of alignment is achieved by identifying control points in corresponding images and using these as guides to align one image to another. These control points can be identified by using features in the image such as edges that are able to be more uniquely matched between images. 

The final method comprises a full end to end pipeline that reads a set of given images from disk, creates a set of templates which relate image pairs when possible and utilizes the constructed templates to form a composite image using the original image set. 

### Prerequisites

This project uses jupyter notebooks and requires numpy, matplotlib, pandas, scikit-image and scipy.

### Running

Open the notebooks in jupyter notbook and run them.

## Authors

* **Paul Mobbs and Alex Martinez Paiz** - *Initial work* - [pmobbs](https://github.com/pmobbs)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* This project makes use of images provided by Dr. Sally Wood of Santa Clara University.