# NICER 
### Neural Image Correction and Enhancement Routine

NICER is an automatic image enhancement application that was created during the course of the master's thesis 
"Using Neural Networks as a Metric Towards Optimal Automated Image Enhancement". The research question 
the thesis aimed to answer was whether a neural network (Google's NIMA, to be specific) can be used as a metric to guide an image 
manipulation network towards perceptually pleasing results.  


## Installation

To install and run this framework, it is recommended that you create a `conda` environment. If you have anaconda installed, simply do so by typing
`conda create -n my_env`, where `my_env` is the name of the environment. To activate the environment, run 
`source activate my_env`.  
Afterwards, head over to https://pytorch.org/get-started/locally and install the appropriate PyTorch version (with `cuda`
support, if you have access to a GPU). In my case, this was: 

`conda install pytorch torchvision cudatoolkit=10.1 -c pytorch`

Once PyTorch is installed, go ahead and clone this repository. Then install the required libraries:

`pip install opencv-python` \
`pip install -r requirements.txt`


## How does it work? 

To automatically enhance images, NICER uses a combination of two neural networks. First, a Context Aggregation Network [[1]](https://arxiv.org/pdf/1511.07122.pdf)
in a similar fashion as in [[2]](http://openaccess.thecvf.com/content_ICCV_2017/papers/Chen_Fast_Image_Processing_ICCV_2017_paper.pdf) 
is used to to apply a set of image processing filters onto the original image. These filters include: 
* Saturation 
* Contrast 
* Brightness
* Shadows
* Highlights
* Exposure
* Local Laplacian Filtering [[3]](http://people.csail.mit.edu/hasinoff/pubs/ParisEtAl11-lapfilters-lowres.pdf)
* Non-Local Dehazing [[4]](http://openaccess.thecvf.com/content_cvpr_2016/papers/Berman_Non-Local_Image_Dehazing_CVPR_2016_paper.pdf)

The resulting enhancement is then passed through Google's aesthetic assessment network NIMA [[5]](https://arxiv.org/abs/1709.05424)
and assigned a beauty-score in the range [0,10], which again is used to calculate a loss. Via Stochastic Gradient Descent, 
the loss is backpropagated towards the combination of filter intensities that created the enhanced image. This process is 
repeated until converge, yielding the most beautiful version of the image - according to the NIMA classifier.  


<img src="https://github.com/mr-Mojo/NICER/blob/master/imgs/pipeline_full.png" width="700" height="400">

More readme to come soon. 
