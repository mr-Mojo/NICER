# NICER 
### Neural Image Correction and Enhancement Routine

This repository contains a PyTorch implementation of: 

**"NICER: Aesthetic Image Enhancement with Humans in the Loop" [ACHI2020]**


by [M. Fischer](https://github.com/mr-Mojo), [K. Kobs](http://www.dmir.uni-wuerzburg.de/staff/kobs/) and [A. Hotho](http://www.dmir.uni-wuerzburg.de/staff/hotho/). 
The publication can be found at the [ThinkMind(TM) Digital Library](https://www.thinkmind.org/index.php?view=article&articleid=achi_2020_5_390_20186). 


## Installation

To install and run this framework, it is recommended that you create a `conda` environment. For further information on managing conda environments, confer 
[the docs](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html). 
Afterwards, head over to [PyTorch](https://pytorch.org/get-started/locally) and install the appropriate PyTorch and Cuda versions. 

Once PyTorch is installed, go ahead and clone this repository. Then install the required libraries:

`pip install opencv-python` \
`pip install -r requirements.txt`

Once everything is set up, you can simply run `python main.py` to load up the application GUI. 

## How does it work? 

NICER is a differentiable, neural image enhancement tool. The original, unedited version of the image is passed through our neural pipeline, which is depicted below. NICER first uses our devised ABN (Adaptive Brightness Normalization - for details, cf. the [paper](https://www.thinkmind.org/index.php?view=article&articleid=achi_2020_5_390_20186) algorithm as a pre-processing step, before feeding the image into the enhancement loop. In the loop, the image is iteratively updated by a Context Aggregation Network (CAN) has been trained to replicate the behaviour of image enhancement operations (e.g., contrast, brightness, saturation, ...). Subsequently, the enhanced image is assessed by a neural image assessor, NIMA, whose aesthetic score prediction is then maximized by backpropagating a loss function towards the CAN filter intensitites via Gradient Descent. 

<img src="https://github.com/mr-Mojo/NICER/blob/master/imgs/fullpipe.png" width="1050" height="750">
NIMA has the advantage that the user can interfere with the enhancement process at any time, be it before, during, or even after the optimization. Furthermore, the optimization happens in a white-box fashion, so users have control and can adjust parameters to their liking. 

## Bibtex 

If you find this work and code useful, please cite our publication in your work: 

```
@article{fischer2020nicer,
  title={NICER: Aesthetic Image Enhancement with Humans in the Loop},
  author={Fischer, Michael and Kobs, Konstantin and Hotho, Andreas},
  journal={arXiv preprint arXiv:2012.01778}, 
  year={2020}
}
```
