# NICER 
### Neural Image Correction and Enhancement Routine

This repository contains a PyTorch implementation of: 

**"NICER: Aesthetic Image Enhancement with Humans in the Loop" [ACHI2020]**
(link will follow when available)

by [M. Fischer](https://github.com/mr-Mojo), [K. Kobs](http://www.dmir.uni-wuerzburg.de/staff/kobs/) and [A. Hotho](http://www.dmir.uni-wuerzburg.de/staff/hotho/)


## Installation

To install and run this framework, it is recommended that you create a `conda` environment. For further information on managing conda environments, confer 
[the docs](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html). 
Afterwards, head over to [PyTorch](https://pytorch.org/get-started/locally) and install the appropriate PyTorch and `cuda`

Once PyTorch is installed, go ahead and clone this repository. Then install the required libraries:

`pip install opencv-python` \
`pip install -r requirements.txt`

If all is well, you should then be able to run `python main.py` to start up the GUI. Code tested on Ubuntu and Windows 10. 