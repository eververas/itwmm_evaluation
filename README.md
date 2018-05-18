# itwmm_evaluation
Evaluation of the itwmm benchmark datasets.

You will first need to install menpo and menpo 3d from [here](http://www.menpo.org/installation/development.html)

The actual steps for menpo and menpo3d installation are:

1. `conda create -n menpo_dev`

2. `source activate menpo_dev`

3. `conda install -c menpo menpoproject`

4. `conda remove --force menpo`

5. `conda remove --force menpo3d`

6. `mkdir path_to_menpo_code`
   `cd path_to_menpo_code`
   Download menpo and menpo3d code and put it here
   
7. `cd path_to_menpo_code/menpo`
   `pip install -e . --no-deps`
   
8. `cd path_to_menpo_code/menpo3d`
   `pip install -e . --no-deps`
   
   
After installation of the dependences are complete, use demo.ipynb to run the benchmark.
