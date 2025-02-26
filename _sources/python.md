# Setting up Python

<!-- <img src="python_logo.png" alt="drawing" width="150" style="display: block; margin: auto; "/> -->

```{image} ./python_logo.png
:width: 150px
:align: center
```

If it is the first time you use Python, these are some useful commands to get started. I suggest to install `Anaconda` which is very convenient to manage Python packages. You can find the documentation [here](https://docs.anaconda.com/anaconda/install/) and the installations files [here](https://www.anaconda.com/download/success) for Windows, Mac and Linux. 

Once you installed Anaconda, the best thing to do is to use the same environment I used to run all codes. To do so, [download the file `CSS.yml`](https://github.com/lorenzodallamico/CSS/blob/main/CSS.yml) file. Go to the download directory, open the terminal and run the following command 

> **Note**: the codes are working on Linux but they are not tested on Mac and Windows. In case of trouble, consult the Anaconda website on how to proceed with the creation and activation of a conda environment.

```bash
conda env create -f CSS.yml
```
This creates the environment we will work with. To use this environment, run

```bash
conda activate CSS
```

> **Note**: at the time I wrote this, I did not re-run all codes that will be used during the course. Us such, there might be some missing packages, but we will see that as we proceed.

Now we are ready to get started!
