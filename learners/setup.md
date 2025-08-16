---
title: Setup
---
## Setup
Please complete the setup at least a day in advance of the workshop. If you run into issues, contact the workshop organizers by email so you're ready to begin on time. The setup steps include: 

1. Setup workshop folder
2. Install Python 3.11.9
3. Setup virtual environment with required packages
4. Download the data

## 1. Setup workshop folder

Create a folder on your desktop called `dl_workshop` for storing the workshop data and required packages.

```shell
cd ~/Desktop
mkdir dl_workshop
cd dl_workshop
pwd 
```

```output
~/Desktop/dl_workshop
```

## 2. Installing Python

[Python][python] is a popular language for scientific computing and a frequent choice for machine learning.

Python version requirement: This workshop requires Python 3.11.9. Newer versions like 3.12 or 3.13 are not yet fully compatible with TensorFlow and may cause issues. Even Python 3.11.9 may have some edge cases, but it works well enough to be the default in Google Colab and is stable for the purposes of this workshop.

To install Python 3.11.9, go to the [official 3.11.9 downloads page](https://www.python.org/downloads/release/python-3119//). Choose the installer that matches your operating system (Windows, macOS, or Linux).

Please set up your Python environment at least a day in advance of the workshop. If you run into issues with installation, contact the workshop organizers by email so you're ready to begin on time.


### Determine which `python` command to use for downstream setup steps

Different systems and Python installations (e.g., Anaconda, Git Bash, system Python, Windows Store, etc.) may register different command names. This quick check helps identify which one points to Python 3.11.9 on your machine.

Run the following in your terminal ([Git Bash recommended for Windows users](https://git-scm.com/downloads), Anaconda Prompt, or macOS/Linux shell):

```shell
python --version
py --version
python3 --version
```

Use whichever one returns Python 3.11.9 for the rest of the setup steps.

Example output:

```output
$ python --version
Python 3.11.9

$ py --version
Python 3.13.2

$ python3 --version
Python was not found...
```
In this case, use python throughout the remainder of the instructions.

If none of the commands return Python 3.11.9:

- Download and install Python 3.11.9. On Windows, be sure to check "Add Python to PATH" during installation
- If you're on Windows using Anaconda Prompt, try using [Git Bash](https://git-scm.com/downloads) instead.  
- Then re-run the checks above in a new terminal window

If you're still stuck, ask the workshop organizers for help before proceeding.

## 3. Configure virtual environment

Open a terminal and run the following commands.

1. Create a [virtual environment](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#create-and-use-virtual-environments) called `venv` using the "venv" command:

```shell
# Use python3 or py instead if one of them points to 3.11.9.
python -m venv venv # 1st "venv" is commmand, 2nd venv is name of the virtual environment / folder
```

If you run the `ls` command from `~/Desktop/dl_workshop`, you should see a new `venv` folder inside it

```shell
ls
```

```output
venv/
```

2. Activate the newly created virtual environment:

::: spoiler

### On Linux/macOs

```shell
source venv/bin/activate
```

:::

::: spoiler

### On Windows

```shell
venv\Scripts\activate
```

If you're using Git Bash on Windows, you need to add the source command first.

```shell
source venv/Scripts/activate
```
:::

**Note**: Remember that you need to activate your environment every time you restart your terminal, and before you launch Jupyter Lab!

3. After activating the enviornment, upgrade pip. This is a good practice to follow when you first initialize your virtual environment (beforing installing additional packages). [Pip](https://pip.pypa.io/en/stable/) is the package management system built into Python.Pip should be available in your system once you installed Python successfully.

```shell
 # remember: use python3 or py instead if it points to 3.11.9
python -m pip install --upgrade pip
```

4. Install the required packages:

Follow the OS-specific instructions below. NOte that It may take 10-20 minutes to install everything.

::: spoiler

### On Linux/macOs

```shell
 # Use python or py instead if one of them points to 3.11.9.
python3 -m pip install jupyter seaborn scikit-learn pandas tensorflow pydot
```

Note for MacOS users: there is a package `tensorflow-metal` which accelerates the training of machine learning models with TensorFlow on a recent Mac with a Silicon chip (M1/M2/M3).
However, the installation is currently broken in the most recent version (as of January 2025), see the [developer forum](https://developer.apple.com/forums/thread/772147).

:::

::: spoiler

### On Windows

```shell
# Use py or python3 instead if one of them points to 3.11.9.
python -m pip install jupyter seaborn scikit-learn pandas tensorflow pydot 
```

:::

Note: Tensorflow makes Keras available as a module too.

An [optional challenge in episode 2](episodes/2-keras.md) requires installation of Graphviz
and instructions for doing that can be found
[by following this link](https://graphviz.org/download/).

### Starting Jupyter Lab

We will teach using Python in [Jupyter Lab][jupyter], a programming environment that runs in a web browser.
Jupyter Lab is compatible with Firefox, Chrome, Safari and Chromium-based browsers.
Note that Internet Explorer and Edge are *not* supported.
See the [Jupyter Lab documentation](https://jupyterlab.readthedocs.io/en/latest/getting_started/accessibility.html#compatibility-with-browsers-and-assistive-technology) for an up-to-date list of supported browsers.

To start Jupyter Lab, open a terminal (Mac/Linux) or Command Prompt (Windows), 
**make sure that you activated the virtual environment you created for this course**,
and type the command:

```shell
jupyter lab
```

### Check your virtual software setup
To check whether all packages installed correctly, start a jupyter notebook in jupyter lab as
explained above (**with virtual environment activated**). Run the following check to verify you have the right version of Python configured.
```python
!python --version
```

If you don't see 3.11.9, make sure your virtual environment was activated prior to launching Jupyter Lab.
```output
Python 3.11.9 
```

Then, run the following lines of code:
```python
import tensorflow
print('Tensorflow version: ', tensorflow.__version__) # >= 2.12.0

import sklearn
print('sklearn version: ', sklearn.__version__) # >= 1.2.2

import seaborn
print('seaborn version: ', seaborn.__version__) # any version
 
import pandas
print('pandas version: ', pandas.__version__) # any version 
```

This should output the versions of all required packages without giving errors.
Most versions will work fine with this lesson, but:

- For Keras and Tensorflow, the minimum version is 2.12.0
- For sklearn, the minimum version is 1.2.2

### Fallback option: cloud environment
If a local installation does not work for you, it is also possible to run this lesson in [Binder Hub](https://mybinder.org/v2/gh/carpentries-incubator/deep-learning-intro/scaffolds). This should give you an environment with all the required software and data to run this lesson, nothing which is saved will be stored, please copy any files you want to keep. Note that if you are the first person to launch this in the last few days it can take several minutes to startup. The second person who loads it should find it loads in under a minute. Instructors who intend to use this option should start it themselves shortly before the workshop begins.

Alternatively you can use [Google colab](https://colab.research.google.com/). If you open a jupyter notebook here, the required packages are already pre-installed. Note that google colab uses jupyter notebook instead of Jupyter Lab.

## 4. Downloading the required datasets

Download the [weather dataset prediction csv][weatherdata] and [Dollar street dataset (4 files in total)][dollar-street]. Create a subfolder in your workshop folder called data, `~/Desktop/dl_workshop/data`, and move all 5 files to the data subfolder:

- `dl_workshop/data/weather_prediction_dataset_light.csv`
- `dl_workshop/data/train_labels.npy`
- `dl_workshop/data/test_labels.npy`
- `dl_workshop/data/train_images.npy`
- `dl_workshop/data/test_images.npy`

**Note**: If you end up using Google Colab for the workshop, you'll want to create a folder called `dl_workshop` within your Google drive, and then create the same `data` subfolder as above with all 5 files uploaded. **Avoid uploading the `venv` folder to your google drive**; you'll be using Colab's pre-built environment instead, and the `venv` folder contains MANY files from the libraries you installed. 

You can access files from Google drive using Colab and the following code:

```python
from google.colab import drive
drive.mount('/content/drive')
```

A prompt will appear asking you to authorize access to your Google Drive. After authorization, your Drive will be accessible under `/content/drive/My Drive/`. You can use standard Python I/O or libraries like pandas, os, glob, etc. to interact with files. Example below:

```python
import pandas as pd
# Load a CSV file from Drive
df = pd.read_csv('/content/drive/My Drive/dl_workshop/data/weather_prediction_dataset_light.csv')
```

You can use the file navigator (folder icon) within Colab to help navigate your Google drive files.

[dollar-street]: https://zenodo.org/api/records/10970014/files-archive
[jupyter]: http://jupyter.org/
[jupyter-install]: http://jupyter.readthedocs.io/en/latest/install.html#optional-for-experienced-python-developers-installing-jupyter-with-pip
[python]: https://python.org
[weatherdata]: https://zenodo.org/record/5071376/files/weather_prediction_dataset_light.csv?download=1
