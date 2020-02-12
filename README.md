# Welcome to University Of Chicago's Python For Analytics
This course introduces the python language of programming to students who are preparing to study analytics and data science

Please follow the instructions below to get your computer ready for this class.

_Note Mac users: Once software is downloaded, if you double click to launch it, you may get permission errors. Try to right click on the downloaded software, pick "open" and continue. (Apple is trying to protect you from accidentally starting malware/virus)_

## Install Python (anaconda distribution)
Please install Python 3.x from this website: https://www.anaconda.com/distribution/
(do not install 2.7, it will be discontinued soon.) 

Mac users:
Accept all default prompts

Windows users:
Accept all default prompts, **except** "Add Anaconda to my PATH envrionment variable." Make sure this is checked.

Anaconda's distribution of Python is widely used in the industry, particularly among data scientists. This distribution makes it easy to use many libraries and packages for data analysis, building models, visualization, etc.

#### Aditional steps:
Execute these statements at the terminal (Windows users should use Anaconda Prompt)
  - `jupyter contrib nbextension install --user`
  - `jupyter nbextensions_configurator enable --user`

Once installed, please start jupyter notebook and execute code provided below
1. Start `Anaconda Navigator` and click `Launch` on the panel labeled `Jupyter Notebook`
2. Create new notebook from the web interface
3. Execute this code:
```
%%timeit
sum(range(1_000_000))
```
4. Execute this code:
```
from psutil import virtual_memory, disk_usage, cpu_count, os

bytes_in_gb = 1024**3

print("Memory:\t",round(virtual_memory().total/bytes_in_gb,4), "Gigabytes")
print("Disk:\t",round(disk_usage(os.path.abspath(os.sep)).total/bytes_in_gb,4), "Gigabytes")
print("CPUs:\t", cpu_count())
```

## Install Git
Please intall Git, a version control sotware, from this website: https://git-scm.com/downloads (you are ok to use default settings)

Note that this is a command-line tool. Once installed, you may not see a new icon to click. We will install a Desktop client to remedy this.

Although we don't make heavy use of version control, you will be introduced to the concept. Installing Git also installs "Git Bash," and comand line environment which simulates Unix/Linux. We will do several exercises which will require this environment.

## Clone this repository [Optional on day 1]
1. Visit this web page: https://github.com/ravescovi/python_for_analytics
2. Click "Clone or download" and pick the "Download ZIP" option (unless you already have a GitHub account)

