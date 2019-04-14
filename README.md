OpenSfM [![Build Status](https://travis-ci.org/mapillary/OpenSfM.svg?branch=master)](https://travis-ci.org/mapillary/OpenSfM)
=======

## Overview
OpenSfM is a Structure from Motion library written in Python. The library serves as a processing pipeline for reconstructing camera poses and 3D scenes from multiple images. It consists of basic modules for Structure from Motion (feature detection/matching, minimal solvers) with a focus on building a robust and scalable reconstruction pipeline. It also integrates external sensor (e.g. GPS, accelerometer) measurements for geographical alignment and robustness. A JavaScript viewer is provided to preview the models and debug the pipeline.

<p align="center">
  <img src="https://docs.opensfm.org/_images/berlin_viewer.jpg" />
</p>

Checkout this [blog post with more demos](http://blog.mapillary.com/update/2014/12/15/sfm-preview.html)


## Getting Started

* [Building the library][]
* [Running a reconstruction][]
* [Documentation][]


[Building the library]: https://docs.opensfm.org/building.html (OpenSfM building instructions)
[Running a reconstruction]: https://docs.opensfm.org/using.html (OpenSfM usage)
[Documentation]: https://docs.opensfm.org  (OpenSfM documentation)


## Additional Notes on Building on MacOS 
(added by cren 03/26/2019) 
Before starting, make sure the Python version installed on you MacOS is version 3 (version 2
does not work) but 3.6 or lower. python 3.7 will give you problems later on when running OIQ scripts. If you already 
have 3.7 installed, downgrade it first with the following:

    brew unlink python # ONLY if you have installed (with brew) another version of python 3
    brew install --ignore-dependencies https://raw.githubusercontent.com/Homebrew/homebrew-core/f2a764ef944b1080be64bd88dca9a1d80130c558/Formula/python.rb
    
    
Remember to always use pip3 to install the dependencies. In particular

    pip3 install opencv-python
    
(added by cren 04/11/2019) 
Our 'dev' branch of OpenSfM was branched off in July 2018. At the time, 'boost-python' was still used. We need to 
'brew install boost-python3' to use boost with python 3.6. However, I'm having difficulty getting it to install on my
MacBook. As a result I decided to pull in pybind support from official OpenSfM, which is now used to replace boost. 
If you find yourself needing to do this, you will:

    git remote add upstream https://github.com/mapillary/OpenSfM.git
    git fetch upstream
    
Then you will want to cherry-pick commits related to pybind one by one. These include all commits in pull request #365 -
https://github.com/mapillary/OpenSfM/pull/365/commits, and commits baab985, 1b92cb0, 4de86c8. Once you have done all of
these:

    git submodule update --init --recursive #important
    python setup.py build #make sure python links to python3
