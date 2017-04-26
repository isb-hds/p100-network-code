# p100-network-code

This code is generated for exploring the results of the Pioneer 100 project.  It contains 5 ipython notebooks and a small library of functions for generating a correlation network from the data collected over the course of the P100 project and detailed in [add publication].

This code is meant to be run in a Jupyter notebook that has the `scipy stack` installed. We recommend using the docker image created by the Jupyter group at https://github.com/jupyter/docker-stacks/tree/master/datascience-notebook.

## Data

The data is available from dbGap, https://www.ncbi.nlm.nih.gov/projects/gapprev/gap/cgi-bin/preview1.cgi?GAP_phs_code=ECvAYTPhZnTzMyUY. It is a tar.gz file and should be extracted to the same directory containing the code.

## Recommended docker image

This image can be downloaded from docker hub using

```
docker pull jupyter/datascience-notebook
```
on a machine with docker installed. This is not required, but is recommended and all instructions will be based on the use of this image.

## Running the Notebooks

An example shell script is provided [startup-notebook.sh](startup-notebook.sh). The startup command is basically
```
docker run -d -p [SOME LOCAL PORT]:8888 -e USE_HTTPS=yes -e GRANT_SUDO=yes -v [LOCAL PATH TO p100-network-code]:[ROOT PATH OF NOTEBOOKS ON JUPYTER]/p100-network-code -v [LOCAL PATH TO UNZIPPED data]:[ROOT PATH OF NOTEBOOKS ON JUPYTER]/data jupyter/datascience-notebook
```

Then, navigate in your browser to https://[your url]:[SOME LOCAL PORT].  For example, if you ran this on your localhost, with SOME LOCAL PORT = 8888, then you would navigate to https://localhost:8888.

Note, it will give you a warning about an invalid certificate, just click okay.  The default password for datascience-notebook is empty, i.e. just hit ENTER.

## The Notebooks

  * [Generate correlation network.ipynb](Generate correlation network.ipynb) - Generates a correlation network of all data for the p100 project
  * [Community Generation-DELTA.ipynb](Community Generation-DELTA.ipynb) - Generates a correlation network for change in data measurements for the p100 project
  * [Community Generation.ipynb](Community Generation.ipynb) - Performs community analysis using the intraomic correlation network
  * [Community Generation-DELTA.ipynb](Community Generation-DELTA.ipynb) - Performs community analysis using the intraomic delta(change) correlation network
  * [GEE longitudinal clinical changes.ipynb](GEE longitudinal clinical changes.ipynb) - Performs a GEE(generalized estimating equation) analysis to demonstrate change over the course of the study in clinically relevant biomarkers
  
