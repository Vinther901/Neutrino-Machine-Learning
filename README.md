# Neutrino-Machine-Learning
Using Machine Learning on simulated IceCube data culminating in a joint [Bachelor thesis](./Bachelor_Project.pdf).

## Table of contents:
1. [ Data ](#data)
    * [ GraphCreator.py ](#GraphCreator)
    * [ GraphCreatorImproved.py ](#GraphCreatorImproved)
    * [ CreateTorchDataset.py ](#CreateTorchDataset)
    * [ MergeDataBases.py ](#MergeDataBases)
    * [ Mock_GraphCreator.ipynb ](#Mock_GraphCreator)
2. [ Model ](#Model)
    * [ Models (folder) ](#Models)
    * [ Trained_Models (folder) ](#Trained_Models)
    * [ Test_trained_models.ipynb ](#Test_trained_models)
3. [ Various ](#Various)
    * [ azimuth_normal_distribution.ipynb ](#azimuth_normal_distribution)
    * [ FunctionCollection.py ](#FunctionCollection)
    * [ Notes.ipynb ](#Notes)
    * [ Simulator ](#Simulator)
    * [ InvestigativeNotebook.ipynb ](#InvestigativeNotebook)

<a name="data"></a>
## 1. Data
This section covers the contents of the notebooks / python files which have anything to do with data manipulation and graph creation.

<a name="GraphCreator"></a>
### GraphCreator
Is a python file for creating graphs, given a set of event numbers.
However, the inverse transforming and so forth is not parallelized, so a new 'better' file is the GraphCreatorImproved.

<a name="GraphCreatorImproved"></a>
### GraphCreatorImproved
This python file loads a bunch of events based on a set of event numbers and returns a dataset consisting of graphs.

<a name="CreateTorchDataset"></a>
### CreateTorchDataset
Was the first endition of a graph creating script, and is not optimal.

<a name="MergeDataBases"></a>
### MergeDataBases
Converts a DB file to a csv file. It is not really employed anymore.

<a name="Mock_GraphCreator"></a>
### Mock_GraphCreator
This notebook investigates the effects of the different functions that can define edges in a graph.

<a name="Model"></a>
## 2. Model
This section contains the scripts / notebooks that have anything to do with model training and testing and so forth.

<a name="Models"></a>
### Models
This is a folder containing a number of Pytorch models which have been tried and tested.

<a name="Trained_Models"></a>
### Trained_Models
This folder contains the weights and optimizer status for different trained models.

<a name="Test_trained_models"></a>
### Test_trained_models
There are numerous python files and notebooks which investigates the performance of a model. This endition is, however, the latest. But I will try to create a newer and more general .py file or notebook (work in progress)..
