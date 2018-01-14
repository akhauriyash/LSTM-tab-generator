# LSTM tablature generator
This mini project utilizes an LSTM network to generate guitar tablature from dataset collected from ultimate guitar.

##  Prerequisites:
  * Nvidia GPU
  * 16 GB RAM (Otherwise you will have to decrease the dataset size)
  * Python
  * TFLearn

## To run:
  Download the data set from [here](https://drive.google.com/drive/u/0/folders/0BxIbIVKS-qnNfmhuUDkyNUdIaHlBOHBuSG4yS215cGtKNkZ0NEtZWi1oYUVWOU8xT3VpUXM?usp=sharing)
  Run the arraysaver.py. Make sure you change the file name in the python code.
  This will generate 2 npy files, which are essentially your dataset.
  Configure the network.py to your specifications and run it to train the network.
    
##  Note:
  This was a weekend project. The data cleaning was not done by me completely as I found a data set 
  midway through the data collecting process. I have however uploaded a code which parses data from the
  website, though it works quite slow.
  
## TO DO:
  - [ ] Correct data collection code and make it legible
  - [ ] Work on a more complicated model
  - [ ] Build a parser for full functionality
