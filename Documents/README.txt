----------------------------------------------------------------------------------------------------
Title: Air Quality Forecasting using LSTM-vanilla
Author: Delaram Nedaei 
Email: delaram.nedaei@gmail.com
Version: 0.1

----------------------------- Abstract -------------------------------------------------------------
Air quality forecasting is an important issue for life of every human and creatures in the world. Therefore,
an automation tool is essential for predicting the air quality. In this project, LSTM is used for predicting 
the air quality. LSTM is implemented in python 3.8 and Keras Library. Experimental results show that 
LSTM can be used for predicing the air quality but statisitical analysis and requirement engineering of the 
user expectation is vital for designing an automation tool for designing an accurate model.

-----------------------------Experimental Setup ----------------------------------------------------
OS: Windows 10 - 64 bit
Programming Language: Python 3.8.3 - 64bit 
IDE: Visual Code 1.45.1
Python Libraries: 
    - Tensorflow.Keras
    - Numpy 
    - sklearn
    - pandas

Directories of the project is organized as follows: 
Project
------- Dataset-
                Test-
                    test.csv
                Train-
                    psi_df_2016_2019.csv
------- Documents-
                README.txt
------- Programming-
                LSTM-
                    Univariate-
                            Main.py
                            Functions.py

-----------------------------Experimental Results ---------------------------------------------------
# number of time steps = 24
# number of features = 1
# number for epochs = 10

Column       MSE
national    70.13
south       44.11
north       50.88
east        47.85
central     54.71
west        52.31

----------------------------- Conclusion ------------------------------------------------------------
In this project an automation tool is implemented for predicing air quality based on the dataset.
For designing an accurate model, more information is requrired about the expected output of the project.  









