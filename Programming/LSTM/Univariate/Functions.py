'''
Delaram Nedaei 
Description: 
Functions 

References: 
[1] https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/
'''
#======================== Functions============================
# library 
from numpy import array

# split a univariate sequence into samples [1]
def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)


def SetTrainingSize(NumberPercent,data):
    return int((NumberPercent/100)*len(data))