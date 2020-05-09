#import libraries
import numpy
import scipy.special
from scipy import signal
from scipy.signal import find_peaks
import scipy.io as spio

#code update 08/05/2020

# Neural network class definition, credit: Dr B. Metcalfe
class n:
    # Init the network, this gets run whenever we make a new instance of this class
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set the number of nodes in each input, hidden and output layer
        self.i_nodes = input_nodes
        self.h_nodes = hidden_nodes
        self.o_nodes = output_nodes
        # Weight matrices, wih (input -> hidden) and who (hidden -> output)
        self.wih = numpy.random.normal(0.0, pow(self.h_nodes, -0.5), (self.h_nodes, self.i_nodes))
        self.who = numpy.random.normal(0.0, pow(self.o_nodes, -0.5), (self.o_nodes, self.h_nodes))

        # Set the learning rate
        self.lr = learning_rate

        # Set the activation function, the logistic sigmoid
        self.activation_function = lambda x: scipy.special.expit(x)


    # Train the network using back-propagation of errors
    def train(self, inputs_list, targets_list):
        # Convert inputs into 2D arrays
        inputs_array = numpy.array(inputs_list, ndmin=2).T
        targets_array = numpy.array(targets_list, ndmin=2).T
    
        # Calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs_array)
    
        # Calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # Calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
    
        # Calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)
        
        # Current error is (target - actual)
        output_errors = targets_array - final_outputs
        
        # Hidden layer errors are the output errors, split by the weights, recombined at hidden nodes
        hidden_errors = numpy.dot(self.who.T, output_errors)
    
        # Update the weights for the links between the hidden and output layers
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)),
        numpy.transpose(hidden_outputs))
        
        # Update the weights for the links between the input and hidden layers
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
        numpy.transpose(inputs_array))


    # Query the network
    def query(self, inputs_list):
        # Convert the inputs list into a 2D array
        inputs_array = numpy.array(inputs_list, ndmin=2).T
        
        # Calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs_array)
        
        # Calculate output from the hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # Calculate signals into final layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        
        # Calculate outputs from the final layer
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs


#Median absolute deviation filter to normalise the input signal
def MAD_filter(signal_data):
    #define the window size that the median will be taken within
    window_size=75
    #pad the input signal by window size
    padding_vector=numpy.zeros(window_size)
    padded_signal = numpy.concatenate((padding_vector, signal_data, padding_vector), axis=None)
    #define length
    length_of_window = len(signal_data)
    #initialise vector to store filtered signal    
    filter_signal=numpy.zeros(len(padded_signal))
    #loop for each data point in original signal
    for i in range(window_size, length_of_window-1):
        #define the data values in the window
        window = padded_signal[i-window_size:i+window_size]
        #find median of the window
        median= numpy.median(window)
        #apply the MAD filter to the current pixel (shift up/down by the median of the window)
        filter_signal[i]= numpy.absolute(window[window_size] - median)
    #trim the padding
    filter_signal = filter_signal[(window_size):(len(signal_data) + window_size)]
    #return the filetered signal
    return filter_signal

#Peak detector to find peaks above the 'max' threshold. Returns index of poistion of the start of the peaks
#defined as the point that the peak value falls below 0.5
def Peak_Detector(signal_data, max, window_size):
    #use find peaks function to find peak vector location- height defined as max
    vector_peak_locs, _ = find_peaks(signal_data, height=max)
    #initialise vector to store final output
    submission_Index = numpy.zeros(len(vector_peak_locs))
    #loop for number of peaks identified
    for i in range(0, len(vector_peak_locs)):
        #find the start of the peak:

        #initialise the height of the peak
        height_check=signal_data[vector_peak_locs[i]]

        #step 'left' along the peak until the value falls below threshold of 0.5
        counter = 0
        while height_check > 0.5:
            counter= counter+1
            height_check = signal_data[vector_peak_locs[i]-counter]

        #define the start index as the point that the peak value falls below 0.5
        submission_Index[i] = vector_peak_locs[i]-counter
    #return vector location of the estimated beginning of peaks
    return submission_Index

#Bandpass Butterworth filter to remove noise from the input signal
#lower cutoff freq=20 and upper cutoff freq=1700 (obtained from analysis of bode plot of the submission data)
def Butterworth_Filter(RecordingData):
    # define the lower cutoff freq
    lcf = 20
    # define the upper cutoff freq
    hcf = 1700
    # define sampling frequency as same as the input data
    sampling_freq = 25000
    # define the Nyquist frequency
    nyq_freq = sampling_freq/2
    #use signal library to find signal coefficients A and B
    B, A = signal.butter(4, [lcf/nyq_freq, hcf/nyq_freq], btype='band')
    #apply filter from library using the A and B coefficients
    filtered_signal = signal.filtfilt(B, A, RecordingData)
    
    return filtered_signal

#main function containing code run when module is initialised
def main():

    #define the window size in which peaks will be observed
    peak_window=30
    #define the scaling factor for normalisation of peaks
    scaling_factor=13

    #define the NN parameters:
    #def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate)
    network = n (31, 10, 4, 0.05)
    #redefine the number of output node for the NN
    output_n = 4

    #load the training data, Index and Class from the .mat file
    training_mat = spio.loadmat('training.mat', squeeze_me=True)
    training_signal_data = training_mat['d']
    training_Index = training_mat['Index']
    training_Class = training_mat['Class']

    #load the submission data from the .mat file
    submission_mat = spio.loadmat('submission.mat', squeeze_me=True)
    submission_signal_data = submission_mat['d']
    print('data loaded')
    
    #apply butterworth and then MAD filter to the submission signal data
    filtered_submission = Butterworth_Filter(submission_signal_data)
    filtered_submission = MAD_filter(filtered_submission)

    #apply peak detector to filtered submission data to get start point index of Peaks above 1.2
    submission_Index = Peak_Detector(filtered_submission, 1.2, peak_window)
    #print number of peaks detected
    print('number of peaks detected:')
    print(len(submission_Index))

    #apply butterworth and then MAD filter to the training signl data
    filtered_training = Butterworth_Filter(training_signal_data)
    filtered_training = MAD_filter(filtered_training)

    #prepare the submission data peaks into windows defined by window size (30)
    submission_matrix = numpy.zeros((len(submission_Index), peak_window+1)) #initialise matrix
    #loop for number of identified peaks
    for i in range(0,len(submission_Index)):
        #find the ith index as an integer
        index_point = int(submission_Index[i])
        #set zeroth bit to empty class , first bit to index value , final bits to vector of the data for that peak
        submission_matrix[i,:]=filtered_submission[index_point:index_point + peak_window + 1]
  
    #find max value in submission peaks
    submission_total_max=numpy.amax(submission_matrix)
    print('submission max:')
    print(submission_total_max)

    #prepare the training data peaks into windows defined by window size (30)
    training_matrix = numpy.zeros((len(training_Index), peak_window+1)) #initialise matrix
    #loop for number of training data peaks
    for i in range(0,len(training_Index)):
        #set zeroth bit to class, first bit to index value , final bits to vector of the data for that peak
        training_matrix[i,:]=filtered_training[training_Index[i]:training_Index[i] + peak_window + 1]


    #find and print the max value in training dataset
    training_total_max=numpy.amax(training_matrix)
    print('training max:')
    print(training_total_max)

    print('training...')

    # Train the neural network on each training sample
    for i in range(0,len(training_Index)):
        # Scale and shift the inputs from 0..255 to 0.01..1
        inputs = (numpy.asfarray(training_matrix[i,:])/ training_total_max * 0.99) + 0.01
        # Create the target output values (all 0.01, except the desired label which is 0.99)
        targets = numpy.zeros(output_n) + 0.01
        # All_values[0] is the target label for this record
        targets[int(training_Class[i]) - 1] = 0.99
        # Train the network. If it is correct
        network.train(inputs, targets)

    print('...training complete!')

    #estimate the effectiveness of the network based on manual classified peaks:

    #define the index and the class of the test data (manually identified within the submission data)
    test_Index=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
    test_Class=[3,1,1,4,4,2,3,1,3,3,4,3,1,4,1,2,3,1,2,3,1,1,3,3,2,4]
    #prepare the test data
    test_matrix = numpy.zeros((len(test_Index), peak_window+1)) #initialise matrix
    #loop for number of training data peaks
    for i in range(0,len(test_Index)):
        #set zeroth bit to class, first bit to index value , final bits to vector of the data for that peak
        test_matrix[i,:]=submission_matrix[test_Index[i],:]

    #query the network with the test data and obtain a scorecard

    # Scorecard list for how well the network performs, initially empty
    scorecard = []

    # Loop through all of the records in the test data set
    for i in range(0,len(test_Index)):
        # The correct label is contained in the test_Class vector
        correct_label = int(test_Class[i])
        # Scale and shift the inputs
        inputs = (numpy.asfarray(test_matrix[i,:]) / submission_total_max * 0.99) + 0.01
        # Query the network
        outputs = network.query(inputs)
        # The index of the highest value output corresponds to the label
        label = numpy.argmax(outputs)
        #shift the label to correspond to desire output
        label=label+1
        # Append either a 1 or a 0 to the scorecard list
        if (label == correct_label):
            scorecard.append(1)
        else:
            scorecard.append(0)
            pass
        pass

    # Calculate the score and print the Estimatd performance of the NN
    scorecard_array = numpy.asarray(scorecard)
    print("Estimated Network Performance = ", (scorecard_array.sum() / scorecard_array.size)*100, '%')

    #run the full submission data set through the NN to obtain classes
    print('running on submission data')
    # submission Class array to store classes
    submission_Class = []
    # Loop through all of the records in the test data set
    for i in range(0,len(submission_Index)):

        # Scale and shift the inputs
        inputs = (numpy.asfarray(submission_matrix[i,:]) / submission_total_max * 0.99) + 0.01
        # Query the network
        outputs = network.query(inputs)
        # The index of the highest value output corresponds to the label
        label = numpy.argmax(outputs)
        #shift the label to correspond to desire output
        label=label+1
        # Append the label onto the submissio_Class list
        submission_Class.append(label)


    #write the submission index and class vectors/list to .mat file
    final_matlab_file={}
    final_matlab_file['Index'] = submission_Index
    final_matlab_file['Class'] = submission_Class
    spio.savemat('11820',final_matlab_file)


    return

#run the main function
main()
