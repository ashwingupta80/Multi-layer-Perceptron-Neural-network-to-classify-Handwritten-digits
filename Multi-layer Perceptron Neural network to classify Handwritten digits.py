import pandas as pd
import os
import sys
import math
import numpy as np

# Hidden layer nodes
inp_lay=784
inp_lay1=748
hid_1= 512
hid_2=256
out_lay=10

# Initialise variables
btch_sz = 32
eph_no = 40
lrn_rt = 0.01

# Initialise weight of layers.
Wt_1=np.sqrt(1 / inp_lay)* np.random.randn(hid_1, inp_lay)
Wt_2=np.sqrt(1 / hid_1)* np.random.randn(hid_2, hid_1) 
Wt_3=np.sqrt(1 / hid_2)* np.random.randn(out_lay, hid_2) 


# initialise bias for the layers.
bias_1=np.sqrt(1 / inp_lay)* np.random.randn(hid_1,1)
bias_2=np.sqrt(1/ inp_lay1)* np.random.randn(hid_2,1) 
bias_3=np.sqrt(1 / hid_2)* np.random.randn(out_lay,1)

# Activation fn- Sigmoid
def sigfn(s):
    # Above 500 make all values to 500 and below -500 to 500
    s=np.minimum(500, np.maximum(s, -500))
    s=-s
    s = (1.0/(1.0 + np.exp(s)))
    return s

# Helper fn of forward propagation that calulates weighted avg and adds bias to it
def forward_helper(wt,bias, activations):
    # Multiply weights with activations.
    wt_avg2=np.matmul(wt, activations)
    # Add bias to weighted avg calculated above.
    wt_sum2=bias+wt_avg2
    return wt_sum2

# Activation fn- Softmax for o/p layer
def sftmax(sft):
    return np.exp(sft - np.max(sft)) / np.sum(np.exp(sft - np.max(sft)), axis=0)
    
def prop_frwd(train_fp, cnt_fwd):
    cnt_fwd+=1
    # Weights and activations for I/P and hidden layer 1.
    # Get weighted sum.
    w_sum_1=forward_helper(Wt_1,bias_1, train_fp)
    # Calc activations
    Act1 = sigfn(w_sum_1)
    
    # Weights and activations for hidden layer 1 and 2.
    w_sum_2=forward_helper(Wt_2,bias_2, Act1)
    Act2 = sigfn(w_sum_2)
    
    # Weights and activations for hidden layer 2 and O/P layer.
    w_sum_3=forward_helper(Wt_3,bias_3, Act2)
    
    # Get softmax activation for final output layer.
    Act3 = sftmax(w_sum_3)
    #print(cnt_fwd)

    return w_sum_1,Act1,w_sum_2,Act2,w_sum_3,Act3


def bk_prop_mul(matrix1, gradient1 ):
    trans2=matrix1.T
    mul_bk= np.matmul(trans2, gradient1)
    return mul_bk

def transl(gradient, matrix, o2):
    trans1=matrix.T
    mul_trans_div= np.matmul(gradient, trans1)*(1/o2) 
    return mul_trans_div

# Learning with one forward and backward pass.
# Write a forward pass for prediction.
def learning(lrn_y, lrn_x, Wt_1, Wt_2, Wt_3, bias_1, bias_2, bias_3):
    countl1=0
    cnt_fwd=0
    # Get the number of total images of digits.
    img_no = lrn_x.shape[1]
    #print("In Learn")
    countl1+=1

    #print(countl1)
    count_eph=0
    
    # Go through all ephocs for all the mini batches. Write mini batches to increase the accuracy and decrease ephocs.
    for q in range(eph_no):
        # Get a int array x with numbers from 0 to 59999 for the purpose of shuffling to increase accruacy.
        x=np.linspace(0, img_no-1, num=img_no, dtype=int)
        sub_array=[]
        
        # Mix the x array and use it to put random numbers in train and labels set for purposing of mixing.
        x_mixed=np.random.permutation(x)
        
        #print(x_mixed)
        count_eph+=1
        
        # Mix train and labels set accordingly.
        X1_mixed = lrn_x[:,x_mixed]
        minbatxy=[]
        #print(count_eph)
        y1_mixed = lrn_y[:,x_mixed]
        
        # Initialise arrays for train and label mini batches.
        minbatx=[]
        minbaty=[]
        
        # Calculate total no. of batches.
        tot_btch = img_no//btch_sz
        
        # Traverse through all batches and generate mini ones for better accuracy.
        for r in range(tot_btch):
            count_batch=0
            # Defines start of the img no or columns
            strt = btch_sz * r
            # Defines finish of the img no or columns.
            finish= btch_sz+ strt
            
            # Filter the mini batch columns according to the size of batch defined above.
            X_smallbatch = X1_mixed[:, strt : finish]
            # Filter the columns for labels
            Y_smallbatch = y1_mixed[:, strt : finish]
            
            # Append train data and labels in sepreate lists.
            minbatx.append(X_smallbatch)
            minbaty.append(Y_smallbatch)
            
            # Combine/ zip them together in a tuple for the ease of traversal through a loop.
            minbatxy=zip(minbatx,minbaty)
            count_batch+=1
            #print(count_batch)
            
        # Check if some columns less then batch size left and convert them in a small batch and append it also.

        quo1,rem1=divmod(img_no,btch_sz)
        floor_batches=img_no // btch_sz
        remaining_batches=floor_batches*btch_sz
        
        # If remainder is positive then add the remaining columns to the list for learning.
        if rem1> 0:
            X_smallbatch = X1_mixed[:, remaining_batches : img_no]
            Y_smallbatch = y1_mixed[:, remaining_batches : img_no]
            minbatx.append(X_smallbatch)
            minbaty.append(Y_smallbatch)
        
        # Insert remaining rows in the tuple set.    
        minbatxy=zip(minbatx,minbaty)
        
        countsmlbat=1
        # Traverse through all the small batches that we appended in list of tuples.
        for pair in minbatxy:
            countsmlbat+=1
            #print(pair)
            train, label = pair
            #print (train)
            #print (label)
            #print(countsmlbat)
            
            ## Do the forward propagation and get weighted sum and activation values after caluculation with activation function.
            
            w_sum_1,Act1,w_sum_2,Act2,w_sum_3,Act3 = prop_frwd(train, cnt_fwd) 
            
            ## Do backward prop using the values from forward prop.
            o2 = train.shape[1]
            
            #print(label.shape)
            
            # Subtact the error at the last layer for correction.
            delta_z3= np.subtract(Act3, label)
            
            # Transpose and multiply
            delta_w3= transl(delta_z3, Act2, o2)
            
            # Calculate slope of bias at final layer
            tt=np.sum(delta_z3, keepdims=True, axis=1)
            delta_b3= (1/o2)*tt
            back_prop_cnt=0
            # Go back final layer.
            delta_a2=bk_prop_mul(Wt_3, delta_z3)
            s1=sigfn(w_sum_2)
            s3=1 - s1
            
            # Calculate slope at second last layer. 
            delta_z2=s1 * s3 *delta_a2
            back_prop_cnt+=1
            # Transpose and multiply
            delta_w2=transl(delta_z2, Act1, o2)
            # Calculate slope of bias at second last layer
            gg=np.sum(delta_z2, keepdims=True, axis=1)
            delta_b2=(1/o2)*gg

            delta_a1=bk_prop_mul(Wt_2, delta_z2)
            s2=sigfn(w_sum_1)
            s4=1 - s2
            # Calculate slope at first layer. 
            delta_z1=s2 *s4*delta_a1
            
            #print(back_prop_cnt)
            delta_w1=transl(delta_z1, train, o2)
            hh=np.sum(delta_z1, keepdims=True, axis=1)
            delta_b1=(1/o2)*hh
            #print("Successfully done back prop and forward prop")
            #print("YAAY")
            
            # Get new slopes/ grad of the the weights and biases.
            # Check results with different learning rate and select the best one.
            updated_delta_w1= delta_w1*lrn_rt
            updated_delta_w2= delta_w2*lrn_rt
            updated_delta_w3= delta_w3*lrn_rt
            updated_delta_b1= delta_b1*lrn_rt
            updated_delta_b2= delta_b2*lrn_rt
            updated_delta_b3= delta_b3*lrn_rt

            # Correct the weights and bias for right predictions.
            Wt_1 -= updated_delta_w1
            Wt_2 -= updated_delta_w2
            Wt_3 -= updated_delta_w3
            bias_1 -= updated_delta_b1
            bias_2 -= updated_delta_b2
            bias_3 -= updated_delta_b3

# Print output in csv for comparison
def push_op(push1):
    push1.to_csv('test_predictions.csv', index=False, header=False)
                
#Get data 
x_learn_get = pd.read_csv(sys.argv[1], header=None)
# Transpose it 
learn_x = x_learn_get.transpose()
# Transform it in numpy array
learn_x = learn_x.to_numpy()


# Same for other two
# Get labels for learning the data
y_learn_get = pd.read_csv(sys.argv[2], header=None)
learn_y = y_learn_get.transpose()
learn_y = learn_y.to_numpy()

# Fetch the testing data. Transformations for it is done after learning phase.
x_testing_get = pd.read_csv(sys.argv[3], header=None)


## Transform labels for learning data to one hot encoded form with 10 classes(0-9)
total_classification=10
#np.max(learn_y)+1
#print(np.max(learn_y)+1)
lp=1
#print(num_classes)
result_bfrdrp=np.eye(total_classification)[learn_y]
#print(result_bfrdrp)
result_fnl = result_bfrdrp[0, :, :]
#print(result_fnl)
converted_y=result_fnl.transpose()


# Learning Phase starts here.
learning(converted_y, learn_x, Wt_1, Wt_2, Wt_3, bias_1, bias_2, bias_3)
#print("Learning done")
#print(learning)

# Transpose the test data and convert it to an numpy array.
testing_x = x_testing_get.transpose()
testing_x = testing_x.to_numpy()

# Now test the data 
w_sum_1,Act1,w_sum_2,Act2,w_sum_3,Act3 = prop_frwd(testing_x, cnt_fwd=0)

# Get the final activation's max value index 
get_prediction = Act3
#print(get_prediction)
### Prediction complete in the test data. 

#On the basis of max probablility pred gives the maximum index.
best_guess = np.argmax(get_prediction, axis=0)

#Create a dataframe from numpy array so that we can write the csv file
throw_out=pd.DataFrame(best_guess)


# write out the test set predictions
push_op(throw_out)