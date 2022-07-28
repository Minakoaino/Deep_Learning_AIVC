#Name:Asimina Tzana
#AM:aivc21015
#E-mail:aivc21015@uniwa.gr
#Artificial Intelligence and Visual Computing

from tensorflow import keras
from random import seed
from random import randint
from numpy import array
from math import ceil
from math import log10
from numpy import argmax
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers import RepeatVector
from keras.utils import np_utils
import statistics
import math

# generate lists of random integers and their sum
# n_examples for number of pairs
# n_numbers pairs (2)
# [ lowest, largest ] 
def random_sum_pairs(n_examples, n_numbers, lowest, largest):
    X, y = list(), list()
    for i in range(n_examples):
        in_pattern = [randint(lowest, largest) for _ in range(n_numbers)]
        out_pattern = sum(in_pattern)
        X.append(in_pattern)
        y.append(out_pattern)
    return X, y


# convert data to strings
def to_string(X, y, n_numbers, largest):
    max_length = n_numbers * ceil(log10(largest + 1)) + n_numbers - 1
    Xstr = list()
    counter = 0;
    for pattern in X:
        if randint(lowest, largest) % 2 == 1:
            strp = '-'.join([str(n) for n in pattern])
            strp = ''.join([' ' for _ in range(max_length - len(strp))]) + strp
            Xstr.append(strp)
            y[counter] = y[counter] - 2*pattern[1]
        else:
            strp = '+'.join([str(n) for n in pattern])
            strp = ''.join([' ' for _ in range(max_length - len(strp))]) + strp
            Xstr.append(strp)
        counter = counter+1
    max_length = ceil(log10(n_numbers * (largest + 1)))
    ystr = list()
    for pattern in y:
        strp = str(pattern)
        strp = ''.join([' ' for _ in range(max_length - len(strp))]) + strp
        ystr.append(strp)
    return Xstr, ystr


# integer encode strings
def integer_encode(X, y, alphabet):
    char_to_int = dict((c, i) for i, c in enumerate(alphabet))
    Xenc = list()
    for pattern in X:
        integer_encoded = [char_to_int[char] for char in pattern]
        Xenc.append(integer_encoded)
    yenc = list()
    for pattern in y:
        integer_encoded = [char_to_int[char] for char in pattern]
        yenc.append(integer_encoded)
    return Xenc, yenc


# one hot encode
def one_hot_encode(X, y, max_int):
    Xenc = list()
    for seq in X:
        pattern = list()
        for index in seq:
            vector = [0 for _ in range(max_int)]
            vector[index] = 1
            pattern.append(vector)
        Xenc.append(pattern)
    yenc = list()
    for seq in y:
        pattern = list()
        for index in seq:
            vector = [0 for _ in range(max_int)]
            vector[index] = 1
            pattern.append(vector)
        yenc.append(pattern)
    return Xenc, yenc


# generate an encoded dataset
def generate_data(n_samples, n_numbers, lowest, largest, alphabet):
    # generate pairs
    X, y = random_sum_pairs(n_samples, n_numbers, lowest,  largest)
    
    # convert to strings
    X, y = to_string(X, y, n_numbers, largest)
    
    # integer encode
    X, y = integer_encode(X, y, alphabet)
   
    # one hot encode
    X, y = one_hot_encode(X, y, len(alphabet))
    # return as numpy arrays
    X, y = array(X), array(y)
    return X, y
#converts input data to appropriate form for using in machine learning process
def convert_data(alphabet, X):
    
    # integer encode
    X, y = integer_encode(X, ["0"], alphabet)
    # one hot encode
    X, y = one_hot_encode(X, y, len(alphabet))
    # return as numpy arrays
    X, y = array(X), array(y)
    return X



# invert encoding
def invert(seq, alphabet):
    int_to_char = dict((i, c) for i, c in enumerate(alphabet))
    strings = list()
    for pattern in seq:
        string = int_to_char[argmax(pattern)]
        strings.append(string)
    return ''.join(strings)

#return average prediction error
def stats(l):
    sum = 0.0
    for i in range(0,len(l)):
        sum=sum+l[i]
    avg = sum/len(l)
    return avg
        
seed(1)
print("Configuring the dataset")
#test dataset length
n_samples = 1000
n_numbers = 2
lowest = 1
largest = 1000
alphabet = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '-', ' ']
n_chars = len(alphabet)
print("Train set items:", end=":")
print(n_samples)
print("Operands:", end=":")
print(n_numbers)
print("Numbers range: ["+str(lowest)+","+str(largest)+"]")
n_in_seq_length = n_numbers * ceil(log10(largest + 1)) + n_numbers - 1
n_out_seq_length = ceil(log10(n_numbers * (largest + 1)))
print("Input Length:"+str(n_in_seq_length))
print("Output Length:"+str(n_out_seq_length))
n_batch = 10
model = keras.models.load_model("outputData\\ModelLSTM.h5")

# evaluate on some new patterns
X, y = generate_data(n_samples, n_numbers, lowest, largest, alphabet)

result = model.predict(X, batch_size=n_batch, verbose=0)
# calculate error
expected = [invert(x, alphabet) for x in y]
predicted = [invert(x, alphabet) for x in result]

correct = 0
error = 0
correct1 = 0
error1 = 0
div=[]
for i in range(n_samples):
    # replace -- with - if needed
    div.append(abs(int(expected[i].replace("--","-")) - int(predicted[i].replace("--","-"))))
    #computing absolutely correct predictions
    if expected[i] == predicted[i]:
        correct = correct + 1
    else:
        error = error + 1
    #computing relatevely correct predictions
    if abs(int(expected[i]) - int(predicted[i]))<=0.05*largest:
        correct1 = correct1 + 1
    else:
        error1 = error1 + 1
print("Correct:"+str(correct)+" Error:"+str(error)  )
print("(~0.05XLargest) Correct:"+str(correct1)+" Error:"+str(error1)  )
a = stats(div)
print("Average Error:"+str(a))

#creates the input to machine learning string by the user's input
def makeString(n,m, largest,op):
    digitsn = int(math.log10(n))+1
    digitsm = int(math.log10(n))+1
    allDigits = digitsn+digitsm+1
    maxdigits = n_in_seq_length
    resultStr = str(n)+op+str(m)
    print(resultStr)
    if (allDigits<maxdigits):
        for i in range(0,maxdigits-allDigits):
            resultStr=" "+resultStr
    return resultStr

print("Type input string")
inputData = input("->");
while inputData != "exit":
    #if + given
    if inputData.find("+")>=0:
        elements = inputData.split("+",1)
        try:
            #split input to take operands
            a = int(elements[0].strip())
            b = int(elements[1].strip())
            #check if operands are in the numbers range
            if a>=largest or a<lowest:
                print("First operand must be in range ["+str(lowest)+","+str(largest-1)+"]")
            if a>largest or a<lowest:
                print("Second operand must be in range ["+str(lowest)+","+str(largest-1)+"]")
            #computing expected result    
            expected_result = a + b
            #making model input
            modelInput = makeString(a, b, largest,"+")
            inputList = [modelInput]
            Xin = convert_data(alphabet, inputList)
            #make prediction depending on model
            p_result = model.predict(Xin ,batch_size=n_batch, verbose=0)
            predicted_result = [invert(x, alphabet) for x in p_result]
            #printg result
            print("Predicted:",end=" ")
            print(predicted_result[0])
            print("Expected:",end=" ")
            print(expected_result)
        except:
            #something went wrong
            print("Error!!!")
    #if + given
    elif inputData.find("-")>=0:
        elements = inputData.split("-",1)
        try:
            a = int(elements[0].strip())
            b = int(elements[1].strip())
            
            if a>=largest or a<lowest:
                print("First operand must be in range ["+str(lowest)+","+str(largest-1)+"]")
            if a>largest or a<lowest:
                print("Second operand must be in range ["+str(lowest)+","+str(largest-1)+"]")
                  
            expected_result = a - b
            modelInput = makeString(a, b, largest,"-")
            print(modelInput)
            inputList = [modelInput]
            Xin = convert_data(alphabet, inputList)
            p_result = model.predict(Xin ,batch_size=n_batch, verbose=0)
            predicted_result = [invert(x, alphabet) for x in p_result]
            print("Predicted:",end=" ")
            print(predicted_result[0])
            print("Expected:",end=" ")
            print(expected_result)
            #print(predicted_result)
        except:
            print("Error!!!")
    print("Type input string")
    inputData = input("->");