# Diary


## 2014-07-28
* Data set
  * X = ((trial = 580) x (channel = 306 = 3 * 102) x (time = 375 = 1.5s * 250 Hz)
  * Y = {0, 1} ^ 580
  * P(Y=1) = P(Y=0) = 0.5 (Training data)
* Running and understanding the benchmark code given.
* Converting from time series to feature vectors
* (9NN, 0-1-loss): 0.35


## 2014-07-29

* Cannot run SVM from MATLAB - run tonight
* Cannot run PCA from MATLAB


## 2014-07-30
#### Pooling Default
* t ∈ [0, 0.5]  : (0.101976, 0.64298)
* t ∈ [0, 1]    : (0.073932, 0.63993)
* t ∈ [-0.5, 1] : (0.050563, 0.64342)
#### prejudices
* t in [0, 0.5] : (0.102613, 0.63731)
* t in [0, 1]   : (0.102613, 0.63731)
#### Same
The above data does not make sense, since we’re only looking at the test error. The idea is to use a part of the training data for training and the rest as test. We can loop over different combinations to achieve an unbiased result. Met with Morteza and discussed this.


## 2014-07-31
#### Pooling
* t ∈ [0, 0.5]    : 0.085914
* t ∈ [0, 1]       : 0.064523
* t ∈ [-0.5, 0.5]: 0.058013
* t ∈ [-0.5, 1]   : 0.043810
#### Minus mean(signal for t ∈ [-0.5, 0]
* t ∈ [0, 0.5]    : 0.088604
* t ∈ [0, 1]       : 0.062443
#### Read
* Transfer learning: http://en.wikipedia.org/wiki/Inductive_transfer
* Transductive learning: http://en.wikipedia.org/wiki/Transduction_(machine_learning)
* Transduction < Induction < Deduction (in Generality)


## 2014-08-01
* Learnt about Ensemble Learning
* Read the paper ‘MEG Decoding Across Subjects.pdf’


## 2014-08-04
* Set up a common cross validation framework.
#### Smoothing - 3c
* t ∈ [0, 0.5]    : 0.081143


## 2014-08-06
* Ensemble learning implementation: 0.413559 
* Obviously does not seem to work. :(


## 2014-08-07
Wasted a lot of time trying to set up SVM-Light. Failed.


## 2014-08-08
Morteza help set up libSVM.


## 2014-08-11
Completed the code for running SVM and getting the output. Letting it run for 2 days (expected runtime).


## 2014-08-13
Ahhh, all efforts wasted. The LAN was not working. Unable to login, I hard reset the machine. Will never get to know if the code ran.
Learnt to always use a console logger. Set it up using `startup.m` and `finish.m` files.
#### Smoothing
* Using single moving (central) average on `n`c successive elements.
* Results in doc.md 
#### SVM
* One last try - Will try to run it with logging on. Hopeful to see some result tomorrow.
