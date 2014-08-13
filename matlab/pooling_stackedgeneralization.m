% DecMeg2014 example code.
% Simple prediction of the class labels of the test set by:
%   - pooling all the triaining trials of all subjects in one dataset.
%   - Extracting the MEG data in the first 500ms from when the stimulus starts.
% - Using a linear classifier (elastic net).
% Implemented by Seyed Mostafa Kia (seyedmostafa.kia@unitn.it) and Emanuele
% Olivetti (olivetti@fbk.eu) as a benchmark for DecMeg 2014.


%	Data preparation
clear all;
disp('DecMeg2014: https://www.kaggle.com/c/decoding-the-human-brain');

%	Path of the data folder
path = '../data/';

%	We throw away all the MEG data outside the first 0.5sec from when the
%	visual stimulus start.
tmin = -0.5;
tmax = 1;
fprintf('Restricting MEG data to the interval [%f, %f] sec.', tmin, tmax);

%	Indices of the subjects in the training sample
subjects_train = 1:16;

err = zeros([1 4]);
for idx = 1 : 4
	fprintf('\n\n\n\nTrial: %i\n========\n\n', idx);
	
	%	Subdivide the training sample into train and test subsamples
	subjects_train_test  = randsample(subjects_train, 4);
	subjects_train_train = setdiff(subjects_train, subjects_train_test);
	
	disp(strcat('Training on subjects: [', num2str(subjects_train_train), ']'));
	disp(strcat('Testing on subjects : [', num2str(subjects_train_test),  ']'));
	
	X_train_train = [];
	y_train_train = [];
	X_train_test  = [];
	y_train_test  = [];
	X_test = [];
	ids_test = [];
	disp(' ');  disp(' ');
	
	% Creating the L0 hypotheses.
	disp('Creating the L0 hypotheses.');
	L0 = [];
	for subjects_train_train_i = subjects_train_train
		disp(' ');
		filename = sprintf(strcat(path, 'train_subject%02d.mat'), subjects_train_train_i);
		disp(strcat('Loading ', filename));
		data = load(filename);
		XX = data.X;
		yy = data.y;
		sfreq = data.sfreq;
		tmin_original = data.tmin;
		disp('Dataset summary:')
		fprintf('XX: %d trials, %d channels, %d timepoints\n', size(XX,1), size(XX,2), size(XX,3));
		fprintf('yy: %d trials\n', size(yy, 1));
		disp(strcat('sfreq:', num2str(sfreq)));
		features = createFeatures(XX, tmin, tmax, sfreq, tmin_original);
% 		X_train_train = [X_train_train; features];
% 		y_train_train = [y_train_train; yy];
		
		%	Training code on train_train
		disp('Training the classifier ...')
		[BFinal,FitInfoFinal] = lasso(features, single(yy), 'Lambda',0.005, 'Alpha',0.9);
		L0 = [L0, [FitInfoFinal.Intercept; BFinal]];
		disp(' ');
	end
	disp(' ');  disp(' ');
	
	% Creating the test subset of the trainset.
	disp('Creating the train subset of the trainset.');
	for subjects_train_test_i = subjects_train_test
		disp(' ');
		filename = sprintf(strcat(path, 'train_subject%02d.mat'), subjects_train_test_i);
		disp(strcat('Loading ', filename));
		data = load(filename);
		XX = data.X;
		yy = data.y;
		sfreq = data.sfreq;
		tmin_original = data.tmin;
		disp('Dataset summary:')
		fprintf('XX: %d trials, %d channels, %d timepoints\n', size(XX,1), size(XX,2), size(XX,3));
		fprintf('yy: %d trials\n', size(yy, 1));
		disp(strcat('sfreq:', num2str(sfreq)));
		features = createFeatures(XX, tmin, tmax, sfreq, tmin_original);
		X_train_test = [X_train_train; features];
		y_train_test = [y_train_train; yy];
	end
	disp(' ');  disp(' ');
	
	
	%	Take the outputs of the different L0 classifiers
	%	Testing the trained classifier on train_test
	y_pred_train_L0 = [ones(size(X_train_test,1),1) X_train_test] * L0;
	y_pred_thresholded_train = zeros(size(y_pred_train_L0));
	for i = 1 : size(L0, 2)
		y_pred_thresholded_train(y_pred_train_L0(:,i) >= mean(y_pred_train_L0(:,i)), i) = 1;
	end
	y_pred_train_L1 = mode(y_pred_thresholded_train, 2);
	
	err(idx) = loss01(y_train_test, y_pred_train_L1);
	fprintf('\nEM = %f\n', err(idx));
	disp(' ');
	
end



% % Crating the testset.
% disp('Creating the testset.');
% subjects_test = 17:23;
% for i = 1 : length(subjects_test)
%     disp(' ');
%     filename = sprintf(strcat(path, 'test_subject%02d.mat'), subjects_test(i));
%     disp(strcat('Loading ', filename));
%     data = load(filename);
%     XX = data.X;
%     ids = data.Id;
%     sfreq = data.sfreq;
%     tmin_original = data.tmin;
%     disp('Dataset summary:')
%     fprintf('XX: %d trials, %d channels, %d timepoints\n', size(XX,1), size(XX,2), size(XX,3));
%     fprintf('Ids: %d trials\n', size(ids, 1));
%     disp(strcat('sfreq:', num2str(sfreq)));
%     features = createFeatures(XX, tmin, tmax, sfreq, tmin_original);
%     X_test = [X_test; features];
%     ids_test = [ids_test; ids];
% end
% disp(' ');  disp(' ');
% 
% 
% 
% % Testing the trained classifier on the test data
% y_pred = [ones(size(X_test,1),1) X_test] * [FitInfoFinal.Intercept;BFinal];
% y_pred_thresholded = zeros(size(y_pred));
% y_pred_thresholded(y_pred >= median(y_pred))= 1;
% 
% 
% 
% % Saving the results in the submission file:
% filename_submission = 'submission.csv';
% disp(strcat('Creating submission file: ', filename_submission));
% f = fopen(filename_submission, 'w');
% fprintf(f,'%s,%s\n','Id','Prediction');
% for i = 1 : length(y_pred_thresholded)
%     fprintf(f,'%d,%d\n', ids_test(i), y_pred_thresholded(i));
% end
% fclose(f);
% disp('Done.');
% 
% 
% 
% %	Display the mean of the Emperical Risk calculated.
% fprintf('\n\n========\nmean(ER) = %f\n========\n\n', mean(err));
