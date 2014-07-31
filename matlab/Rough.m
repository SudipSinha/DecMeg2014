%tic;  SVMObj = svmtrain(X_train, y_train);  toc;
%tic;  y_pred_train_svm = svmclassify(SVMObj, X_train);  toc;
%tic;  y_pred_test_svm  = svmclassify(SVMObj, X_test);   toc;
%tic;  err_train = loss01(y_train, y_pred_train_svm);  toc;

% load fisheriris;
% y = species;
% c = cvpartition(y,'k',10);
% fun = @(xT,yT,xt,yt)(sum(~strcmp(yt,classify(xt,xT,yT))));
% rate = crossval(fun,meas,y,'partition',c);

% %   kNN Classifier
% %	k = 5
% knn_pred_train = knnClassify(X_train, y_train, X_train, 5);
% fprintf('Emperical risk w.r.t. the 0-1-loss for k=5 = %f\n', loss01(y_train, knn_pred_train));
% %	k = 7
% knn_pred_train = knnClassify(X_train, y_train, X_train, 7);
% fprintf('Emperical risk w.r.t. the 0-1-loss for k=7 = %f\n', loss01(y_train, knn_pred_train));
% %	k = 9
% knn_pred_train = knnClassify(X_train, y_train, X_train, 9);
% fprintf('Emperical risk w.r.t. the 0-1-loss for k=9 = %f\n', loss01(y_train, knn_pred_train));
% %	k = 11
% knn_pred_train = knnClassify(X_train, y_train, X_train, 11);
% fprintf('Emperical risk w.r.t. the 0-1-loss for k=11 = %f\n', loss01(y_train, knn_pred_train));
% %	k = 13
% knn_pred_train = knnClassify(X_train, y_train, X_train, 13);
% fprintf('Emperical risk w.r.t. the 0-1-loss for k=13 = %f\n', loss01(y_train, knn_pred_train));
% %	k = 15
% knn_pred_train = knnClassify(X_train, y_train, X_train, 15);
% fprintf('Emperical risk w.r.t. the 0-1-loss for k=15 = %f\n', loss01(y_train, knn_pred_train));


% %	SVM
% % SVMStruct = svmtrain(X_train, y_train);
