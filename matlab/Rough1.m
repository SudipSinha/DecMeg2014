%# read some training data
labels = Y;
data = X;

%# grid of parameters
folds = 4;
C = 10 .^ (-4:2:4);
% C = 1; folds = 4;

%# grid search, and cross-validation
cv_acc = zeros(numel(C),1);
for i=1:numel(C)
	disp(i);
    cv_acc(i) = libsvmtrain(labels, data, sprintf('-t 0 -h 0 -c %f -v %f', C(i), folds));
end

%# pair (C,gamma) with best accuracy
[~,idx] = max(cv_acc);

% %# contour plot of paramter selection
% contour(C, gamma, reshape(cv_acc,size(C))), colorbar
% hold on
% plot(C(idx), gamma(idx), 'rx')
% text(C(idx), gamma(idx), sprintf('Acc = %.2f %%',cv_acc(idx)), ...
%     'HorizontalAlign','left', 'VerticalAlign','top')
% hold off
% xlabel('log_2(C)'), ylabel('log_2(\gamma)'), title('Cross-Validation Accuracy')
% 
% %# now you can train you model using best_C and best_gamma
best_C = 2^C(idx)
% best_gamma = 2^gamma(idx);