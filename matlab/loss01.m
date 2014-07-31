function [ loss ] = loss01( y, y_pred )

loss = mean(y ~= y_pred);

end
