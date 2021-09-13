function [y]=unnormalize(y_pre,Max,Min)

    y=y_pre.*repmat((Max-Min),[length(y_pre),1])+repmat(Min,[length(y_pre),1]);          %预测值反归一化
