clear all
% load('dataset/Hopkins155/1R2RC/1R2RC_truth.mat');
load('dataset/Yale.mat')
X = fea(1:10:10:1000, :)';
s = gnd(1:10:1000);
% y0 = x(1:2, :, :);
% X = zeros(58, 459);
% for pt = 1:459
%     layer = y0(:, pt, :);
%     X(:, pt) = layer(:);
% end
[Missrate, truth] = SSC(X, s, 100, 0, 'PCA', 'Lasso', 0.001, 16);
% [Missrate, truth] = LRR(X, s, 0, 0, 'PCA', 0.1, 2);
for cls = 1:3
    idx = find(truth == cls);
    plot(y0(1, idx, 1), y0(2, idx, 1), '+')
    hold on
end