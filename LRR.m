function [Missrate, Grps] = LRR(X, s, r, Cst, ProjM, lambda, n)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input:
% X: data matrix, col->sample, row->observation
% s: truth
% Cst: 1 for affine subspace
% ProjM: projection method
% OptM: optimization method, {'L1Perfect','L1Noise','Lasso','L1ED'}
% lambda: regularization parameter for 2,1-norm
% n: number of estimated clusters
% K: number of coefficients to build the similarity graph
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% clc, clear all, close all
sdm = [length(find(s==1)) length(find(s==2)) length(find(s==3))];
K = max(sdm); %Number of top coefficients to build the similarity graph, enter K=0 for using the whole coefficients
% K = 0;
if Cst == 1
    K = max(sdm) + 1; %For affine subspaces, the number of coefficients to pick is dimension + 1 
end

Xp = DataProjection(X,r,ProjM);
% [u s v a e it] = inexact_alm_rpca(Xp, lambda);
% CMat = (u*u').^2;
[u, sh, v] = svd(Xp);
r = rank(sh);
V = v(:, 1:r);
CMat = V*V';
[CMatC,sc,OutlierIndx,Fail] = OutlierDetection(CMat,s);
if (Fail == 0)
    CKSym = BuildAdjacency(CMatC,K);
    Grps = SpectralClustering(CKSym,n);
    Grps = bestMap(sc,Grps);
    Missrate = sum(sc(:) ~= Grps(:)) / length(sc);
    save Lasso_001.mat CMat CKSym Missrate Fail
else
    save Lasso_001.mat CMat Fail
end