function [missrate,grp] = spectral_clustering(W,n,gtruth)
% W = W./repmat(max(W,[],1),1);
W(isnan(W))=0;
if (n == 1)
    gtruth = ones(1,size(W,1));
end

MAXiter = 1000;
REPlic = 20;
N = size(W,1);
n = max(gtruth);

% cluster the data using the normalized symmetric Laplacian 
D = diag( 1./sqrt(sum(W,1)+eps) );
L = eye(N) - D * W * D;
[~,~,V] = svd(L,'econ');
Yn = V(:,end:-1:end-n+1);
for i = 1:N
    Yn(i,:) = Yn(i,1:n) ./ norm(Yn(i,1:n)+eps);
end

if n > 1
    grp = kmeans(Yn(:,1:n),n,'maxiter',MAXiter,'replicates',REPlic,'EmptyAction','singleton');
else 
    grp = ones(1,N);
end

% compute the misclassification rate if n > 1
% missrate = missclassGroups(grp,gtruth) ./ length(gtruth); 
grp=bestMap(gtruth,grp);
missrate = sum(grp'~=gtruth)/length(gtruth);