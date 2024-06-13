DATAopts.Shape = 'Two Lines';    
DATAopts.Number = [2500, 2500];  DATAopts.AmbDim = 2; 
DATAopts.Angles = [0, pi/2];     DATAopts.NoiseSigma = 0.000;

DATAopts.Shape = 'Two Planes';    
DATAopts.Number = [2500, 2500];  DATAopts.AmbDim = 3; 
DATAopts.Angles = [0, pi/2];   DATAopts.NoiseSigma = 0.000; 

for i=1:10
    i
    [X, LabelsGT] = simdata(DATAopts, i);
    %smce(X,lambda,KMax,dim,n,gtruth,verbose)
    [Yc,Yj,clusters,missrate] = smce(X', 10, 50, 1, 2, LabelsGT, false);
    missrate
end
