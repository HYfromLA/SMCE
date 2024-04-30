DATAopts.Shape = 'Dollar Sign';         DATAopts.Number = [7000, 3000];         % 4
DATAopts.AmbDim = 10;                    DATAopts.NoiseSigma = 0.030;

[X, LabelsGT] = simdata(DATAopts, 1);
lambdas = [1 5 10 20 50]; KMaxs = [5 10 20 30 40 50 80 100];
smce_tune = []; 
for i=1:length(lambdas)
    i
    for j=1:length(KMaxs)
        j
        [Yc,Yj,clusters,missrate] = smce(X',lambdas(i),KMaxs(j),2,max(LabelsGT),LabelsGT,false);
        smce_tune=[smce_tune; [lambdas(i) KMaxs(j) 1-missrate]]; 
    end
end
save('smce_result.mat', 'smce_tune');
