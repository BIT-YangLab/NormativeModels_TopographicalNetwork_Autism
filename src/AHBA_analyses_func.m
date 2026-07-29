% Reference: Morgan et al. PNAS, 2019.
% Adapted by Ziteng Han, 2024.
% Adapted by Ruoqi Yang, 2025.

clc; clear all; 


%% Step 1: Load Data
% Load AHBA gene expression data (predictors)
X = readtable('ABIDE_gene_expression_data_100.csv','VariableNamingRule','preserve');   
geneData = X(:, 2:end);
geneMatrix = table2array(geneData);   
columnNames  =  X.Properties.VariableNames;   
geneNames  =  columnNames(2:end);   

% Load PLS target data (surface area deviations)
weight=load('deviation_values.mat');
Y = weight.Y; 

% Z-score normalization
X = zscore(geneMatrix);  
Y = zscore(Y);     

%% Step 2: PLS regression & variance explained
dim=15; 
[XL,YL,XS,YS,BETA,PCTVAR,MSE,stats] = plsregress(X,Y,dim);

% Correlation of first 5 components with target
for k=1:5
    [R_temp, p_temp] = corr(XS(:,k), weight.cov_avg);
    fprintf('PLS Component %d: R = %.3f, p = %.3f\n', k, R_temp, p_temp);
end

%% Step 3: Permutation test
t = 1; % chosen PLS dimension
[R, p] = corr(XS(:,t), weight.cov_avg);

num_permutations = 1000; 
R_perm = zeros(num_permutations,1);

for i = 1:num_permutations
    perm_idx = randperm(length(weight.cov_avg)); 
    perm_Y = weight.cov_avg(perm_idx); 
    R_perm(i) = corr(XS(:,t), perm_Y); 
end

p_perm = sum(abs(R_perm) >= abs(R)) / num_permutations;
fprintf('Permutation test p-value: %.4f\n', p_perm);

% Histogram of null distribution
figure;
histogram(R_perm, 30, 'FaceColor', [0.6,0.6,0.6], 'EdgeColor', 'k');
hold on;
plot([R R], ylim, 'r-', 'LineWidth', 2);
xlabel('Permuted Correlation Coefficients');
ylabel('Frequency');
title(sprintf('Permutation Test for PLS Component %d', t));
grid on;

%% Step 4: Bootstrap gene weights
bootnum = 1000;
genes=geneNames; 
geneindex=1:length(genes);
dim = 1; 

[XL,YL,XS,YS,BETA,PCTVAR,MSE,stats] = plsregress(X,Y,dim);
[R1, ~] = corr(XS, Y); 

for k = 1:dim
    if R1(k,1) < 0  
        stats.W(:,k) = -1 * stats.W(:,k);
        XS(:,k) = -1 * XS(:,k);
    end
end

PLSids = cell(dim,1);
geneindex_sorted = cell(dim,1);
PLSweights_init = cell(dim,1);

for k = 1:dim
    [PLSweights_init{k}, idx] = sort(stats.W(:,k), 'descend');
    PLSids{k} = genes(idx);
    geneindex_sorted{k} = geneindex(idx);
end

% Save ROI scores
for k = 1:dim
    filename = sprintf('PLS%d_ROIscores.csv', k);
    csvwrite(filename, XS(:,k));
end

% Bootstrap resampling
PLSweights = cell(dim,1);
for k = 1:dim, PLSweights{k} = []; end

for i = 1:bootnum
    myresample = randsample(size(X,1), size(X,1), 1);
    Xr = X(myresample, :);
    Yr = Y(myresample, :);
    [XL,YL,XS,YS,BETA,PCTVAR,MSE,stats] = plsregress(Xr, Yr, dim); 
    
    for k = 1:dim
        temp = stats.W(:,k); 
        newW = temp(idx); 
        if corr(PLSweights_init{k}, newW) < 0  
            newW = -1 * newW;
        end
        PLSweights{k} = [PLSweights{k}, newW];
    end
end

% Compute Z-scores
PLSsw = cell(dim,1); PLSZ = cell(dim,1);
for k = 1:dim
    PLSsw{k} = std(PLSweights{k}, 0, 2); 
    PLSZ{k} = PLSweights_init{k} ./ PLSsw{k}; 
end

% Save sorted bootstrap gene weights
for k = 1:dim
    [Z_sorted, ind] = sort(PLSZ{k}, 'descend');
    PLSsorted = PLSids{k}(ind); 
    geneindex_final = geneindex_sorted{k}(ind);
    filename = sprintf('PLS%d_geneWeights_100.csv', k);
    fid = fopen(filename, 'w');
    for i = 1:length(genes)
        fprintf(fid, '%s, %d, %f\n', PLSsorted{i}, geneindex_final(i), Z_sorted(i));
    end
    fclose(fid);
end

fprintf('Bootstrap complete! Gene lists saved for PLS1 to PLS%d.\n', dim);
