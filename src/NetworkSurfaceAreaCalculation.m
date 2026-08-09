clc
clear all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Script Name: NetworkSurfaceAreaCalculation.m
%
% Purpose:
%   This script calculates the surface area of each functional brain 
%   network for a given group of subjects (e.g., ASD, HC) from the ABIDE 
%   dataset. The script reads subject lists, phenotypic data, and network 
%   labeling results, then computes and stores the network size information 
%   along with subject age.
%
% Inputs:
%   - priors.mat: Contains prior information and network labels
%   - subject list (.txt): A text file with subject IDs for the selected site/group
%   - phenotypic data (.csv): Includes demographic information such as age
%   - CIFTI files (.dlabel.nii / .dscalar.nii): Functional networks and 
%     surface area data for each subject
%   - midwall.mat: Mask file for excluding NaN positions in cortical surface
%
% Outputs:
%   - A MATLAB table containing the surface area of each functional network
%     (per subject), along with subject age
%   - A .csv file (e.g., ABIDE1_ASD_sf.csv) saved in the current directory
%
% Dependencies:
%   - FieldTrip functions (ft_read_cifti, ft_read_cifti_mod)
%   - pfm_calculate_network_size function
%   - MATLAB Statistics and Machine Learning Toolbox (for readtable, writetable)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Calculate the surface area of each network (HC group)
group = 'ASD';
site = 'ABIDE1';

priors = load('./PFM-Depression-main/PFM-Tutorial/Utilities/priors.mat');
networkLabels = priors.Priors.NetworkLabels; 

filePath = sprintf('./PFM-Depression-main/data_list/%s_%s_Male_6-30_list_fd0.3.txt', site, group);
fileID = fopen(filePath, 'r');
fileLines = textscan(fileID, '%s', 'Delimiter', '\n');
fclose(fileID);

% Load the phenotypic data
csvPath = sprintf('./data/%s/%s_Phenotypic_V1_0b_yrqcheck.csv', site, site);
phenotypicData = readtable(csvPath);

VA = ft_read_cifti(['./data/' site '/' subjectName '/ses-1/anat/' subjectName '.midthickness_va.32k_fs_LR.dscalar.nii']);
VA = VA.midthickness_va;
load('./PFM-Tutorial/midwall.mat');
VA(nan_positions) = NaN;

maxCols = 21;
allResultsTable = table();

for i = 1:length(fileLines{1})
    subjectName = fileLines{1}{i}; 

%     subjectPath = ['./data/' site '/' subjectName '/pfm/output_NetworkSize.mat'];
    FunctionalNetworks = ft_read_cifti_mod(['./Abide_infomap/data/' site '/' subjectName '/pfm/Bipartite_PhysicalCommunities+FinalLabeling_filter.dlabel.nii']);
    
    Structures = {'CORTEX_LEFT','CORTEX_RIGHT'}; % in this case, cortex only
    % calculate the size of each functional brain network
    [networkSize,uCi] = pfm_calculate_network_size(FunctionalNetworks,VA,Structures);
%     save('output_NetworkSize.mat', 'NetworkSize', 'uCi');

%     if isfile(subjectPath)  % Check if the file exists
%     t = load(subjectPath);  

%     networkSize = t.NetworkSize;
%     uCi = t.uCi;
    
    resultVector = zeros(1, maxCols);  

    for j = 1:length(networkSize)
        resultVector(uCi(j)) = networkSize(j);  
    end

    % Get the age for the current subjectName
    ageIndex = strcmp(phenotypicData.FILE_ID, subjectName);
    if any(ageIndex)
        age = phenotypicData.AGE_AT_SCAN(ageIndex);
        if numel(age) > 1
            age = age(1);  % Take the first matching value
        end
    else
        age = NaN;  % If FILE_ID not found, set age to NaN
    end

    % Ensure that column names are valid MATLAB variable names
    validNetworkLabels = matlab.lang.makeValidName(networkLabels(1:maxCols));

    resultTable = array2table(resultVector, 'VariableNames', validNetworkLabels);  
    resultTable.Properties.RowNames = {subjectName};  
    
    % Add age to the resultTable
    resultTable.age = age;  

    % Fix merging issues
    if isempty(allResultsTable)
        allResultsTable = resultTable;
    else
        allResultsTable = [allResultsTable; resultTable];  
    end

%     else
%         fprintf('File %s does not exist, skipping...\n', subjectPath);  % If file does not exist, skip it
%     end
end

disp(allResultsTable);

% Save results
outputPath = sprintf('./%s_%s_sf.csv', site, group); 
writetable(allResultsTable, outputPath, 'WriteRowNames', true);  % Write table to CSV file
disp(['Results saved to ' outputPath]);
