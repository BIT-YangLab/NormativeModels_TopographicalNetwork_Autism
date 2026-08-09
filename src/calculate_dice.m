%{
Dice similarity calculation for individual subjects (HC vs ASD)
--------------------------------------------------------------------------
Overview
- This script computes the Dice similarity coefficient between each subject’s
  parcellation and the corresponding group-level parcellation (20 networks).
- Groups: Healthy Controls (HC) and Autism Spectrum Disorder (ASD).
- The analysis is based on ABIDE1 and ABIDE2 datasets, with normative models 
  provided in "PFM-Depression-main/Nomitivemodel/output_dlabel/".

Inputs
- Subject lists:
    ./PFM-Depression-main/data_list/ABIDE1_HC_Male_6-30_list_fd0.3.txt
- Group-level parcellations:
    HC_group.dlabel.nii, ASD_group.dlabel.nii
- Individual subject parcellations:    ./data/ABIDE1/<subject>/pfm/Bipartite_PhysicalCommunities+FinalLabeling_filter.dlabel.nii    ./data/ABIDE2/<subject>/pfm/Bipartite_PhysicalCommunities+FinalLabeling_filter.dlabel.nii

Method
- For each subject:
    1. For each of the 20 labels, compute Dice coefficient:
         Dice = 2 * |A ∩ B| / (|A| + |B| + eps)
- Store subject × label Dice coefficients in a matrix.

Outputs
- Dice_mode_HC.mat
- Dice_mode_ASD.mat
  Each file contains a matrix (n_subjects × 20 networks) of Dice coefficients.

Dependencies
- MATLAB toolboxes:
    + ft_read_cifti (FieldTrip CIFTI support, e.g., from cifti-matlab-master)
    + Add paths to: Gordon2016_mod, PFM-Depression-main/dlabel_visual, cifti-matlab-master

Note
- NaNs in the parcellation vectors are removed before comparison.
- An epsilon term is added in denominator to prevent division by zero.

%}


clc;
clear all;

% Add required paths
addpath(genpath('./Gordon2016_mod')); 
addpath(genpath('./PFM-Depression-main/dlabel_visual')); 
addpath('./cifti-matlab-master'); 

groups = {'HC', 'ASD'};

for g = 1:length(groups)
    group = groups{g};

    % Load subject lists (ABIDE1 + ABIDE2)
    list1_path = ['./PFM-Depression-main/data_list/ABIDE1_' group '_Male_6-30_list_fd0.3.txt']; 
    list2_path = ['./PFM-Depression-main/data_list/ABIDE2_' group '_Male_6-30_list_fd0.3.txt']; 

    list1 = importdata(list1_path);
    list2 = importdata(list2_path);
    subjects = [list1; list2];
    n = length(subjects);

    % Load group-level parcellation (B)
    if strcmp(group, 'HC')
        B_struct = ft_read_cifti('./PFM-Depression-main/Nomitivemodel/output_dlabel/HC_group.dlabel.nii'); 
    else
        B_struct = ft_read_cifti('./PFM-Depression-main/Nomitivemodel/output_dlabel/ASD_group.dlabel.nii'); 
    end

    B = B_struct.indexmax(1:59412);
    B(any(isnan(B), 2)) = [];

    Dice_all = zeros(n, 20);  % Dice scores (subjects × 20 networks)

    for i = 1:n
        subj = subjects{i};

        % Subject-level parcellation path
        if ismember(subj, list1)
            path_A = fullfile('./data/ABIDE1', subj, 'pfm', 'Bipartite_PhysicalCommunities+FinalLabeling_filter.dlabel.nii'); 
        else
            path_A = fullfile('./data/ABIDE2', subj, 'pfm', 'Bipartite_PhysicalCommunities+FinalLabeling_filter.dlabel.nii'); 
        end

        % Load subject-level parcellation (A)
        A_struct = ft_read_cifti(path_A);
        A = A_struct.indexmax(1:59412);
        A(any(isnan(A), 2)) = [];

        % Compute Dice similarity for each label
        for label = 1:20
            A_bin = (A == label);
            B_bin = (B == label);
            Dice_all(i, label) = 2 * sum(A_bin & B_bin) / (sum(A_bin) + sum(B_bin) + eps);
        end
        disp(Dice_all(i,:));
    end

    % Save Dice matrix for this group
    save(['Dice_mode_' group '.mat'], 'Dice_all'); 
end
