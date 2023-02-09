% Copyright 2017: Johannes L. Schoenberger <jsch at inf.ethz.ch>

% Revised 2020-7-17: San Jiang <jiangsan@cug.edu.cn>
%  - Make this script adapt to the revision of the master COLMAP.
%  - Add some SOTA learned local feature descriptors.

close all;
clear;
clc;

DATASET_NAMES = {'Fountain'};
% DATASET_NAMES = {'2015-1-25', '2016-4-17', 'yincheng', 'yincheng2'};
% DATASET_NAMES = {'yincheng2'};

for i = 1:length(DATASET_NAMES)
    %% Set the pipeline parameters.

    % TODO: Change this to where your dataset is stored. This directory should
    %       contain an "images" folder and a "database.db" file.
    DATASET_PATH = ['E:/datasets/' DATASET_NAMES{i}];
%     DATASET_PATH = ['E:/datasetsUAV/' DATASET_NAMES{i}];

    % TODO: Change this to where VLFeat is located.
    VLFEAT_PATH = 'E:/vlfeat-0.9.21';
    
    % TODO: Change this to where MatConvNet is located.
    MATCONVNET_PATH = 'E:/matconvnet-1.0-beta25';

    % TODO: Change this to where the COLMAP build directory is located.
    COLMAP_PATH = 'E:/colmap/colmap17_x64/bin';

    % Radius of local patches around each keypoint.
    PATCH_RADIUS = 15;

    % Whether to run matching on GPU.
    MATCH_GPU = gpuDeviceCount() > 0;

    % Number of images to match in one block.
    MATCH_BLOCK_SIZE = 50;

    % Maximum distance ratio between first and second best matches.
    MATCH_MAX_DIST_RATIO = 0.8;

    % Mnimum number of matches between two images.
    MIN_NUM_MATCHES = 15;
    
    % Maximum nunber images for image retrieval.
    MAX_NUM_IMAGES = '20';

    %% Setup the pipeline environment.

    run(fullfile(VLFEAT_PATH, 'toolbox/vl_setup'));
    run(fullfile(MATCONVNET_PATH, 'matlab/vl_setupnn'));

    IMAGE_PATH = fullfile(DATASET_PATH, 'images');
    KEYPOINT_PATH = fullfile(DATASET_PATH, 'keypoints');
    DESCRIPTOR_PATH = fullfile(DATASET_PATH, 'descriptors');
    MATCH_PATH = fullfile(DATASET_PATH, 'matches');
    DATABASE_PATH = fullfile(DATASET_PATH, 'database.db');

    %% Create the output directories.

    if ~exist(KEYPOINT_PATH, 'dir')
        mkdir(KEYPOINT_PATH);
    end
    if ~exist(DESCRIPTOR_PATH, 'dir')
        mkdir(DESCRIPTOR_PATH);
    end
    if ~exist(MATCH_PATH, 'dir')
        mkdir(MATCH_PATH);
    end

    %% Extract the image names and paths.
    
    image_files = dir(fullfile(IMAGE_PATH, '**/*.*'));
    image_names = {};
    image_paths = {};
    keypoint_paths = {};
    descriptor_paths = {};
    num_images = 0;
    for i = 1:length(image_files)
        % Skip directory.
        if image_files(i).isdir
            continue;
        end
        
        num_images = num_images + 1;
        
        image_folder = image_files(i).folder;
        sub_folder = image_folder(length(IMAGE_PATH) + 2:length(image_folder));
        
        image_name = fullfile(sub_folder, image_files(i).name);
        
        image_names{num_images} = image_name;
        image_paths{num_images} = fullfile(IMAGE_PATH, image_name);
        
        % Replace the dir seperaor '/' arsing from image name.
        image_name_ex = strrep(image_name, '/', '-');
        image_name_ex = strrep(image_name, '\', '-');
        keypoint_paths{num_images} = fullfile(KEYPOINT_PATH, [image_name_ex '.bin']);
        descriptor_paths{num_images} = fullfile(DESCRIPTOR_PATH, [image_name_ex '.bin']);
    end

    %% TODO: Compute the keypoints and descriptors.
    
    feature_extraction_root_sift
%     feature_extraction_dsp_sift
    
%     patch_extraction
    
    % feature_extraction_tfeat.py
%     feature_extraction_l2net
    % etc.

    %% Match the descriptors.
    %
    %  NOTE: - You must exhaustively match Fountain, Herzjesu, South Building,
    %          Madrid Metropolis, Gendarmenmarkt, and Tower of London.
    %        - You must approximately match Alamo, Roman Forum, Cornell.

    if num_images < 100
        exhaustive_matching
    else
        VOCAB_TREE_PATH = fullfile(DATASET_PATH, '../Oxford5k/vocab-tree.bin');
        approximate_matching
    end
end
