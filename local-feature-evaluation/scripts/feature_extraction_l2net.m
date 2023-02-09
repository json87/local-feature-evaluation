% Copyright 2017: Johannes L. Schoenberger <jsch at inf.ethz.ch>

% Revised 2020-7-17: San Jiang <jiangsan@cug.edu.cn>
%  - Make this script adapt to the revision of the master COLMAP.
%  - Add some SOTA learned local feature descriptors.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% if flagCS   %if CS structure is used, testdata should be in size of 64*64*1*N 
%     % the pixel value of the input patches shoud be in range of 0 tp 255.
%     testPatch = randi([0 255],64,64,1,10,'single');
% else
%     testPatch = randi([0 255],32,32,1,10,'single');
% end
% 
% desFloat = cal_L2Net_des(rootPath,trainSet,flagCS,flagAug,testPatch,batchSize,flagGPU);
% 
% desBinary = desFloat;
% desBinary(find(desBinary>0)) = 1;
% desBinary(find(desBinary<=0)) = -1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Setup the parameters of the L2Net.
rootPath = 'E:/datasetsUAV/pretrained-models';
trainSet = 'LIB';%'YOS','ND','HP'
flagCS = 1;
flagAug = 1;
flagGPU = 1;
batchSize = 1000;

% Compute descriptors for all images.
for i = 1:num_images
    fprintf('Computing features for %s [%d/%d]', ...
            image_names{i}, i, num_images);

    if exist(keypoint_paths{i}, 'file') ...
            && exist(descriptor_paths{i}, 'file')
        fprintf(' -> skipping, already exist\n');
        continue;
    end

    tic;
    
    % Loaded archieved patches.
    patches_path = [descriptor_paths{i} '.patches.mat'];
    patches_file = matfile(patches_path);
    patches3163 = patches_file.patches;
    
    if length(size(patches3163)) ~= 3
        fprintf(' -> skipping, invalid input');
        if flagCS
            write_descriptors(descriptor_paths{i}, zeros(0, 256));
        else
           write_descriptors(descriptor_paths{i}, zeros(0, 128));
        end
        continue;
    end
    
    % Expand patches to the desired size for l2net.
    if flagCS
        patches = single(zeros(size(patches3163, 1), 1, 64, 64));
        patches(:, 1, 1:63, 1:63) = patches3163;
        patches(:, 1, 64, 1:63) = patches3163(:, 63, :);
        patches(:, 1, 1:63, 64) = patches3163(:, :, 63);
        patches(:, 1, 64, 64) = patches3163(:, 63, 63);
        patches = permute(patches, [3, 4, 2, 1]);
    else
        patches = single(zeros(size(patches3163, 1), 1, 32, 32));
        patches(:, 1, 1:31, 1:31) = patches3163;
        patches(:, 1, 32, 1:31) = patches3163(:, 31, :);
        patches(:, 1, 1:31, 32) = patches3163(:, :, 31);
        patches(:, 1, 32, 32) = patches3163(:, 31, 31);
        patches = permute(patches, [3, 4, 2, 1]);
    end
    
    % Generate a 128(or 256)*N matrix, each colum is a descriptor
    descriptors = cal_L2Net_des(rootPath, trainSet, flagCS, flagAug, patches, batchSize, flagGPU);
    
    if size(descriptors, 2) == 0
        if flagCS
            descriptors = zeros(0, 128);
        else
            descriptors = zeros(0, 256);
        end
    else
        descriptors = descriptors';
    end
    
    % Make sure that each keypoint has one descriptor.
    assert(size(patches, 4) == size(descriptors, 1));

    % Write the descriptors to disk for matching.
    write_descriptors(descriptor_paths{i}, descriptors);

    fprintf(' in %.3fs\n', toc);
end
