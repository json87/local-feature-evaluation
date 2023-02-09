pool = gcp('nocreate');
if isempty(pool)
    pool = parpool(maxNumCompThreads());
end

parfor i = 1:num_images
    fprintf('Extracting patches for %s [%d/%d]', ...
            image_names{i}, i, num_images);

    patches_path = [descriptor_paths{i} '.patches.mat'];

    if exist(keypoint_paths{i}, 'file') ...
            && exist(patches_path, 'file')
        fprintf(' -> skipping, already exist\n');
        continue;
    end

    tic;

    % Read the image for keypoint detection, patch extraction and
    % descriptor computation.
    image = imread(image_paths{i});
    if ismatrix(image)
        image = single(image);
    else
        image = single(rgb2gray(image));
    end

    % Read the pre-computed SIFT keypoints.
    keypoints = read_keypoints(keypoint_paths{i});

    % Compute the descriptors for the detected keypoints.
    if size(keypoints, 1) == 0
        patches = zeros(0, 2 * PATCH_RADIUS + 1, 2 * PATCH_RADIUS + 1);
    else
        % Extract the local patches for all keypoints.
        patches = extract_patches(image, keypoints, PATCH_RADIUS);
    end

    % Dump the patches to a temporary file.
    patches_file = matfile(patches_path, 'writable', true);
    patches_file.patches = patches;

    fprintf(' in %.3fs\n', toc);
end
