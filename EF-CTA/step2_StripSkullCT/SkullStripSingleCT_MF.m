clear;
clc;
close all;

addpath('NIfTI_20140122');

% pathNCCTImage = ['NCCT.nii.gz'];
% PathNCCT_Brain = ['NCCT_brain.nii.gz'];
read_dir = '/data2/xxxxxxxxx/xxxxx/dataset/from_zhou/batch2+3/step1_image/';
save_dir = '/data2/xxxxxxxxx/xxxxx/dataset/from_zhou/batch2+3/step2_image';

mkdir(save_dir);
files = dir(fullfile(read_dir, '*'));
n = length(files);
disp(['all sample= ',num2str(n-2)]);
for i=3:length(files)
    disp([num2str(i),'/',num2str(length(files)),' ',files(i).name]);
    pathNCCTImage = [fullfile(read_dir,files(i).name)];
    PathNCCT_Brain = [fullfile(save_dir,files(i).name)];

    disp(['------Strip skull of patient ' pathNCCTImage]);

    try
        % load the subject image
        ImgSubj_nii = load_untouch_nii(pathNCCTImage);
        ImgSubj_hdr = ImgSubj_nii.hdr;
        ImgSubj = ImgSubj_nii.img;
        %ImgSubj = double(ImgSubj);

        % skull stripping
        NCCT_Thr = 100; % for NCCT images
        CTA_Thr = 400; % for CTA images

        [brain] = SkullStripping(double(ImgSubj),CTA_Thr);

        % save image
        Output_nii.hdr = ImgSubj_hdr;
        Output_nii.img = int16(brain);
        save_nii(Output_nii, PathNCCT_Brain);

        disp([pathNCCTImage '----skull tripping finished']);
    catch ME
        disp(['Error processing ' files(i).name ': ' ME.message]);
        disp(['Continuing to next file...']);
        continue;
    end

end
