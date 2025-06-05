clc;
clear;
close all;
warning('off','all');
addpath('../00-preprocess', '00-build', '00-my_fune', 'Tools', 'Tools/export_fig','Quality_Indices', 'analysis_MatLab/evaluation', 'Multifocus_Image_Fusion_Evaluation', 'Multifocus_Image_Fusion_Evaluation\matlabPyrTools');
%%
files = dir("./data/RoadSceneFusion/test/ir/*"); %获取RoadSceneFusion的测试数据
% files=dir('./data/TNO/test_train/ir/*'); %获取TNO的训练数据
% files=dir('./data/TNO/test/ir/*'); %获取TNO的测试数据
% files = dir("./data/medFusion/test/SPECT-MRI/MRI/*");
files = sort_nat({files.name}); 
files = files(3:end); %选择需要比较的图像
% files = convertCharsToStrings({files.name});

%% Settings
opts.exp_desc = 'test/IR_VIS';
opts.task = 'IR_VIS';
opts.dataset = "RoadSceneFusion"; %choose RoadSceneFusion, TNO, medFusion, corresponding to <files>
opts.copy_list = [];
opts.mode = 'test_v2'; %%test_v2: print once, test: print to logs for each alg, search: gridsearch for optimization, search_v2 used for internal grid search in single method.
% opts.metric = ["EN", "MI", "PSNR", "SSIM", "Qabf", "SCD", "VIF", "SD", ... 
%                "AG", "CC", "SF", "MSE", "Nabf"]; %all
% opts.metric = ["NCIE", "Q_G", "PWW", "Q_P", "Q_S", "Q_CV", "Q_CB", "Q_MI", "Q_TE", "Q_Y", "FMI_pixel", "FMI_dct", "FMI_w", "MS_SSIM"];
% opts.metric = ["EN", "MI", "PSNR", "SSIM", "Qabf", "SCD", "VIF", "SD", "AG", "SF", "MSE", "CC"]; %
opts.metric = ["Q_S", ];
if strcmp(opts.dataset, 'RoadSceneFusion')
    DL_lists = ["-", "densefuse", "ifcnn", "u2fusion", "ydtr", "dcformer_u2fusion", "GTF", "nsct", "densefuse_reproduce", "NSST", 'defuse', 'rfnnest', 'DDcGAN_500_optim_205', ...
        "dcformer_RS_TNO_u2fusion_loss_wx_new", "swinFuse", "SwinFusion", "LRRNet"]; %需要比较的方法列表
elseif strcmp(opts.dataset, 'TNO')
    DL_lists = ["-", "DenseFuse_TNO", "FusionGAN_TNO", "IFCNN_TNO", "PIAFusion_TNO", "PMGI_TNO", ...
        "my_U2Fusion_t_TNO", "RFN-Nest_TNO", "SDNet_TNO", "SeAFusion_TNO", "U2Fusion_TNO", ...
        "my_U2Fusion_ir_vi_TNO", "my_U2Fusion_vi_ir_TNO", ...
        "pia_percp_loss_TNO", "pia_loss_TNO", "dcformer_new_training_data_TNO", "dcformer_RoadScene_and_TNO",...
        "dcformer_RoadScene_and_TNO_U2FusionLoss", "new_U2Fusion", "dcformer_u2fusion_5_2_10_TNO", "dcformer_RS_TNO_u2fusion_loss_wo_weighted_ssim", "defuse", ...
        "new_U2Fusion_100", "", "diffusion_change_sample", "dcformer_RS_TNO_u2fusion_loss_wx_new", "defuse", "ydtr_TNO", "LRRNet"]; %
elseif strcmp(opts.dataset, 'medFusion')
    %     data_dir = "./";
    DL_lists = ["-", "MATR"]; %需要比较的方法列表
end

Algorithms = DL_lists(2:end);

% Alg_names=fieldnames(Algorithms);
% Alg_names = Alg_names(6:end);
% Alg_names = DL_lists(2:end); %从列表中选择比较的方法
% Alg_names = {"dcformer_RS_TNO_u2fusion_loss_wo_weighted_ssim"};
% Alg_names = {"densefuse", "dcformer_u2fusion", "densefuse", "GTF", "ifcnn", "nsct", "u2fusion", 'ydtr'};
% Alg_names = {"DenseFuse_TNO", "FusionGAN_TNO", "IFCNN_TNO", "PIAFusion_TNO", "PMGI_TNO", "RFN-Nest_TNO", "SDNet_TNO", "SeAFusion_TNO", "U2Fusion_TNO", "pia_percp_loss_TNO", "pia_loss_TNO", "dcformer_new_training_data_TNO", "dcformer_RoadScene_and_TNO"};
% Alg_names = {'DDcGAN_500_optim_205'};
% Alg_names = {"ifcnn", "u2fusion", "ydtr"};
Alg_names = {"LRRNet"};
A_num = length(Algorithms);
Re_tensor = cell(A_num,1);
MatrixTimes = zeros(A_num, 1);

fprintf('###################### Please wait......######################\n')
%% Show result
location  = [10 50 45 85];%设置放大区域
range_bar = [0, 0.2]; %残差图调色
data_name = strcat('3_EPS/', opts.dataset, '/', opts.dataset, '_');  % director to save EPS figures
mkdir(dirpath(char(data_name)));

%%
flag_cut_bounds = 0;% Cut Final Image
Qblocks_size = 32; % for Q2n
dim_cut = 30;% Cut Final Image
thvalues = 0;
L = 11;% Radiometric Resolution
maxvalue = 255; %data and results should be 0-255.
opts.ratio = 1;

printEPS = 0; % save figures
flag_show = 0; % save figures
flagvisible = 'off'; %'on', show figures
flag_savemat = 0; %save mat
flag_zoomin = 1;
flag_fastEval = 0;

%%
% files = files(3:end);
exm_num = length(files);
% exm_num = 3;
for num = 1:exm_num
%     if ~ismember(num, [7, 8, 42]) %183, 187, 198, 199
        alg = 0;
        opts.file = files{num};
        opts.file
%             ir_image = imread(strcat("D:/Datasets/FLIR/RoadSceneFusion/training_data/test/ir test/", num2str(num-1),'.bmp'));
%             vi_image = imread(strcat("D:/Datasets/FLIR/RoadSceneFusion/training_data/test/vi test/", num2str(num-1),'.bmp'));
        %     ir_image = imread(strcat("./data/",opts.dataset,"/test_train/ir/", files{num}));
        %     vi_image = imread(strcat("./data/",opts.dataset, "/test_train/vi/", files{num}));
        %     ir_image = imread(strcat("./data/",opts.dataset,"/test/SPECT-MRI/SPECT/", files{num}));
        %     vi_image = imread(strcat("./data/",opts.dataset, "/test//SPECT-MRI/MRI/", files{num}));
        data.ir_image = imread(strcat("./data/",opts.dataset,"/test/ir/", files{num}));
        data.vi_image = imread(strcat("./data/",opts.dataset, "/test/vi/", files{num}));
        ir_image=data.ir_image;
        vi_image=data.vi_image;
        run_fusion_EN();
%     end
end



%%
if ~contains(opts.mode, 'search')
    %% 测试和直接加载结果的时候
    %     matrix2latex(Avg_MatrixResults(:, [1,2,3,4,7,8,9,10,11,12]),strcat(opts.task, '_', opts.dataset, '_Avg_RR_Assessment.tex'), 'rowLabels',Alg_names,'columnLabels', ...
    %         [{'EN'},{'EN-std'},{'MI'},{'MI-std'},{'SSIM'},{'SSIM-std'},{'Qabf'},{'Qabf-std'},{'SCD'},{'SCD-std'},{'VIF'},{'VIF-std'},{'Time'},{'Time-std'}],'alignment','c','format', '%.4f');
    fprintf('\n')
    disp('#######################################################')
    disp(['Display the Avg/Std performance for:', num2str(1:exm_num)])
    disp('#######################################################')
    fprintf([ repmat('|====%s====',1,numel(opts.metric))], opts.metric);
    fprintf('|\n');
    for i=1:length(Alg_names)
        fprintf("%s ", Alg_names{i});
        fprintf([ repmat('%.4f ',1,numel(Avg_MatrixResults(i, :))) '\n'], Avg_MatrixResults(i, :));
    end
else
    %% 搜索best结果的时候
    matrix2latex(Best_Avg_MatrixResults(:, [1,2,3,4,5,6]),strcat(opts.task, '_', opts.dataset, '_searchBest_Avg_RR_Assessment.tex'), 'rowLabels',Alg_names,'columnLabels', ...
        [{'EN'},{'EN-std'},{'MI'},{'MI-std'},{'SSIM'},{'SSIM-std'},{'Qabf'},{'Qabf-std'},{'SCD'},{'SCD-std'},{'FMI_p'},{'FMI_p-std'},{'Time'},{'Time-std'}],'alignment','c','format', '%.4f');
    fprintf('\n')
    disp('#######################################################')
    disp(['Display the Avg/Std (search best) performance for:', num2str(1:exm_num)])
    disp('#######################################################')
    fprintf([ repmat('|====%s====',1,numel(opts.metric))], opts.metric);
    fprintf('|\n');
    for i=1:length(Alg_names)
        fprintf("%s ", Alg_names{i});
        fprintf([ repmat('%.4f ',1,numel(Best_Avg_MatrixResults(i, :))) '\n'], Best_Avg_MatrixResults(i, :));
    end
end
fprintf('###################### Complete execution! ! !######################\n')



