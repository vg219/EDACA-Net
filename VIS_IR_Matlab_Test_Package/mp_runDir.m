function mp_runDir(fusion_dir, dataset_name, method_name, rgb_test, test_mode_easy, varargin)
    % configs
    addpath('/Data3/cao/ZiHanCao/exps/panformer/VIS_IR_Matlab_Test_Package')
    addpath('analysis_MatLab/evaluation/');
    addpath('Quality_Indices/');

    % parse the args
    args = parseInputs(varargin{:});

    % open a file to log
    if isfield(args, 'file_name')
        fileID = fopen(strcat('logs/', args.file_name), 'w');
    else
        fileID = fopen('logs/VIS_IR_log.txt', 'w');
    end

    if isfield(args, 'test_ext')
        test_ext = args.test_ext;
    else
        test_ext = 'jpg';
    end

    if isfield(args, 'save_excel_file')
        save_excel_file = args.save_excel_file;
    else
        save_excel_file = 1;
    end

    dataset_name = upper(dataset_name);
    save_dir = strcat('Metric/', dataset_name); %存放Excel结果的文件夹


    % setting and lauching and parallel pool
    delete(gcp('nocreate'))
    dualfprintf(fileID, 'close any existing pools\n')
    num_threads = 8;
    dualfprintf(fileID, 'Running with %i threads\n', num_threads)
    parpool('local', num_threads);

    switch dataset_name
        case 'ROADSCENE'
            vi_dir = '/Data3/cao/ZiHanCao/datasets/RoadSceneFusion/test/vi test';
            test_ext = 'bmp';
        case 'TNO'
            vi_dir = '/Data3/cao/ZiHanCao/datasets/TNO/new_test_data/vi';
            test_ext = 'png';
        case 'LLVIP'
            vi_dir = '/Data3/cao/ZiHanCao/datasets/LLVIP/data/visible/test';
            test_ext = 'jpg';
        case 'MSRS'
            vi_dir = '/Data3/cao/ZiHanCao/datasets/MSRS/test/vi';
            test_ext = 'jpg';
        case 'M3FD'
            vi_dir = '/Data3/cao/ZiHanCao/datasets/M3FD/M3FD_Fusion/vi';
            test_ext = 'jpg';
        case 'MED_HARVARD_SPECT_MRI'
            vi_dir = '/Data3/cao/ZiHanCao/datasets/MedHarvard/xmu/SPECT-MRI/test/SPECT';
            test_ext = 'png';
        case 'MED_HARVARD_PET_MRI'
            vi_dir = '/Data3/cao/ZiHanCao/datasets/MedHarvard/xmu/PET-MRI/test/PET';
            test_ext = 'png';
        case 'MED_HARVARD_CT_MRI'
            vi_dir = '/Data3/cao/ZiHanCao/datasets/MedHarvard/xmu/CT-MRI/test/CT';
            test_ext = 'png';
        otherwise
            error('dataset_name %s is not supported\n', dataset_name)
    end

    switch dataset_name
        case 'ROADSCENE'
            ir_dir = '/Data3/cao/ZiHanCao/datasets/RoadSceneFusion/test/ir test';
        case 'TNO'
            ir_dir = '/Data3/cao/ZiHanCao/datasets/TNO/new_test_data/ir';
        case 'LLVIP'
            ir_dir = '/Data3/cao/ZiHanCao/datasets/LLVIP/data/infrared/test';
        case 'MSRS'
            ir_dir = '/Data3/cao/ZiHanCao/datasets/MSRS/test/ir';
        case 'M3FD'
            ir_dir = '/Data3/cao/ZiHanCao/datasets/M3FD/M3FD_Fusion/ir';
        case 'MED_HARVARD_SPECT_MRI'
            ir_dir = '/Data3/cao/ZiHanCao/datasets/MedHarvard/xmu/SPECT-MRI/test/MRI';
        case 'MED_HARVARD_PET_MRI'
            ir_dir = '/Data3/cao/ZiHanCao/datasets/MedHarvard/xmu/PET-MRI/test/MRI';
        case 'MED_HARVARD_CT_MRI'
            ir_dir = '/Data3/cao/ZiHanCao/datasets/MedHarvard/xmu/CT-MRI/test/MRI';
        otherwise
            msg = sprintf('dataset_name %s is not supported\n', dataset_name);
            error(msg)
    end

    if isfield(args, 'fused_ext')
        fused_ext = args.fused_ext;
    else
        fused_ext = 'png';
    end

    easy = test_mode_easy; %% easy=1 用于测试：EN, SF,SD,PSNR,MSE, MI, VIF, AG, CC, SCD, Qabf等指标； easy=0 用于测试：Nabf, SSIM, MS_SSIM, FMI_pixel, FMI_dct, FMI_w等指标
    if easy
        test_metrics = ["EN", "SF", "SD", "PSNR", "MSE", "MI", "VIF", "AG", "CC", "SCD", "Qabf", "SSIM"];
    else
        test_metrics = ["MS_SSIM", "Nabf", "FMI_pixel", "FMI_dct", "FMI_w"];
    end

    % excel file settings
    Method_name = method_name;
    if save_excel_file
        row_name1 = 'row1';
        row_data1 = 'row2';
        row = 'A';
        row_name = strrep(row_name1, 'row', row);
        row_data = strrep(row_data1, 'row', row);
    end

    % get the file path of vi and ir images
    ir_dir = fullfile(ir_dir); % 源图像A所在文件夹 此处是'Evaluation\Image\Source-Image\TNO\ir'
    vi_dir = fullfile(vi_dir); % 源图像B所在文件夹 此处是'Evaluation\Image\Source-Image\TNO\vi'
    Fused_dir = fullfile(fusion_dir); % 融合结果所在文件夹 此处是 'Evaluation\Image\Algorithm\SeAFusion_TNO'

    % get the name of ir/vis images names
    fileFolder = fullfile(ir_dir); % 源图像A所在文件夹 此处是'Evaluation\Image\Source-Image\TNO\ir'
    dirOutput = dir(fullfile(fileFolder,'*.*'));
    fileNames = {dirOutput.name};
    [m, num] = size(fileNames);

    % store the results of each image
    mpResults = {};


    % starting parallel pool
    parfor i = (1: num)
        results = processFusedImages(i, fileNames, ir_dir, vi_dir, Fused_dir, num, test_metrics, test_ext, fused_ext, rgb_test, Method_name, easy);
        mpResults{i} = results;
        fprintf('[%i/%i] finished %s\n', i, num, fileNames{i});
    end

    % close the parallel pool
    delete(gcp('nocreate'));

    % handle the messages and results
    EN_set = [];    
    SF_set = [];
    SD_set = [];
    PSNR_set = [];
    MSE_set = [];
    MI_set = [];
    VIF_set = [];
    AG_set = [];
    CC_set = [];
    SCD_set = []; 
    Qabf_set = [];
    SSIM_set = []; 
    MS_SSIM_set = [];
    Nabf_set = [];
    FMI_pixel_set = [];
    FMI_dct_set = []; 
    FMI_w_set = [];

    for i = 1: length(mpResults)
        result = mpResults{i};

        msgs = result.msgs;
        if isfield(result, 'metrics')
            metrics = result.metrics;
            has_metrics = 1;
        else
            has_metrics = 0;
        end

        % print the messages
    %     for j = 1: length(msgs)
    %         dualfprintf(fileID, msgs(j));
    %     end
        dualfprintf(fileID, msgs);

        if ~has_metrics
            continue
        end
        % print the metrics
        EN = metrics.EN;
        SF = metrics.SF;
        SD = metrics.SD;
        PSNR = metrics.PSNR;
        MSE = metrics.MSE;
        MI = metrics.MI;
        VIF = metrics.VIF;
        AG = metrics.AG;
        CC = metrics.CC;
        SCD = metrics.SCD;
        Qabf = metrics.Qabf;
        SSIM = metrics.SSIM;

        EN_set = [EN_set, EN];
        SF_set = [SF_set,SF];
        SD_set = [SD_set, SD];
        PSNR_set = [PSNR_set, PSNR];
        MSE_set = [MSE_set, MSE];
        MI_set = [MI_set, MI];
        VIF_set = [VIF_set, VIF];
        AG_set = [AG_set, AG];
        CC_set = [CC_set, CC];
        SCD_set = [SCD_set, SCD];
        Qabf_set = [Qabf_set, Qabf];
        SSIM_set = [SSIM_set, SSIM];
        
        if ~easy
            MS_SSIM = metrics.MS_SSIM;
            Nabf = metrics.Nabf;
            FMI_pixel = metrics.FMI_pixel;
            FMI_dct = metrics.FMI_dct;
            FMI_w = metrics.FMI_w;
            
            Nabf_set = [Nabf_set, Nabf];
            MS_SSIM_set = [MS_SSIM_set, MS_SSIM];
            FMI_pixel_set = [FMI_pixel_set, FMI_pixel];
            FMI_dct_set = [FMI_dct_set,FMI_dct];
            FMI_w_set = [FMI_w_set, FMI_w];
        end
    end


    % print the mean and std
    dualfprintf(fileID, '--------------------------------------------\n');
    dualfprintf(fileID, 'Fusion Method: %s\n', Method_name);
    dualfprintf(fileID, 'EN:    [%.4f, %.4f]\n', mean(EN_set), std(EN_set));
    dualfprintf(fileID, 'MI:    [%.4f, %.4f]\n', mean(MI_set), std(MI_set));
    dualfprintf(fileID, 'SD:    [%.4f, %.4f]\n', mean(SD_set), std(SD_set));
    dualfprintf(fileID, 'SF:    [%.4f, %.4f]\n', mean(SF_set), std(SF_set));
    dualfprintf(fileID, 'MSE:   [%.4f, %.4f]\n', mean(MSE_set), std(MSE_set));
    dualfprintf(fileID, 'PSNR:  [%.4f, %.4f]\n', mean(PSNR_set), std(PSNR_set));
    dualfprintf(fileID, 'VIF:   [%.4f, %.4f]\n', mean(VIF_set), std(VIF_set));
    dualfprintf(fileID, 'AG:    [%.4f, %.4f]\n', mean(AG_set), std(AG_set));
    dualfprintf(fileID, 'SCD:   [%.4f, %.4f]\n', mean(SCD_set), std(SCD_set));
    dualfprintf(fileID, 'CC:    [%.4f, %.4f]\n', mean(CC_set), std(CC_set));
    dualfprintf(fileID, 'Qabf:  [%.4f, %.4f]\n', mean(Qabf_set), std(Qabf_set));
    dualfprintf(fileID, 'SSIM:  [%.4f, %.4f]\n', mean(SSIM_set), std(SSIM_set));
    if ~easy
        dualfprintf(fileID, 'Nabf:       [%.4f, %.4f]\n', mean(Nabf_set), std(Nabf_set));
        dualfprintf(fileID, 'MS_SSIM:    [%.4f, %.4f]\n', mean(MS_SSIM_set), std(MS_SSIM_set));
        dualfprintf(fileID, 'FMI_pixel:  [%.4f, %.4f]\n', mean(FMI_pixel_set), std(FMI_pixel_set));
        dualfprintf(fileID, 'FMI_dct:    [%.4f, %.4f]\n', mean(FMI_dct_set), std(FMI_dct_set));
        dualfprintf(fileID, 'FMI_w:      [%.4f, %.4f]\n', mean(FMI_w_set), std(FMI_w_set));
    end
    fclose(fileID);


    if ~save_excel_file
        return
    end
    if exist(save_dir,'dir')==0
        mkdir(save_dir);
    end
    file_name = fullfile(save_dir, strcat('Metric_', Method_name, '.xlsx')); %存放Excel文件的文件名
    %% 将测试结果写入 Excel， 此处采用writetable， 第一行可能会有问题，算法名在第二行，评估结果从第三行开始
    if easy ==1
        SD_table = table(SD_set');
        PSNR_table = table(PSNR_set');
        MSE_table = table(MSE_set');
        MI_table = table(MI_set');
        VIF_table = table(VIF_set');
        AG_table = table(AG_set');
        CC_table = table(CC_set');
        SCD_table = table(SCD_set');
        EN_table = table(EN_set');
        Qabf_table = table(Qabf_set');
        SF_table = table(SF_set');
        SSIM_table = table(SSIM_set');
        method_name = cellstr(Method_name);
        method_table = table(method_name);

        writetable(SD_table,file_name,'Sheet','SD','Range',row_data);
        writetable(PSNR_table,file_name,'Sheet','PSNR','Range',row_data);
        writetable(MSE_table,file_name,'Sheet','MSE','Range',row_data);
        writetable(MI_table,file_name,'Sheet','MI','Range',row_data);
        writetable(VIF_table,file_name,'Sheet','VIF','Range',row_data);
        writetable(AG_table,file_name,'Sheet','AG','Range',row_data);
        writetable(CC_table,file_name,'Sheet','CC','Range',row_data);
        writetable(SCD_table,file_name,'Sheet','SCD','Range',row_data);
        writetable(EN_table,file_name,'Sheet','EN','Range',row_data);
        writetable(Qabf_table,file_name,'Sheet','Qabf','Range',row_data);
        writetable(SF_table,file_name,'Sheet','SF','Range',row_data);
        writetable(SSIM_table,file_name,'Sheet','SSIM','Range',row_data);

        writetable(method_table,file_name,'Sheet','SD','Range',row_name);
        writetable(method_table,file_name,'Sheet','PSNR','Range',row_name);
        writetable(method_table,file_name,'Sheet','MSE','Range',row_name);
        writetable(method_table,file_name,'Sheet','MI','Range',row_name);
        writetable(method_table,file_name,'Sheet','VIF','Range',row_name);
        writetable(method_table,file_name,'Sheet','AG','Range',row_name);
        writetable(method_table,file_name,'Sheet','CC','Range',row_name);
        writetable(method_table,file_name,'Sheet','SCD','Range',row_name);
        writetable(method_table,file_name,'Sheet','EN','Range',row_name);
        writetable(method_table,file_name,'Sheet','Qabf','Range',row_name);
        writetable(method_table,file_name,'Sheet','SF','Range',row_name);
        writetable(method_table,file_name,'Sheet','SSIM','Range',row_name);
    else    
        Nabf_table = table(Nabf_set');
        FMI_pixel_table = table(FMI_pixel_set');
        FMI_dct_table = table(FMI_dct_set');
        FMI_w_table = table(FMI_w_set');
        MS_SSIM_table = table(MS_SSIM_set');
        method_name = cellstr(Method_name);
        method_table = table(method_name);

        writetable(Nabf_table,file_name,'Sheet','Nabf','Range',row_data);
        writetable(FMI_pixel_table,file_name,'Sheet','FMI_pixel','Range',row_data);
        writetable(FMI_dct_table,file_name,'Sheet','FMI_dct','Range',row_data);
        writetable(FMI_w_table,file_name,'Sheet','FMI_w','Range',row_data);
        writetable(MS_SSIM_table,file_name,'Sheet','MS_SSIM','Range',row_data);

        writetable(method_table,file_name,'Sheet','Nabf','Range',row_name);
        writetable(method_table,file_name,'Sheet','FMI_pixel','Range',row_name);
        writetable(method_table,file_name,'Sheet','FMI_dct','Range',row_name);
        writetable(method_table,file_name,'Sheet','FMI_w','Range',row_name);
        writetable(method_table,file_name,'Sheet','MS_SSIM','Range',row_name);
        dualfprintf(fileID, 'saved excel file.')
    end

    fprintf('done\n')

    delete(gcp('nocreate'))
    fprintf('multiprocess pool closed.\n')

end


%% helper function to parse the args
function params = parseInputs(varargin)
    % Create an input parser object
    p = inputParser;
    
    % Allow any number of name-value pairs
    p.KeepUnmatched = true;
    
    % Parse the input arguments
    parse(p, varargin{:});
    
    % Return the parsed parameters
    params = p.Unmatched;
end

function dualfprintf(fileID, formatSpec, varargin)
    % 检查 fileID 是否有效
    if fileID < 0
        error('Invalid file identifier.');
    end

    % Print to terminal
    fprintf(formatSpec, varargin{:});
    % Print to file
    fprintf(fileID, formatSpec, varargin{:});
end



% data = 1: 10;
% results = {};

% parfor i = 1:10
%     results{i} = print_num(data(i));
% end

% delete(gcp('nocreate'));

% disp('Results:')
% for i = 1:10
%     disp(results{i}.num)
% end

% function result=print_num(num)
%     dualfprintf('%i\n',num)
%     result.num = num * num;
% end