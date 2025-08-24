function runDir(fusion_dir, dataset_name, method_name, rgb_test, test_mode_easy, varargin)
    % vargin:
    % vi_dir: str
    % ir_dir: str
    % fused_ext: str. Default: jpg

    addpath('analysis_MatLab/evaluation/');
    addpath('Quality_Indices/');

    % metrics saved dir
    dataset_name = upper(dataset_name);
    save_dir = strcat('Metric/', dataset_name); %存放Excel结果的文件夹

    % parse the args
    args = parseInputs(varargin{:});

    % open a file to log
    if isfield(args, 'file_name')
        file_id = fopen(args.file_name, 'w');
    else
        file_id = fopen('logs/VIS_IR_log.txt', 'w');
    end

    dualfprintf(file_id, 'save_dir: %s\n', save_dir);

    if isfield(args, 'fused_ext')
        fused_ext = args.fused_ext;
    else
        fused_ext = 'jpg';
    end

    if isfield(args, 'test_ext')
        test_ext = args.test_ext;
    else
        test_ext = 'jpg';
    end

    if isfield(args, 'vi_dir')
        vi_dir = args.vi_dir;
    else  % use dataset_name to get the vi dir
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
    end
    
    if isfield(args, 'ir_dir')
        ir_dir = args.ir_dir;
    else  
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
    end
    
    easy = test_mode_easy; %% easy=1 用于测试：EN, SF,SD,PSNR,MSE, MI, VIF, AG, CC, SCD, Qabf等指标； easy=0 用于测试：Nabf, SSIM, MS_SSIM, FMI_pixel, FMI_dct, FMI_w等指标
    if easy
        test_metrics = ["EN", "SF", "SD", "PSNR", "MSE", "MI", "VIF", "AG", "CC", "SCD", "Qabf", "SSIM"];
    else
        test_metrics = ["MS_SSIM", "Nabf", "FMI_pixel", "FMI_dct", "FMI_w"];
    end

    row_name1 = 'row1';
    row_data1 = 'row2';
    Method_name = method_name;
    row = 'A';
    row_name = strrep(row_name1, 'row', row);
    row_data = strrep(row_data1, 'row', row);
    
    % get the name of ir/vis images names
    fileFolder=fullfile(fusion_dir); % 源图像A所在文件夹 此处是'Evaluation\Image\Source-Image\TNO\ir'
    dirOutput=dir(fullfile(fileFolder,'*.*'));
    fileNames = {dirOutput.name};
    [m, num] = size(fileNames);

    % ir, vi, and fused image directories
    ir_dir = fullfile(ir_dir); % 源图像A所在文件夹 此处是'Evaluation\Image\Source-Image\TNO\ir'
    vi_dir = fullfile(vi_dir); % 源图像B所在文件夹 此处是'Evaluation\Image\Source-Image\TNO\vi'
    Fused_dir = fullfile(fusion_dir); % 融合结果所在文件夹 此处是 'Evaluation\Image\Algorithm\SeAFusion_TNO'

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

    for j = 1:num
        if (isequal(fileNames{j}, '.') || isequal(fileNames{j}, '..'))
            dualfprintf(file_id, 'skip %s\n', fileNames{j});
            continue;
        else
            dualfprintf(file_id, '--------------------------------------------\n');
            dualfprintf(file_id, '[%s/%s] Processing %s\n', string(j), string(num), fileNames{j});

            fileName_source_ir = fullfile(ir_dir, fileNames{j});
            fileName_source_vi = fullfile(vi_dir, fileNames{j});
            fileName_Fusion = fullfile(Fused_dir, fileNames{j});

            % try to read the fused image with different extension
            fileName_source_ir = strrep(fileName_source_ir, fused_ext, test_ext);
            fileName_source_vi = strrep(fileName_source_vi, fused_ext, test_ext);

            ir_image = imread(fileName_source_ir);
            vi_image = imread(fileName_source_vi);
            fused_image   = imread(fileName_Fusion);
            
            if size(ir_image, 3)>2
                ir_image = rgb2gray(ir_image);
            end

            % dualfprintf('ir_image size: %d %d %d\n', size(ir_image, 1), size(ir_image, 2), size(ir_image, 3))
            % dualfprintf('vi_image size: %d %d %d\n', size(vi_image, 1), size(vi_image, 2), size(vi_image, 3))
            % dualfprintf('fused_image size: %d %d %d \n', size(fused_image, 1), size(fused_image, 2), size(fused_image, 3))
            
            if ~rgb_test
                if size(vi_image, 3)>2
                    vi_image = rgb2gray(vi_image);
                end
        
                if size(fused_image, 3)>2
                    fused_image = rgb2gray(fused_image);
                end
            end

            % if length(size(fused_image)) > 3 && length(size(ir_image)) < 3
            %     % ensure the ir image to be 3-dim
            %     ir_image = cat(3, ir_image, ir_image, ir_image);
            % end

            if size(fused_image) > 2
                [m, n, ~] = size(fused_image);
            else
                [m, n] = size(fused_image);
            end

            ir_size = size(ir_image);
            vi_size = size(vi_image);
            fusion_size = size(fused_image);

            % may resize vi and ir to fused size
            if ir_size(1) ~= m || ir_size(2) ~= n
                ir_image = imresize(ir_image, [m, n]);
            end

            if vi_size(1) ~= m || vi_size(2) ~= n
                vi_image = imresize(vi_image, [m, n]);
            end

            if length(ir_size) <= 3 && length(vi_size) <= 3  % assert to be an image
                % dualfprintf('fused_image size: %d %d %d\n', size(fused_image, 1), size(fused_image, 2), size(fused_image, 3))
                metrics = analysis_Reference(fused_image,ir_image,vi_image,test_metrics);

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

            else
                disp('unsucessful!')
                disp(fileName_Fusion)
            end
    
            dualfprintf(file_id, 'Fusion Method: %s, Image Name: %s\n', Method_name, fileNames{j});

            %% print all metrics
            if easy
                dualfprintf(file_id, 'EN = %.4f\n', EN);
                dualfprintf(file_id, 'MI = %.4f\n', MI);
                dualfprintf(file_id, 'SD = %.4f\n', SD);
                dualfprintf(file_id, 'SF = %.4f\n', SF);
                dualfprintf(file_id, 'MSE = %.4f\n', MSE);
                dualfprintf(file_id, 'PSNR = %.4f\n', PSNR);
                dualfprintf(file_id, 'VIF = %.4f\n', VIF);
                dualfprintf(file_id, 'AG = %.4f\n', AG);
                dualfprintf(file_id, 'SCD = %.4f\n', SCD);
                dualfprintf(file_id, 'CC = %.4f\n', CC);
                dualfprintf(file_id, 'Qabf = %.4f\n', Qabf);
                dualfprintf(file_id, 'SSIM = %.4f\n', SSIM);
            else
                dualfprintf(file_id, 'Nabf = %.4f\n', Nabf);
                dualfprintf(file_id, 'MS_SSIM = %.4f\n', MS_SSIM);
                dualfprintf(file_id, 'FMI_pixel = %.4f\n', FMI_pixel);
                dualfprintf(file_id, 'FMI_dct = %.4f\n', FMI_dct);
                dualfprintf(file_id, 'FMI_w = %.4f\n', FMI_w);
            end
        end
    end
    % print the mean and std
    dualfprintf(file_id, '--------------------------------------------\n');
    dualfprintf(file_id, 'Fusion Method: %s\n', Method_name);
    dualfprintf(file_id, 'EN:    [%.4f, %.4f]\n', mean(EN_set), std(EN_set));
    dualfprintf(file_id, 'MI:    [%.4f, %.4f]\n', mean(MI_set), std(MI_set));
    dualfprintf(file_id, 'SD:    [%.4f, %.4f]\n', mean(SD_set), std(SD_set));
    dualfprintf(file_id, 'SF:    [%.4f, %.4f]\n', mean(SF_set), std(SF_set));
    dualfprintf(file_id, 'MSE:   [%.4f, %.4f]\n', mean(MSE_set), std(MSE_set));
    dualfprintf(file_id, 'PSNR:  [%.4f, %.4f]\n', mean(PSNR_set), std(PSNR_set));
    dualfprintf(file_id, 'VIF:   [%.4f, %.4f]\n', mean(VIF_set), std(VIF_set));
    dualfprintf(file_id, 'AG:    [%.4f, %.4f]\n', mean(AG_set), std(AG_set));
    dualfprintf(file_id, 'SCD:   [%.4f, %.4f]\n', mean(SCD_set), std(SCD_set));
    dualfprintf(file_id, 'CC:    [%.4f, %.4f]\n', mean(CC_set), std(CC_set));
    dualfprintf(file_id, 'Qabf:  [%.4f, %.4f]\n', mean(Qabf_set), std(Qabf_set));
    dualfprintf(file_id, 'SSIM:  [%.4f, %.4f]\n', mean(SSIM_set), std(SSIM_set));
    if ~easy
        dualfprintf(file_id, 'Nabf:       [%.4f, %.4f]\n', mean(Nabf_set), std(Nabf_set));
        dualfprintf(file_id, 'MS_SSIM:    [%.4f, %.4f]\n', mean(MS_SSIM_set), std(MS_SSIM_set));
        dualfprintf(file_id, 'FMI_pixel:  [%.4f, %.4f]\n', mean(FMI_pixel_set), std(FMI_pixel_set));
        dualfprintf(file_id, 'FMI_dct:    [%.4f, %.4f]\n', mean(FMI_dct_set), std(FMI_dct_set));
        dualfprintf(file_id, 'FMI_w:      [%.4f, %.4f]\n', mean(FMI_w_set), std(FMI_w_set));
    end
    fclose(file_id);

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
    
    end
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