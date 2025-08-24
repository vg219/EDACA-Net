function results=processFusedImages(idx, fileNames, ir_dir, vi_dir, Fused_dir, num, test_metrics, test_ext, fused_ext, rgb_test, Method_name, easy)
    msgs = [];
    results = struct();  % results of each image


    if (isequal(fileNames{idx}, '.') || isequal(fileNames{idx}, '..'))
        msg = sprintf('skip %s\n', fileNames{idx});
        msgs = [msgs, msg];

        results.msgs = msgs;

        return
    else
        msg = sprintf('--------------------------------------------\n');
        msgs = [msgs, msg];
        msg = sprintf('[%s/%s] Processing %s\n', string(idx), string(num), fileNames{idx});
        msgs = [msgs, msg];

        fileName_source_ir = fullfile(ir_dir, fileNames{idx});
        fileName_source_vi = fullfile(vi_dir, fileNames{idx}); 
        fileName_Fusion = fullfile(Fused_dir, fileNames{idx});
        ir_image = imread(fileName_source_ir);
        vi_image = imread(fileName_source_vi);

        % try to read the fused image with different extension
        fileName_Fusion = strrep(fileName_Fusion, test_ext, fused_ext);
        fused_image   = imread(fileName_Fusion);

        % check the image
        if size(ir_image, 3)>2
            ir_image = rgb2gray(ir_image);
        end

        if ~rgb_test
            if size(vi_image, 3)>2
                vi_image = rgb2gray(vi_image);
            end
    
            if size(fused_image, 3)>2
                fused_image = rgb2gray(fused_image);
            end
        end

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
            % fprintf('fused_image size: %d %d %d\n', size(fused_image, 1), size(fused_image, 2), size(fused_image, 3))
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

            if ~easy
                MS_SSIM = metrics.MS_SSIM;
                Nabf = metrics.Nabf;
                FMI_pixel = metrics.FMI_pixel;
                FMI_dct = metrics.FMI_dct;
                FMI_w = metrics.FMI_w;
            end

        else
            msg = sprintf('unsucessful!');
            msgs = [msgs, msg];
            msg = sprintf(fileName_Fusion);
            msgs = [msgs, msg];

            error('runtime error, meet the analysis_Reference error')
            return
        end

        msg = sprintf('Fusion Method: %s, Image Name: %s\n', Method_name, fileNames{idx});
        msgs = [msgs, msg];

        %% print all metrics
        if easy
            msg = sprintf('EN = %.4f\n', EN);
            msgs = [msgs, msg];
            msg = sprintf('MI = %.4f\n', MI);
            msgs = [msgs, msg];
            msg = sprintf('SD = %.4f\n', SD);
            msgs = [msgs, msg];
            msg = sprintf('SF = %.4f\n', SF);
            msgs = [msgs, msg];
            msg = sprintf('MSE = %.4f\n', MSE);
            msgs = [msgs, msg];
            msg = sprintf('PSNR = %.4f\n', PSNR);
            msgs = [msgs, msg];
            msg = sprintf('VIF = %.4f\n', VIF);
            msgs = [msgs, msg];
            msg = sprintf('AG = %.4f\n', AG);
            msgs = [msgs, msg];
            msg = sprintf('SCD = %.4f\n', SCD);
            msgs = [msgs, msg];
            msg = sprintf('CC = %.4f\n', CC);
            msgs = [msgs, msg];
            msg = sprintf('Qabf = %.4f\n', Qabf);
            msgs = [msgs, msg];
            msg = sprintf('SSIM = %.4f\n', SSIM);
            msgs = [msgs, msg];
        else
            msg = sprintf('Nabf = %.4f\n', Nabf);
            msgs = [msgs, msg];
            msg = sprintf('MS_SSIM = %.4f\n', MS_SSIM);
            msgs = [msgs, msg];
            msg = sprintf('FMI_pixel = %.4f\n', FMI_pixel);
            msgs = [msgs, msg];
            msg = sprintf('FMI_dct = %.4f\n', FMI_dct);
            msgs = [msgs, msg];
            msg = sprintf('FMI_w = %.4f\n', FMI_w);
            msgs = [msgs, msg];
        end 
    end
    results.msgs = msgs;
    results.metrics = metrics;
end