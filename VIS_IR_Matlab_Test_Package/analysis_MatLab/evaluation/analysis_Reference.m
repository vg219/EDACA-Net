function metrics = analysis_Reference(image_f,image_ir,image_vis, choose_metrics) %[EN, SF, SD, PSNR, MSE, MI, VIF, AG, CC, SCD, Qabf, Nabf, SSIM]

    metrics = struct();
    grey_level = 256;

    if length(size(image_f)) > 2
        [s1,s2,c_ir] = size(image_ir);
        [s1,s2,c_vis] = size(image_vis);
        rgbFlag = 1;
    else
        [s1, s2] = size(image_f);
        c_ir = 1;
        c_vis = 1;
        rgbFlag = 0;
    end
%     fprintf('rgbFlag: %d\n', rgbFlag)

    cSeq = c_ir + c_vis;
    imgSeq = zeros(s1, s2, cSeq);

    % may cause error
    imgSeq(:, :, 1:c_ir) = image_ir;
    imgSeq(:, :, c_ir+1:end) = image_vis;

    image1 = im2double(image_ir);
    image2 = im2double(image_vis);
    image_fused = im2double(image_f);

    % disp(size(image_fused))
    
    for name = choose_metrics
        if ismember("EN", name)
            metrics.EN = entropy(image_f);
        end
        if ismember("MI", choose_metrics)
            if rgbFlag
                MI1 = MI_evaluation(image_ir, image_vis(:,:,1), image_f(:,:,1), grey_level);
                MI2 = MI_evaluation(image_ir, image_vis(:,:,2), image_f(:,:,2), grey_level);
                MI3 = MI_evaluation(image_ir, image_vis(:,:,3), image_f(:,:,3), grey_level);
                metrics.MI = (MI1 + MI2 + MI3) / 3;
            else
                metrics.MI = MI_evaluation(image_ir, image_vis, image_f, grey_level);
            end

        end
        if ismember("PSNR", choose_metrics)
            % disp(size(image1))
            % disp(size(image2))
            % disp(size(image_fused))
            metrics.PSNR = PSNR_evaluation(image1,image2,image_fused);
            if rgbFlag
                metrics.PSNR = mean(metrics.PSNR, "all");
            end
        end
        if ismember("SSIM", choose_metrics)
            if ~rgbFlag
                metrics.SSIM = quality_assess2d(imgSeq / 255.0, image_fused);
            else
                ssim_s = [];
                for rgb_i = 1: 3
                    ssim_i = quality_assess2d(imgSeq / 255.0, image_fused(:, :, rgb_i));
                    ssim_s = [ssim_s, ssim_i];
                end
                metrics.SSIM = mean(ssim_s, "all");
            end
            % SSIM_a
            % SSIM1 = ssim(image_fused,image1);
            % SSIM2 = ssim(image_fused,image2);
            % SSIM = mef_ssim(imgSeq, image_f);
        end
        if ismember("MS_SSIM", choose_metrics)
            [metrics.MS_SSIM,t1,t2]= analysis_ms_ssim(imgSeq, image_f);
            % error('MS_SSIM not implemented')
        end
        if ismember("Qabf", choose_metrics)
            if ~rgbFlag
                metrics.Qabf = analysis_Qabf(image1, image2, image_fused);
            else
                Qabf_s = [];
                for i = 1: 3
                    Qabf_i = analysis_Qabf(image1, image2(:, :, i), image_fused(:, :, i));
                    Qabf_s = [Qabf_s, Qabf_i]; 
                end
                metrics.Qabf = mean(Qabf_s, "all");
            end
        end
        if ismember("SCD", choose_metrics)
            if ~rgbFlag
                metrics.SCD = analysis_SCD(image1, image2, image_fused);
            else
                SCD_s = [];
                for i = 1: 3
                    SCD_i = analysis_SCD(image1, image2(:, :, i), image_fused(:, :, i));
                    SCD_s = [SCD_s, SCD_i];
                end
                metrics.SCD = mean(SCD_s, "all");
            end
        end
        if ismember("VIF", choose_metrics)
            if ~rgbFlag
                metrics.VIF = vifp_mscale(image_ir, image_f) + vifp_mscale(image_vis, image_f);
            else
                VIF_s = [];
                for i = 1: 3
                    metrics.VIF = 0.5 * vifp_mscale(image_ir, image_f(:, :, i)) + 0.5 * vifp_mscale(image_vis(:, :, i), image_f(:, :, i));
                    VIF_s = [VIF_s, metrics.VIF];
                end
                metrics.VIF = mean(VIF_s, "all");
            end
        end
        if ismember("SD", choose_metrics)
            metrics.SD = SD_evaluation(double(image_f));
            if rgbFlag
                metrics.SD = mean(metrics.SD, "all");
            end
        end
        if ismember("AG", choose_metrics)
            metrics.AG = AG_evaluation(image_f);
        end
        if ismember("CC", choose_metrics)
            if ~rgbFlag
                metrics.CC = CC_evaluation(image1, image2, image_fused);
            else
                CC_s = [];
                for i = 1: 3
                    CC_i = CC_evaluation(image1, image2(:, :, i), image_fused(:, :, i));
                    CC_s = [CC_s, CC_i];
                end
                metrics.CC = mean(CC_s, "all");
            end
        end
        if ismember("SF", choose_metrics)
            if ~rgbFlag
                metrics.SF = SF_evaluation(image_fused);
            else
                SF_s = [];
                for i = 1: 3
                    SF_i = SF_evaluation(image_fused(:, :, i));
                    SF_s = [SF_s, SF_i];
                end
                metrics.SF = mean(SF_s, "all");
            end
        end
        if ismember("MSE", choose_metrics)
            metrics.MSE = MSE_evaluation(image1,image2,image_fused);
            if rgbFlag
                metrics.MSE = mean(metrics.MSE, "all");
            end
        end
        if ismember("Nabf", choose_metrics)
            metrics.Nabf = analysis_nabf(image_fused, image1, image2);
        end
        if ismember("FMI_pixel", choose_metrics)
            metrics.FMI_pixel = analysis_fmi(image1,image2,image_fused);
        end
        if ismember("FMI_dct", choose_metrics)
            metrics.FMI_dct = analysis_fmi(image1,image2,image_fused,'dct');
        end
        if ismember("FMI_w", choose_metrics)
            metrics.FMI_w = analysis_fmi(image1,image2,image_fused,'wavelet');
        end
        % convert uint8 -> double in metric function
        if ismember("NCIE", choose_metrics)
            metrics.NCIE = metricWang(image_ir,image_vis,image_f);
        end
        if ismember("Q_G", choose_metrics) %Xydeas and Petrovic
            metrics.Xydeas = metricXydeas(image_ir,image_vis,image_f);
        end
        if ismember("PWW", choose_metrics) % Q_M: 0-255 input
            metrics.PWW = metricPWW(image_ir,image_vis,image_f);
        end
        if ismember("Q_P", choose_metrics)
            metrics.Q_P = metricZhao(image_ir,image_vis,image_f);
        end
        if ismember("Q_S", choose_metrics)
            metrics.Q_S = metricPeilla(image_ir,image_vis,image_f, 1); %ssim.L = 255;
        end
        if ismember("Q_CV", choose_metrics)
            metrics.Q_CV = metricCV(image_ir,image_vis,image_f);
        end
        if ismember("Q_CB", choose_metrics)
            metrics.Q_CB = metricChenBlum(image_ir,image_vis,image_f);
        end
        if ismember("Q_C", choose_metrics) %ssim: L=255
            metrics.Q_C = metricCvejic(image_ir,image_vis,image_f, 2);
        end
        if ismember("Q_MI", choose_metrics) %uint8
            metrics.Q_MI=metricMI(image_ir,image_vis,image_f,1);
        end
        if ismember("Q_TE", choose_metrics) %uint8
            % Tsallis entropy $Q_{TE}$
            metrics.Q_TE=metricMI(image_ir,image_vis,image_f,3);
        end
        if ismember("Q_Y", choose_metrics)  %ssim: L=255
            % Yang $Q_Y$
            metrics.Q_Y =metricYang(image_ir,image_vis, image_f);
        end
    end
    
    end
    
    
    
    
    
    
    
    