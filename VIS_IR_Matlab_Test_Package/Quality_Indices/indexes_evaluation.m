%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Description: 
%           Reduced resolution quality indexes. 
% 
% Interface:
%           [Q_index, SAM_index, ERGAS_index, sCC, Q2n_index] = indexes_evaluation(I_F,I_GT,ratio,L,Q_blocks_size,flag_cut_bounds,dim_cut,th_values)
%
% Inputs:
%           I_F:                Fused Image;
%           I_GT:               Ground-Truth image;
%           ratio:              Scale ratio between MS and PAN. Pre-condition: Integer value;
%           L:                  Image radiometric resolution; 
%           Q_blocks_size:      Block size of the Q-index locally applied;
%           flag_cut_bounds:    Cut the boundaries of the viewed Panchromatic image;
%           dim_cut:            Define the dimension of the boundary cut;
%           th_values:          Flag. If th_values == 1, apply an hard threshold to the dynamic range.
%
% Outputs:
%           Q_index:            Q index;
%           SAM_index:          Spectral Angle Mapper (SAM) index;
%           ERGAS_index:        Erreur Relative Globale Adimensionnelle de Synth�se (ERGAS) index;
%           sCC:                spatial Correlation Coefficient between fused and ground-truth images;
%           Q2n_index:          Q2n index.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [psnr, ssim, Q_index, SAM_index, ERGAS_index, cc, sCC, Q2n_index, rmse] = indexes_evaluation(I_F,I_GT,ratio,L,Q_blocks_size,flag_cut_bounds,dim_cut,th_values,maxvalue)

if max(I_GT(:))>1
    I_GT = I_GT/maxvalue;
    I_F = I_F/maxvalue;
end

if flag_cut_bounds
    I_GT = I_GT(dim_cut:end-dim_cut,dim_cut:end-dim_cut,:);
    I_F = I_F(dim_cut:end-dim_cut,dim_cut:end-dim_cut,:);
end

if th_values
    I_F(I_F > 2^L) = 2^L;
    I_F(I_F < 0) = 0;
end

dim = ndims(I_GT);
[H, W, C, F] = size(I_GT);


if dim == 4
    I_GT = reshape(I_GT, [H, W, C*F]);
    I_F = reshape(I_F, [H, W, C*F]);
end

cd Quality_Indices

metrics = quality_assess(I_GT, I_F);
psnr=metrics(1);
ssim=metrics(2);
Q2n_index = q2n(I_GT*maxvalue,I_F*maxvalue,Q_blocks_size,Q_blocks_size);
Q_index = Q(I_GT,I_F,1);
SAM_index = SAM(I_GT,I_F);
ERGAS_index = ERGAS(I_GT,I_F,ratio);
sCC = SCC(I_F,I_GT);
cc = CC_a(I_F,I_GT);
cc = mean(cc);
rmse = RMSE_a(I_F,I_GT);

cd ..

end