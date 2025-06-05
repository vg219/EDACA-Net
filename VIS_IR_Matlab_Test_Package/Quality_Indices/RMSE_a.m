function out = RMSE_a(ref,tar,mask)
%--------------------------------------------------------------------------
% Root mean squared error (RMSE)
%
% USAGE
%   out = RMSE(ref,tar,mask)
%
% INPUT
%   ref : reference HS data (rows,cols,bands)
%   tar : target HS data (rows,cols,bands)
%   mask: binary mask (rows,cols) (optional)
%
% OUTPUT
%   out : RMSE (scalar)
%
%--------------------------------------------------------------------------
[rows,cols,bands] = size(ref);
if nargin == 2
out = (sum(sum(sum((tar-ref).^2)))/(rows*cols*bands)).^0.5;
else
    out.rmse_map = (sum((tar-ref).^2,3)/(bands)).^0.5;
    ref = reshape(ref,[],bands);
    tar = reshape(tar,[],bands);
    out.ave = (sum(sum((ref(mask~=0,:)-tar(mask~=0,:)).^2))/(sum(sum(mask~=0))*bands)).^0.5;
% figure; imagesc(rmse_map,[0 0.05]); axis image
end