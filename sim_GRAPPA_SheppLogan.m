%% Load Shepp Logan Data

h   = h5read('shepp-logan-32.h5', '/noncartesian');
d0  = permute(h.r + 1j*h.i, [2,3,1]);
N   = 128;
% Compress coils
Nc  = 12;
[u,s,v] = svd(reshape(d0,[],size(d0,3)),'econ');
d0  = reshape(u(:,1:Nc)*s(1:Nc,1:Nc), size(d0,1), size(d0,2), Nc);  % dims are now read, spoke, channel

% Set dead time gap
gap = 5;
d   = d0;
d(1:gap, :, :) = 0;

%% Apply 1D GRAPPA
Nsrc = 5;
Ncal = 8;
cal_src = zeros(Nsrc * Nc, Ncal);
cal_tgt = zeros(Nc, Ncal);
for is = 1:size(d, 2)
    for ig = gap:-1:1
        for ic = 1:Ncal
            cal_src(:, ic) = reshape(d(ig+2+ic:ig+1+Nsrc+ic,is,:), Nsrc * Nc, 1);
            cal_tgt(:, ic) = squeeze(d(ig+1+ic, is, :));
        end
        W = cal_tgt / cal_src;
        src = reshape(d(ig + 1:ig+Nsrc, is, :), Nsrc * Nc, 1);
        d(ig, is, :) = W * src;
    end
end

shell_plots(d0, d, gap, 16, 3);
