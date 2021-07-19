%% Setup path
addpath('utils');

%% Load Shepp Logan Data

h   = h5read('shepp-logan-32.h5', '/noncartesian');
d0  = permute(h.r + 1j*h.i, [2,3,1]);
k0  = pi*permute(h5read('shepp-logan-32.h5', '/trajectory'), [2,3,1]);
N   = 128;

% Compress coils
Nc  = 12;
[u,s,v] = svd(reshape(d0,[],size(d0,3)),'econ');
d0  = reshape(u(:,1:Nc)*s(1:Nc,1:Nc), size(d0,1), size(d0,2), Nc); 

% Set dead time gap
gap = 5;
k   = k0(gap+1:end,:,:);
d   = d0(gap+1:end,:,:);

%% Truncate k-space for recon of only central portion

% Truncate data
N_reduced = 32;
dr  = reshape(d(1:N_reduced-2*gap, :, :),[],Nc);

% Redefine forward operator for truncated data
E   = xfm_NUFFT([N_reduced,N_reduced,N_reduced,1],[],[],(N/N_reduced)*reshape(k(1:N_reduced-2*gap,:,:),[],1,3));

%% Reconstruction

% Solves min_x ||Ex-d||^2_2 such that rank(H(x)) = r

kernel  = [5,5,5];
r       = 250;
rho     = 1E6;
niters  = 100;

x = SLR.ADMM(reshape(dr,[],Nc).*E.w, E, kernel, r, rho, niters);

%% Evaluate Results
E   = xfm_NUFFT([N_reduced,N_reduced,N_reduced,1],[],[],(N/N_reduced)*reshape(k0(1:gap,:,:),[],1,3), 'wi',1, 'PSF',[]);
d2  = zeros(E.dsize(1),Nc);
for i = 1:Nc
    d2(:,i) = E*x(:,:,:,i);
end

d2  = reshape(d2,gap,[],Nc);
shell_plots(d0, d2, gap, 16, 3);
