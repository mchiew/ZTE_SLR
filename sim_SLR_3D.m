%% Setup path
addpath('utils');

%% Generate Simulated Data

% Load Sensitivities
load('sens_3D.mat');
Nc  = size(sens,4);

% Define numerical phantom
N   = 110;
x0  = repmat(phantom(N),1,1,110).*sens;

%  Use 2D golden ratio radial ordering
M   = round(110*110*pi); % 110**110*pi
gr  = (sqrt(5)+1)/2;
m   = sphere_traj(M);
k0  = zeros(110,M,3);
for i = 1:M
    k0(:,i,1) = linspace(0,pi*m(i,1),N);
    k0(:,i,2) = linspace(0,pi*m(i,2),N);
    k0(:,i,3) = linspace(0,pi*m(i,3),N);
end

% Set dead time gap
gap = 4;
k   = k0(gap+1:end,:,:);

% Define forward operator (requires Fessler IRT toolbox)
E0   = xfm_NUFFT([N,N,N,1],[],[],reshape(k(:,:,:),[],1,3), 'Jd', [3,3,3]);

% Create noisy k-space samples
d = zeros(E0.dsize(1),Nc);
for i = 1:Nc
    d(:,i) = E0*x0(:,:,:,i)+0.01*(randn(E0.dsize(1),1)+1j*randn(E0.dsize(1),1));
end

%% Truncate k-space for recon of only central portion

% Truncate data
N_reduced = 32;
dr  = reshape(d,N-gap, [], Nc);
dr  = reshape(dr(1:N_reduced, :, :),[],Nc);

% Redefine forward operator for truncated data
E   = xfm_NUFFT([N_reduced,N_reduced,N_reduced,1],[],[],(N/N_reduced)*reshape(k(1:N_reduced,:,:),[],1,3), 'Jd', [3,3,3]);

%% Reconstruction

% Solves min_x ||Ex-d||^2_2 such that rank(H(x)) = r

% Define parameters
r       = 128; % rank constraint
kernel  = [5,5,5];
dims    = [N_reduced,N_reduced,N_reduced];
step    = 1;
iters   = 100;

% Initialise output
x   = zeros(N_reduced,N_reduced,N_reduced,Nc);

% Pre-comupte E'*d
dx  = x;
for c = 1:Nc
    dx(:,:,:,c)   = reshape(E'*dr(:,c),dims);
end

for i = 1:iters
    % Compute data consistency gradient step 
    % x <-- x - step*E'(Ex-d)
    for c = 1:Nc
        x(:,:,:,c)    = x(:,:,:,c) - step*(E.mtimes2(x(:,:,:,c)) - dx(:,:,:,c));
    end
    
    % Fourier transform data and form Hankel matrix
    % z <-- H(F(x))
    y   = E.fftfn(x,1:3);
    z   = zeros(Nc*prod(kernel),prod([N_reduced,N_reduced,N_reduced]-kernel+1));
    for c = 1:Nc
        z((c-1)*prod(kernel)+1:c*prod(kernel),:) = Hankel_fwd(y(:,:,:,c), kernel, dims);
    end
    
    % Truncate Hankel matrix to rank r
    % z <-- truncate_rank(z,r)
    [u,s,v] = svd(z, 'econ');
    z       = u(:,1:r)*s(1:r,1:r)*v(:,1:r)';
    
    % Pseudo-invert Hankel transform
    % x <-- F_inv(pinv_H(z))
    z   = permute(reshape(z,[],Nc,size(z,2)),[1,3,2]);
    for c = 1:Nc
        y(:,:,:,c)    = Hankel_pinv(z(:,:,c), kernel, dims);
    end
    x = E.ifftfn(y,1:3);
end

%% Evaluate Results
E   = xfm_NUFFT([N_reduced,N_reduced,N_reduced,1],[],[],(N/N_reduced)*reshape(k0(1:gap,:,:),[],1,3), 'Jd', [3,3,3], 'wi',1, 'PSF',[]);
d2  = zeros(E.dsize(1),Nc);
for i = 1:Nc
    d2(:,i) = E*x(:,:,:,i)*(N/N_reduced)^(3/2);
end

E0  = xfm_NUFFT([N,N,N,1],[],[],reshape(k0(1:gap,:,:),[],1,3), 'Jd', [3,3,3], 'wi',1, 'PSF',[]);
d0  = zeros(E0.dsize(1),Nc);
for i = 1:Nc
    d0(:,i) = E0*x0(:,:,:,i);
end

d0  = reshape(d0,gap,[],Nc);
d2  = reshape(d2,gap,[],Nc);
%% Display Result for first 4 coils
clf;

% subsample spokes for display
idx = 1:10:size(d0,2);

for i = 1:gap
    for j = 1:4
        % Magnitude
        subplot(2*gap, 4, 2*(i-1)*gap + j);
        hold on;
        scatter(1:length(idx),abs(d0(gap+1-i,idx,j)), 5, 'filled');
        scatter(1:length(idx),abs(d2(gap+1-i,idx,j)), 5, 'filled');
        ylim([0 6E4]);
        
        if j==1
            ylabel(sprintf('k%d Mag',gap+1-i));
        end
        if i==1
            title(sprintf('Coil %d',j));
        end
        
        % Phase
        subplot(2*gap, 4, 2*(i-1)*gap + gap + j);
        hold on;
        scatter(1:length(idx),angle(d0(gap+1-i,idx,j)), 5, 'filled');
        scatter(1:length(idx),angle(d2(gap+1-i,idx,j)), 5, 'filled');
        ylim([-pi pi]);
        
        if j==1
            ylabel(sprintf('k%d Phs',gap+1-i));
        end
    end
end

%% Helper Functions

function h = Hankel_fwd(x, kernel, dims)
    Nx  = dims(1);
    Ny  = dims(2);
    Nz  = dims(3);
        
    h   = zeros(prod(kernel), prod([Nx,Ny,Nz]-kernel+1));

    idx = 0;
    for kx = 1:Nx - kernel(1) + 1
        for ky = 1:Ny - kernel(2) + 1
            for kz = 1:Nz - kernel(3) + 1
                idx = idx + 1;
                h(:, idx) = reshape(x(kx:kx+kernel(1)-1, ky:ky+kernel(2)-1,kz:kz+kernel(3)-1),prod(kernel),1);
            end
        end
    end
end

function x = Hankel_pinv(h, kernel, dims)
    Nx  = dims(1);
    Ny  = dims(2);
    Nz  = dims(3);
        
    x   = zeros(dims);
    m   = zeros(dims);
    
    idx = 0;
    for kx = 1:Nx - kernel(1) + 1
        for ky = 1:Ny - kernel(2) + 1
            for kz = 1:Nz - kernel(3) + 1
                idx = idx + 1;
                x(kx:kx+kernel(1)-1,ky:ky+kernel(2)-1,kz:kz+kernel(3)-1) = x(kx:kx+kernel(1)-1,ky:ky+kernel(2)-1,kz:kz+kernel(3)-1) + reshape(h(:,idx),kernel);
                m(kx:kx+kernel(1)-1,ky:ky+kernel(2)-1,kz:kz+kernel(3)-1) = m(kx:kx+kernel(1)-1,ky:ky+kernel(2)-1,kz:kz+kernel(3)-1) + 1;
            end
        end
    end
    x   = x./m;
end

function x = sos(x)
    x = sqrt(sum(abs(x).^2,ndims(x)));
end

function traj = sphere_traj(N)
    traj = zeros(N,3);
    for n = 1:N
        z = (2*n-N-1)/N;
        traj(n,1) = cos(sqrt(N*pi)*asin(z))*sqrt(1-z^2);
        traj(n,2) = sin(sqrt(N*pi)*asin(z))*sqrt(1-z^2);
        traj(n,3) = z;
    end
end