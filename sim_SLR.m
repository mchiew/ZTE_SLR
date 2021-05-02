%% Setup path
addpath('utils');

%% Generate Simualted Data

% Load Sensitivities
load('sens_2D.mat');
Nc  = size(sens,4);

% Define numerical phantom
N   = 110;
x0  = phantom(N).*sens;

%  Use 2D golden ratio radial ordering
M   = 346; % 110*pi
gr  = (sqrt(5)+1)/2;
k   = zeros(N, M, 2);
for i = 1:M
    k(:,i,1) = linspace(0, pi*real(exp(1j*(i-1)*pi/gr)),N);
    k(:,i,2) = linspace(0, pi*imag(exp(1j*(i-1)*pi/gr)),N);
end

% Set dead time gap
gap = 4;
k   = k(gap+1:end,:,:);

% Define forward operator (requires Fessler IRT toolbox)
E   = xfm_NUFFT([N,N,1,1],[],[],reshape(k(:,:,:),[],1,2));

% Create noisy k-space samples
d = zeros(E.dsize(1),8);
for i = 1:8
    d(:,i) = E*x0(:,:,1,i)+0.01*(randn(E.dsize(1),1)+1j*randn(E.dsize(1),1));
end

%% Reconstruction

% Solves min_x ||Ex-d||^2_2 such that rank(H(x)) = r

% Define parameters
r       = 32; % rank constraint
kernel  = [5,5];
dims    = [N,N];
step    = 1;
iters   = 100;

% Initialise output
x   = zeros(N,N,Nc);

% Pre-comupte E'*d
dx  = x;
for c = 1:Nc
    dx(:,:,c)   = reshape(E'*d(:,c),dims);
end

for i = 1:iters
    % Compute data consistency gradient step 
    % x <-- x - step*E'(Ex-d)
    for c = 1:Nc
        x(:,:,c)    = x(:,:,c) - step*(E.mtimes2(x(:,:,c)) - dx(:,:,c));
    end
    
    % Fourier transform data and form Hankel matrix
    % z <-- H(F(x))
    y   = E.fftfn(x,1:2);
    z   = [];
    for c = 1:Nc
        z   = cat(1,z,Hankel_fwd(y(:,:,c), kernel, dims));
    end
    
    % Truncate Hankel matrix to rank r
    % z <-- truncate_rank(z,r)
    [u,s,v] = svd(z, 'econ');
    z       = u(:,1:r)*s(1:r,1:r)*v(:,1:r)';
    
    % Pseudo-invert Hankel transform
    % x <-- F_inv(pinv_H(z))
    z   = permute(reshape(z,[],Nc,size(z,2)),[1,3,2]);
    for c = 1:Nc
        y(:,:,c)    = Hankel_pinv(z(:,:,c), kernel, dims);
    end
    x = E.ifftfn(y,1:2);
end

%% Display Result
imshow([sos(x) sos(x0)],[0,1]);

%% Helper Functions

function h = Hankel_fwd(x, kernel, dims)
    Nx  = dims(1);
    Ny  = dims(2);
        
    h   = zeros(prod(kernel), prod([Nx,Ny]-kernel+1));

    idx = 0;
    for kx = 1:Nx - kernel(1) + 1
        for ky = 1:Ny - kernel(2) + 1
            idx = idx + 1;
            h(:, idx) = reshape(x(kx:kx+kernel(1)-1, ky:ky+kernel(2)-1),prod(kernel),1);
        end
    end
end

function x = Hankel_pinv(h, kernel, dims)
    Nx  = dims(1);
    Ny  = dims(2);
        
    x   = zeros(dims);
    m   = zeros(dims);
    
    idx = 0;
    for kx = 1:Nx - kernel(1) + 1
        for ky = 1:Ny - kernel(2) + 1
            idx = idx + 1;
            x(kx:kx+kernel(1)-1,ky:ky+kernel(2)-1) = x(kx:kx+kernel(1)-1,ky:ky+kernel(2)-1) + reshape(h(:,idx),kernel);
            m(kx:kx+kernel(1)-1,ky:ky+kernel(2)-1) = m(kx:kx+kernel(1)-1,ky:ky+kernel(2)-1) + 1;
        end
    end
    x   = x./m;
end

function x = sos(x)
    x = sqrt(sum(abs(x).^2,ndims(x)));
end