classdef SLR
        
    methods (Static)    
        
        function h = h_fwd(x, dims, kernel)
                                   
            Nx  = dims(1);
            Ny  = dims(2);
            Nz  = dims(3);
            Nc  = dims(4);

            h   = zeros(prod(kernel), Nc, prod([Nx,Ny,Nz]-kernel+1));

            idx = 0;
            for kx = 1:Nx - kernel(1) + 1
                for ky = 1:Ny - kernel(2) + 1
                    for kz = 1:Nz - kernel(3) + 1
                        idx = idx + 1;
                        h(:, :, idx) = reshape(x(kx:kx+kernel(1)-1, ky:ky+kernel(2)-1,kz:kz+kernel(3)-1,:),prod(kernel),Nc);
                    end
                end
            end
            
            h = reshape(h, prod(kernel)*Nc, []);
            
        end
        
        function x = h_adj(h, dims, kernel)   
           
            Nx  = dims(1);
            Ny  = dims(2);
            Nz  = dims(3);
            Nc  = dims(4);

            x   = zeros(dims);
            
            h = reshape(h, prod(kernel), Nc, []);

            idx = 0;
            for kx = 1:Nx - kernel(1) + 1
                for ky = 1:Ny - kernel(2) + 1
                    for kz = 1:Nz - kernel(3) + 1
                        idx = idx + 1;
                        x(kx:kx+kernel(1)-1,ky:ky+kernel(2)-1,kz:kz+kernel(3)-1,:) = x(kx:kx+kernel(1)-1,ky:ky+kernel(2)-1,kz:kz+kernel(3)-1,:) + reshape(h(:,:,idx),[kernel Nc]);
                    end
                end
            end

        end
        
        function sz = h_size(dims, kernel)
            sz  =   [prod(dims-kernel+1) prod(kernel)];
        end
        
        function N = h_norm(dims, kernel)
            U   =   [1:kernel(1) kernel(1)*ones(1,dims(1)-2*kernel(1)) kernel(1):-1:1];
            V   =   [1:kernel(2) kernel(2)*ones(1,dims(2)-2*kernel(2)) kernel(2):-1:1];
            W   =   [1:kernel(3) kernel(3)*ones(1,dims(3)-2*kernel(3)) kernel(3):-1:1];
            N   =   reshape(U,[],1,1).*reshape(V,1,[],1).*reshape(W,1,1,[]);
        end
        
        
        function x = ADMM(d, E, kernel, r, p, niters)
            if nargin < 5
                niters = 100;
            end
            
            dims = [E.Nd size(d,2)];
            
            % precompute first term of RHS of x-update step
            xd = zeros(dims);
            for c = 1:dims(4)
                xd(:,:,:,c) = reshape(E'*d(:,c),E.Nd);
            end
            
            % ADMM constants and initialisation
            p = 1E-4;
            x = xd;
            z = SLR.h_fwd(E.fftfn(x,1:3), dims, kernel);
            u = 0*z;
            
            % get normalisation factor
            N = SLR.h_norm(dims, kernel);
            
            % ADMM iterations
            for i = 1:niters               
                
                % x-update
                [x, ~]  = pcg(@(x)x_helper(x, E, N, p, dims), reshape(xd + (p/2)*E.ifftfn(SLR.h_adj(z-u, dims, kernel),1:3),[],1), 1E-4, 100);
                x       = reshape(x, dims);

                % z-update
                H       = SLR.h_fwd(E.fftfn(x,1:3), dims, kernel);
                [V,~]   = SLR.half_SVD(H + u);
                z       = V(:,1:r)*(V(:,1:r)'*(H+u));
                
                % u-update
                u       = u + H - z;
                
            end
            
            
            function y = x_helper(x, E, N, p, dims)
                x = reshape(x, dims);
                y = x;
                for ii = 1:dims(4)
                    y(:,:,:,ii) = E.mtimes2(x(:,:,:,ii));
                end
                y = reshape(y + (p/2)*E.ifftfn(N.*E.fftfn(x,1:3),1:3),[],1);
            end
        end
        
        function [U,S] = half_SVD(X)

        [U,D]   =   eig(X*X','vector');

        [~,ii]  =   sort(D,'descend');

        S       =   sqrt(D(ii));
        U       =   U(:,ii);
        end
    end
end