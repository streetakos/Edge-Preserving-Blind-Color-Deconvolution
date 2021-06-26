function [E, H] = deconv_st(Y,M,sa,max_iter)

% Spatially variant variational Bayesian blind image deconvolution with Students T prior
% M:        Color stain matrix (values in OD space) (3 x 2)
% Y:        Stained RGB image (values [0,1]) (sa x sa x 3)
% sa:       Image width/height
% max_iter: Maximum number of iterations of the Variational EM

% Transform image to OD space and vectorize each channel
Y = rgb2od(Y);
Y = reshape(Y,sa^2,3);
im_size = length(Y);


% Rensure that the reference color vector is unary
M(:,1) = M(:,1) / norm(M(:,1),2);
M(:,2) = M(:,2) / norm(M(:,2),2);



% Load the padded Laplacian 2D filter and Circulant matrix
F =load('tempfilter500.mat');
F = F.fil;
F_full = load('temp500.mat');
F_full = F_full.F;

% Initialize Prior parameters
gamma = ones(2, 1) .*10;
b = 10;
a_prior = ones(2,1) ;
b_prior = ones(2,1);


% Initialize Variational parameters
mean_m = M;
mean_c = M\Y';
mean_c = mean_c';
Cov_m = zeros(3,3,2);
Cov_c = zeros(sa,sa,2);
mean_a = ones(im_size,2);	
a_ =  ones(im_size,2);	
b_ =  ones(im_size,2);

% Compute the eighenvalues of F'*F
FF_eigs = power(real(fft2(F)),2);
F_eigs  = real(fft2(F)); 

% Main EM loop
for iteration = 1:max_iter
        
        %%%%%%%%%%
        % M-step %
        %%%%%%%%%%
            
        % Don't calculate gamma in the first iteration leads to singular matrix    
        if iteration > 1
            gamma(1) = 3 / (trace((mean_m(:,1) - M(:,1))*(mean_m(:,1) - M(:,1))') + trace(Cov_m(:,:,1)));
            gamma(2) = 3 / (trace((mean_m(:,2) - M(:,2))*(mean_m(:,2) - M(:,2))') + trace(Cov_m(:,:,2)));
        
        
            b = 3*im_size/( trace((Y' - mean_m*mean_c')*(Y' - mean_m*mean_c')' ) ...
            + mean_m(:,1)'*mean_m(:,1)*sum(sum(Cov_c(:,:,1))) + mean_m(:,2).'*mean_m(:,2)*sum(sum(Cov_c(:,:,2))) ...
            + mean_c(:,1)'*mean_c(:,1)*trace(Cov_m(:,:,1)) + mean_c(:,2)'*mean_c(:,2)*trace(Cov_m(:,:,2)) ...
            + sum(sum(Cov_c(:,:,1)))*trace(Cov_m(:,:,1)) + sum(sum(Cov_c(:,:,2)))*trace(Cov_m(:,:,2))  ...
            );  
        end
 
%         % Gamma parameters update        
%         b_prior_temp =  b_prior;
%         for s = 1:2
%            b_prior(s) = (im_size*a_prior(s)) ./ sum(mean_a(:,s));
%         end
%         
%         for s = 1:2
%             fun = @(a) im_size*log(b_prior_temp(s)) - im_size* psi(a) + sum( psi(a_(:,s)) -  log(b_(:,s))) ;
%             a_prior(s) = iterationsfzero(fun, a_prior(s));
%         end      

        
        %%%%%%%%%%
        % E-step %
        %%%%%%%%%%
        
        % Update Gamma
        for s = 1:2
            a_(:,s) = a_prior(s).* ones(im_size,1) + 0.5;
            cc = doublyblockCirculant_by_vector_product(F_eigs,mean_c(:,s),sa) ;
            bb= real(ifft2(FF_eigs .* Cov_c(:,:,s)));
            b_(:,s) = b_prior(s).* ones(im_size,1) + 0.5.*( bb(1,1) + cc.*cc);
            mean_a(:,s) = a_(:,s)./b_(:,s);
        end
        
        % Update C
        % Approxiamated Covariance matrix (Compute only the eigenvalues)
        for s = 1:2
            Cov_c(:,:,s) = 1./(b*( mean_m(:,s)'*mean_m(:,s) + trace(Cov_m(:,:,s)) ) +  mean(mean_a(:,s)) .*  FF_eigs) ;
        end        
        
        % Real Precision Matrix (Sparce Matrxi)
        Cov_c0 = b.* (mean_m(:,1)'*mean_m(:,1) + trace(Cov_m(:,:,1))).* speye(im_size) + ...
                           F_full * spdiags(mean_a(:,1,:),0,im_size,im_size) *F_full; 
        Cov_c1 = b.* (mean_m(:,2)'*mean_m(:,2) + trace(Cov_m(:,:,2))).* speye(im_size) + ...
                           F_full * spdiags(mean_a(:,2,:),0,im_size,im_size) *F_full;       
       
        % Use previous parameters to calculate the new ones
        CC = mean_c;
        
        % Find mean concentrations using conjugate gradient)
        z =  b.* mean_m(:,1)' * (Y -  repelem(CC(:,2),1,3) .*   mean_m(:,2 )')'  ;
        [mean_c(:,1),~] =  cgs(Cov_c0, z',1e-6,140); 
                
        z = b.*  mean_m(:,2)' * (Y -  repelem(CC(:,1),1,3)  .*   mean_m(:,1 )')'  ;
        [mean_c(:,2),~] = cgs(Cov_c1, z',1e-6,140); 

        % Update M
        for s = 1:2
            Cov_m(:,:,s) = inv(  (b*( sum(mean_c(:,s).^2) + sum(sum(Cov_c(:,:,s)))  )  + gamma(s)) .* eye(3) ) ;
        end
        
        % Use previous parameters to calculate the new ones        
        MM = mean_m;
        mean_m(:,1) = Cov_m(:,:,1)*(b.*sum(mean_c(:,1).* (Y -  repelem(mean_c(:,2),1,3)  .*   MM(:,2 )'))'  + gamma(1).*M(:,1));
        mean_m(:,2) = Cov_m(:,:,2)*(b.*sum(mean_c(:,2).* (Y -  repelem(mean_c(:,1),1,3)  .*   MM(:,1 )'))'  + gamma(2).*M(:,2));
 
        % Make unit vectors
        mean_m(:,2) = mean_m(:,2) / norm(mean_m(:,2),2);
        mean_m(:,1) = mean_m(:,1) / norm(mean_m(:,1),2);
        Cov_m(:,:,1) = Cov_m(:,:,1) / (mean_m(:,1)'*mean_m(:,1)); 
        Cov_m(:,:,2) = Cov_m(:,:,2) / (mean_m(:,2)'*mean_m(:,2));
        

        % Create the separated images H and E
        H = 255*exp(-mean_m(:,2)*mean_c(:,2)');
        H = reshape(H', sa, sa, 3);
        H = uint8(H);
        E = 255*exp(-mean_m(:,1)*mean_c(:,1)');
        E = reshape(E', sa, sa, 3);
        E = uint8(E);
        
        % Check for convergence
        crit = 1/(10^4);
        if iteration > 1 && ( norm(mean_c(:,1) - mean_c_prev(:,1),2)/norm(mean_c(:,1),2) < crit )&& ...
            (norm(mean_c(:,2) - mean_c_prev(:,2),2)/norm(mean_c(:,2),2) < crit)
            display('Reached')
            return
        end
        mean_c_prev = mean_c;
        
        
    end
    % Create the separated images H and E
    H = 255*exp(-mean_m(:,2)*mean_c(:,2)');
    H = reshape(H', sa, sa, 3);
    H = uint8(H);
    E = 255*exp(-mean_m(:,1)*mean_c(:,1)');
    E = reshape(E', sa, sa, 3);
    E = uint8(E);
end
        
        
        
        
        
        
        
        
