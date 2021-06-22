function [res] = doublyblockCirculant_by_vector_product(Ceigenvalues, v, M)
    res = real( ifft2(Ceigenvalues .* fft2(flip( reshape(v,M,M),2)  )));
    res = flip(res,2);
    res = res(:);
end