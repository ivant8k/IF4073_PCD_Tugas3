function E = edge_laplace(gray, t)
K = [0 1 0; 1 -4 1; 0 1 0];
R = my_conv2(gray, K);
E = zero_cross(R, t);
end
