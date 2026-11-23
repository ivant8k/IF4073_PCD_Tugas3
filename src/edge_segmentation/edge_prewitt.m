function E = edge_prewitt(gray, t)
Gx = [-1 0 1; -1 0 1; -1 0 1];
Gy = Gx';
gx = my_conv2(gray, Gx);
gy = my_conv2(gray, Gy);
mag = sqrt(gx.^2 + gy.^2);
mag = mag / max(mag(:));
E = mag > t;
end
