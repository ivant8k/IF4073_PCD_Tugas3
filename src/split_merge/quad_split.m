function B = quad_split(I, r, c, s, T, minBlock)
blk = I(r:r+s-1, c:c+s-1);
if s <= minBlock || predicate_homogeneity(blk, T)
    B = [r c s];
else
    h = s / 2;
    B = [quad_split(I, r, c, h, T, minBlock);
         quad_split(I, r, c+h, h, T, minBlock);
         quad_split(I, r+h, c, h, T, minBlock);
         quad_split(I, r+h, c+h, h, T, minBlock)];
end
end
