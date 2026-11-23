function ok = predicate_homogeneity(blk, T)
ok = std(blk(:)) <= T;
end
