function w = rank_pegasos(x,pair,cls,max_cls,lambda2,lambda1,N,w,iteration)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% input
%%%     x:      dim * xlen (single)
%%%     pair:   2 * pair_len (int32), start from 0
%%%     cls:    1 * pair_len (int32), start from 0
%%%     max_cls:number of classes (int32)
%%%     lambda2:trade-off parameter for L2 norm, e.g. 1e-3 (single)
%%%     lambda1:trade-off parameter for L1 norm, e.g. 1e-3 (single)
%%%     N:      L0 (N=0), L1 (N=1) or L2 (N=2) norm learning (int32)
%%% output
%%%     w:      dim * max_cls (single)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

w = mexRankPegasos(single(x), int32(pair), int32(cls), int32(max_cls) ,...
    single(lambda2), single(lambda1), int32(N), single(w), int32(iteration));