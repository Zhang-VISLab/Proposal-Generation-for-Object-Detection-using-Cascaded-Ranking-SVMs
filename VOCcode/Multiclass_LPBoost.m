function w = Multiclass_LPBoost(score,label,group,ratio,overlap,cls_num,positive)
%%%
total_num = 2e4;
if ~positive
    score = [score;-score];
end
dim = size(score,1);
id = unique(group);
per_num = max(1,floor(total_num/length(id)));
pair = [];
R = [];
parfor i = 1:length(id)
    ind1 = find(group==id(i)&label==1);
    ind2 = find(group==id(i)&label~=1);
    ind1 = ind1(:);
    ind2 = ind2(:);
    if ~isempty(ind1) && ~isempty(ind2)
        [ix1 ix2] = meshgrid(1:length(ind1),1:length(ind2));
        ix1 = ix1(:);
        ix2 = ix2(:);
        if length(ix1)>per_num
            ind = randperm(length(ix1));
            ix1 = ix1(ind(1:per_num));
            ix2 = ix2(ind(1:per_num));
        end
        ind1 = ind1(ix1);
        ind2 = ind2(ix2);
        pair = [pair,[ind1,ind2]'];        
        r1 = double(ratio(ind1));
        r2 = double(ratio(ind2));
        R = [R,[r1;r2]];
    end
end
[S I J] = mexSparseMatrices(double(score),int32(pair),int32(R));
xlen = max(J);
S = sparse(double(I),double(J),double(S),dim*cls_num,double(xlen));
C = 1e3;
if isempty(overlap)
    loss = ones(xlen,1);
else
    loss = overlap(pair(1,:)')-overlap(pair(2,:)');    
end
param.msglev = 3;
[xmin, fmin, status, extra] = glpk(double(loss),S,ones(dim*cls_num,1),...
    zeros(xlen,1),C*ones(xlen,1),[],[],-1,param);
w = extra.lambda;
w = reshape(w,dim,[]);

return;
