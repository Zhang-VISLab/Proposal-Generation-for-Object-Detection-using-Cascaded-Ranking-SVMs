function [obj cumobj] = evaluate_bbox(VOCopts,save_fp,cls,num)
%%%
load([save_fp,cls,'_val.mat'],'bbox');
cp=sprintf(VOCopts.annocachepath,VOCopts.testset);
load(cp,'gtids','recs'); 
obj = [];
cumobj = [];
pos = bbox.pos;
img_id = bbox.img_id;
parfor i = 1:length(gtids)
    % find objects of class and extract difficult flags for these objects
    clsinds=strmatch(cls,{recs(i).objects(:).class},'exact');
    diff=[recs(i).objects(clsinds).difficult];    
    % assign ground truth class to image
    if isempty(clsinds)
%         gt=-1;          % no objects of class
        continue;
    elseif any(~diff)
        gt=1;           % at least one non-difficult object of class
    else
%         gt=0;           % only difficult objects
        continue;
    end
         
    clsinds(diff) = [];     % delete difficult ones  
    IX = find(img_id==i);
    len = length(IX);
    P = pos(:,IX);
    K = zeros(len,length(clsinds));
    for j = 1:length(clsinds)
        bbgt = recs(i).objects(clsinds(j)).bbox;
        [ov, K(:,j)] = mexOverlap(double(P),double(bbgt)',0);            
    end     
    C = [];
    for j = 0:0.01:1
        for k = 1:length(num)
            tmp = K(1:min(num(k),size(K,1)),:);
            tmp(tmp>=j) = 1;
            tmp(tmp<j) = 0;
            A = sum(max(tmp,[],1));                
            B = sum(max(tmp,[],2));                
            C = [C; min(sum(A),sum(B))];            
        end
    end    

    obj = [obj,C];
    C = [];
    for j = 1:1e3
        tmp = K(1:min(j,size(K,1)),:);
        tmp(tmp>=VOCopts.minoverlap) = 1;
        tmp(tmp<VOCopts.minoverlap) = 0;
        A = sum(max(tmp,[],1));                
        B = sum(max(tmp,[],2));                
        C = [C; min(sum(A),sum(B))];
    end
    cumobj = [cumobj,C];
end
cumobj = sum(cumobj,2)./bbox.obj_totalNum;
obj = sum(obj,2)./bbox.obj_totalNum;
obj = reshape(obj, length(num), []);
return;
