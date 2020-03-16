function init_spatial_filter(fp,VOCopts,cls,dim,nOriBins,base,save_fp)
%%%
cp=sprintf(VOCopts.annocachepath,VOCopts.trainset);
load(cp,'gtids','recs');   
minT = ceil(log(10)/log(base));
maxT = ceil(log(500)/log(base));
T_num = maxT-minT+1;
det_sel_num = 1e1;
%%%
x = single([]);
lx = uint8([]);
group = uint16([]);
ratio = uint8([]);
parfor i=1:length(gtids)
    % find objects of class and extract difficult flags for these objects
    clsinds=strmatch(cls,{recs(i).objects(:).class},'exact');
    diff=[recs(i).objects(clsinds).difficult];    
    % assign ground truth class to image
    if isempty(clsinds)
        gt=-1;          % no objects of class
    elseif any(~diff)
        gt=1;           % at least one non-difficult object of class
    else
        gt=0;           % only difficult objects
    end
         
    if gt == 1
        fdp=[fp,gtids{i},'.png'];
        im = imread(fdp);
        [imy imx imz] = size(im);           
        if imz ~= 1
            im = rgb2gray(im);
        end

        clsinds(diff) = [];     % delete difficult ones  
        for k = 1:length(clsinds)               
            bbgt = recs(i).objects(clsinds(k)).bbox;       %   (x_{min},y_{min},x_{max},y_{max})                                                             
            minH = max(minT,floor(log(bbgt(4)-bbgt(2)+1)/log(base)));
            minW = max(minT,floor(log(bbgt(3)-bbgt(1)+1)/log(base)));
            for h = minH:min(maxT,minH+1)
                for w = minW:min(maxT,minW+1)                    
                    bb = [bbgt(1),bbgt(2),min(imx,bbgt(1)+base^w),min(imy,bbgt(2)+base^h)];                    
                    [ov, loss] = mexOverlap(double(bb)',double(bbgt)',VOCopts.minoverlap);                    
                    if ov==1
                        img = im(bbgt(2):bbgt(4),bbgt(1):bbgt(3));      
                        img = imresize(img,[dim dim]);
                        mag = VOG(single(img),nOriBins,1);       
                        x = [x,mag-mean(mag)];
                        img = fliplr(img);
                        mag = VOG(single(img),nOriBins,1);       
                        x = [x,mag-mean(mag)];
                        lx = [lx,ones(1,2)];
                        group = [group,i*ones(1,2)];                        
                        r = sub2ind([T_num T_num],h-minT+1,w-minT+1);
                        ratio = [ratio,r*ones(1,2)];                        
                    end
                end
            end
            %%% add data
            [xx yy] = meshgrid(1:imx,1:imy);
            pos = [xx(:),yy(:)];
            ind = randperm(size(pos,1));
            p = pos(ind(1:min(det_sel_num,size(pos,1))),:);
            ind = randperm(size(pos,1));
            s = pos(ind(1:min(det_sel_num,size(pos,1))),:);
            pos = [p,min(imx,p(:,1)+s(:,1)),min(imy,p(:,2)+s(:,2))];                        
            bbgt = [];
            for n = 1:length(clsinds)                       
                bbgt = [bbgt,recs(i).objects(clsinds(n)).bbox']; 
            end
            [ov, loss] = mexOverlap(double(pos)',double(bbgt),VOCopts.minoverlap);
            for m = 1:size(pos,1)                                      
                if ov(m) == -1
                    img = im(pos(m,2):pos(m,4),pos(m,1):pos(m,3));                       
                    group = [group,i];                        
                    img = imresize(img,[dim dim]);
                    mag = VOG(single(img),nOriBins,1);
                    x = [x,mag-mean(mag)];
                    lx = [lx,0];
                    ratio = [ratio,0];
                end
            end
        end        
    end         
end
%%% find dominant ratio
pr = single(ratio);
dr = unique(pr);
dr(dr==0) = [];
%%% learn
sp.dr = dr;
%%% ratio
W = [];
for j = 1:length(dr)      
    fprintf('%i : ',dr(j));
    id = group(ratio==dr(j));
    id = unique(id);
    ix = [];    
    Z = [];
    for k = 1:length(id)
		ind = find(group==id(k));
		ix = [ix,ind];
        Z = [Z,id(k)*ones(1,sum(group==id(k)),'uint16')];
    end	
    w = zeros(size(x,1),1);
    tic
    w = rankSVM_online(x(:,ix),lx(ix),Z,zeros(1,length(ix)),1,1e-2,0,2,w,0,0);    
    toc

    W = [W,w];
end
sp.w = W;
try
    save([save_fp,cls,'.mat'],'sp','-append');
catch
    save([save_fp,cls,'.mat'],'sp');
end
return;
