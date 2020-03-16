function locate_bbox_linear(img_fp,VOCopts,cls,det_top_num,dim,nOriBins,base,save_fp,istrain)
%%%
load([save_fp,cls,'.mat'],'sp');  
cas_num = length(det_top_num);
det_sel_num = 50;
minT = ceil(log(10)/log(base));
maxT = ceil(log(500)/log(base));
T_num = maxT-minT+1;
T = zeros(dim,dim,nOriBins,length(sp.dr),'single');
for i = 1:length(sp.dr)    
    for j = 1:nOriBins
        T(:,:,j,i) = reshape(sp.w(1+dim^2*(j-1):dim^2*j,i),[dim dim]);   
    end
end
%%% test
if istrain
    cp=sprintf(VOCopts.annocachepath,VOCopts.trainset);
else
    cp=sprintf(VOCopts.annocachepath,VOCopts.testset);
end
load(cp,'gtids','recs');  
img_id = zeros(1,0,'uint16');
pos = zeros(4,0,'uint16');
cascade = zeros(1,0,'uint8');
ratio = zeros(1,0,'uint8');
obj_totalNum = 0;
dr = sp.dr;
valid = sp.valid;
rw = sp.rw;

parfor i = 1:length(gtids)
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
        fdp=[img_fp,gtids{i},'.png'];
        im = imread(fdp);
        [imy imx imz] = size(im);  
        if imz ~= 1
            im = rgb2gray(im);                        
        end
        P = zeros(4,0,'uint16');
        S = zeros(1,0,'single');
        R = zeros(1,0,'uint8');
        F = zeros(nOriBins+1,0,'single');
        for h = 1:T_num
            height = base^(h+minT-1);
            if height > base*imy
                break;
            else
                height = min(imy,height);
            end
            for w = 1:T_num  
                width = base^(w+minT-1);
                if width > base*imx
                    break;
                else
                    width = min(imx,width);               
                end
                r = sub2ind([T_num T_num],h,w);
                if any(dr==r) && (valid(dr==r)>0)
                    img = imresize(im,round([imy/height imx/width]*dim));
                    mag = VOG(single(img),nOriBins,0);                                
                    G = zeros(size(img,1)-dim+1,size(img,2)-dim+1,nOriBins,'single');
                    for j = 1:nOriBins                    
                        img = mag(:,:,j);
                        if any(img(:))                            
                            G(:,:,j) = cvlib_mex('MatchTemplate', single(img), T(:,:,j,dr==r), 'CCORR');                                                                 
                        end
                    end                                 
                    G1 = double(sum(G,3));
                    [pos1, val] = mexLocalMax(G1, [dim dim], det_sel_num, ceil([dim dim]*0.3));
                    [gy gx gz] = size(G);
                    ind = sub2ind([gy gx],pos1(:,2),pos1(:,1));                
                    G = permute(G,[3 1 2]);                                
                    G = reshape(G,gz,[]);    
                    G = G(:,ind);
                    G = [G;ones(1,size(G,2))];       
                    F = [F,G];
                    pos1 = mexTrueLocation(pos1',[width/dim,height/dim],[imy imx]);
                    P = [P,pos1];
                    R = [R,r*ones(1,size(pos1,2))];                     
                    S = [S,rw(:,dr==r)'*G]; 
                end
            end
        end  
        %%% ranking     
        if ~isempty(S)                            
            [B IX] = sort(-S); 
            IX = IX(1:min(det_top_num,length(IX)));
            F = F(:,IX);
            P = P(:,IX);
            R = R(:,IX);
            ratio = [ratio,uint8(R)];
            pos = [pos,uint16(P)];   
            img_id = [img_id, i*ones(1,length(IX),'uint16')];           
            clsinds(diff) = [];     % delete difficult ones  
            obj_totalNum = obj_totalNum + length(clsinds);            
        end  
    end    
end

bbox.img_id = img_id;
bbox.ratio = ratio;
bbox.pos = pos;
bbox.obj_totalNum = obj_totalNum;
if istrain
    save([save_fp,cls,'.mat'],'bbox','-append');
else
    save([save_fp,cls,'_val.mat'],'bbox');
end
return;
