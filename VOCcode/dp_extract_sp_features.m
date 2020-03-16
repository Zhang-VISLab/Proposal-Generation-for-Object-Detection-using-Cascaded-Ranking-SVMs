function extract_sp_features(img_fp,VOCopts,cls,dim,nOriBins,base,save_fp)
%%%
load([save_fp,cls,'.mat'],'sp'); 
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
cp=sprintf(VOCopts.annocachepath,VOCopts.trainset);
load(cp,'gtids','recs'); 
img_id = zeros(1,0,'uint16');
label = zeros(1,0,'int8');
overlap = zeros(1,0,'single');
ratio = zeros(1,0,'uint8');
score = zeros(nOriBins,0,'single');
pos = zeros(4,0,'uint16');
dr = sp.dr;
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
    
    if gt == 0
        continue;
    end    
    if gt == 1
    fdp=[img_fp,gtids{i},'.png'];
    im = imread(fdp);
    [imy imx imz] = size(im);  
    if imz ~= 1
        im = rgb2gray(im);                        
    end    
                   
%     clsinds(diff) = [];     % delete difficult ones  
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
            if any(dr==r)  
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
                [P, val] = mexLocalMax(G1, [dim dim], det_sel_num, ceil([dim dim]*0.3));
                [gy gx gz] = size(G);                
                G = permute(G,[3 1 2]);                       
                bbgt = [];
                for k = 1:length(clsinds)                       
                    bbgt = [bbgt,recs(i).objects(clsinds(k)).bbox']; 
                end
                pos1 = mexTrueLocation(P',[width/dim,height/dim],[imy imx]);   
                [lx, loss] = mexOverlap(double(pos1),double(bbgt),VOCopts.minoverlap);
                label = [label,int8(lx)'];
                overlap = [overlap,single(loss')];
                img_id = [img_id,i*ones(1,size(P,1),'uint16')];
                ratio = [ratio,r*ones(1,size(P,1),'uint8')];
                G = reshape(G,gz,[]);  
                ind = sub2ind([gy gx],P(:,2),P(:,1));                                        
                score = [score,single(G(:,ind))];     
                pos = [pos,pos1];
            end
        end
    end
    end      
end
save([save_fp,cls,'_train.mat'],'score','label','overlap','img_id','ratio','pos');
return;
