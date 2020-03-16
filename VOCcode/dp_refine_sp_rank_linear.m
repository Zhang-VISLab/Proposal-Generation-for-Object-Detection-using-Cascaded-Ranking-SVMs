function refine_sp_rank_linear(cls,base,save_fp)
%%%
load([save_fp,cls,'.mat'],'sp'); 
minT = ceil(log(10)/log(base));
maxT = ceil(log(500)/log(base));
T_num = maxT-minT+1;
%%% learn classifiers
load([save_fp,cls,'_train.mat'],'score','label','overlap','img_id','ratio');
score = [score;ones(1,size(score,2))];
dr = sp.dr;
while(1)
    w = Multiclass_LPBoost(score,label,img_id,ratio,overlap',T_num^2,0);
    if any(w(:))
        w = w(1:size(score,1),:) - w(size(score,1)+1:end,:);
        w = w(:,dr);
        rw = w;
        valid = ones(1,length(dr));
        s = sum(w.^2,1);
        valid(s<=0) = 0;               
        break;
    end
end
sp.rw = rw;
sp.valid = valid;
save([save_fp,cls,'.mat'],'sp','-append');
return;
