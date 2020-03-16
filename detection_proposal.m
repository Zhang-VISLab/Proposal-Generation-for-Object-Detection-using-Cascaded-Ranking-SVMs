
tic
% change this path if you install the VOC code elsewhere
addpath([cd '/VOCcode']);
% run('vlfeat-0.9.8\toolbox\vl_setup.m');

% initialize VOC options
VOCinit;

% train and test detector for each class
det_top_num = 1000;     %   the number of final detection candidates
dim = 16;               %   window size for extracting features
nOriBins = 4;           %   the number of feature oritations
base = 2;               %   the scale quantization basis, base^n = scale
challenge_id = 'VOC2006';
fp = [challenge_id,'\PNGImages\'];
save_fp = ['results\',challenge_id,'_',num2str(dim),'x',num2str(nOriBins),'_',num2str(base,'%4.2f'),'\'];

%  main fuction
mkdir(save_fp);
copyfile(['local\',challenge_id,'\trainval_anno.mat'],[save_fp,'trainval_anno.mat']);
copyfile(['local\',challenge_id,'\test_anno.mat'],[save_fp,'test_anno.mat']);

for i=1:VOCopts.nclasses
    cls=VOCopts.classes{i};        
    fprintf([cls,'\n']);
    dp_init_spatial_filter(fp,VOCopts,cls,dim,nOriBins,base,save_fp); 
    dp_extract_sp_features(fp,VOCopts,cls,dim,nOriBins,base,save_fp);
    dp_refine_sp_rank_linear(cls,base,save_fp);      
    dp_locate_bbox_linear(fp,VOCopts,cls,det_top_num,dim,nOriBins,base,save_fp,0);
end

% draw_recall_overlap_curves
obj = cell(1,VOCopts.nclasses);
cumobj = cell(1,VOCopts.nclasses);
Z = cell(1,VOCopts.nclasses);
num = [1,10,100,1000];
for i=1:VOCopts.nclasses
    cls=VOCopts.classes{i};        
    fprintf([cls,'\n']);    
    [obj{i} cumobj{i}] = dp_evaluate_bbox(VOCopts,save_fp,cls,num);
    for j = 1:length(det_top_num)
        Z{i} = [Z{i};trapz(obj{i}(:,1+(j-1)*101:j*101)')./101];
    end
end
save([save_fp,'result.mat'],'obj','cumobj','Z');
toc

