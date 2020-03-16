function M = VOG(im,nOriBins,isnormalized)
[fx fy] = gradient(im);
ori = atan2(fy,fx);
mag = abs(fx)+abs(fy);
oriSpacing = 2*pi/nOriBins;
oriBins = -pi:oriSpacing:pi-oriSpacing;
ori = ori(:);
w = 1 - mod(abs(repmat(ori,[1 nOriBins]) - repmat(oriBins,[length(ori) 1])),pi)./pi;
w = normalize(w,2);
mag = repmat(mag(:),[1 nOriBins]) .* w;
M = zeros(size(im,1),size(im,2),nOriBins);
for i = 1:nOriBins
    M(:,:,i) = reshape(mag(:,i),[size(im,1) size(im,2)]);
end
if isnormalized
    M = M(:);
end
return;
