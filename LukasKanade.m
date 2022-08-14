clear all;
load traffic_frames

im1 = im2double(rgb2gray(fr1));
im2 = im2double(rgb2gray(fr2));


window = 40;
w = window / 2;

C = corner(im2);


[Ix, Iy] = imgradientxy(im1);
It = conv2(im1, ones(2), 'valid') - conv2(im2, ones(2), 'valid');

u = zeros(length(C), 1);
v = zeros(length(C), 1);

for k = 1  : length(C(:, 2))

    x = C(k, 2);
    y = C(k, 1);
    if x - w < 1 || y - w < 1 || x + w > size(im1, 1) - 1 || y + w > size(im1, 2) - 1
        continue
    end
    
    i = C(k, 2);
    j = C(k, 1);
    
    tmp = Ix(i - w : i + w, j - w : j + w);
    
    Ix_window = tmp(:);
    
    tmp = Iy(i - w : i + w, j - w : j + w);
    Iy_window = tmp(:);

    tmp = It(i - w : i + w, j - w : j + w);
    b = tmp(:);
    
    A = [Ix_window Iy_window];
    nu = pinv(A) * b;
    
    u(k) = nu(1);
    v(k) = nu(2);
end

figure();
imshow(fr2);
hold on;
quiver(C(:,1), C(:,2), u,v, 1,'r')