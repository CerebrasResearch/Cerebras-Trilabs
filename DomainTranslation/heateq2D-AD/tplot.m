f = fopen('test-t1/tstamp.bin');
x1 = fread(f,[3,Inf],'uint16');
fclose(f);

nx = 2;
ny = 2;
ndt = 512;

x1 = reshape(x1,[3 ndt nx ny]);

d1 = zeros(ndt,nx,ny);
two16 = 2^16;
for i=1:3
  d1 = two16*d1 + squeeze(x1(4-i,:,:,:));
end

d1 = reshape(d1,[ndt nx*ny]);
%plot(d1,'o-')
