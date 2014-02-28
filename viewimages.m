function [] = viewimages( imagepath, width ,height , depth )

fin = fopen([imagepath 'input.raw']);
inputImage = fread(fin,[width height],['ubit' num2str(depth) '=>uint64'],0,'s')';

fout = fopen([imagepath 'output.raw']);
outputImage = fread(fout,[width height],['ubit' num2str(depth) '=>uint64'],0,'s')';

ftest = fopen([imagepath 'test.raw']);
testImage = fread(ftest,[width height],['ubit' num2str(depth) '=>uint64'],0,'s')';

fclose(fin);
fclose(fout);
fclose(ftest);

subplot(1,3,1), imshow(inputImage,[0 (2^depth)-1])
subplot(1,3,2), imshow(outputImage,[0 (2^depth)-1])
subplot(1,3,3), imshow(testImage,[0 (2^depth)-1])



end

