function [testImage,outputImage] = raw2mat( imagepath, width ,height , depth )

fout = fopen([imagepath 'output.raw']);
outputImage = fread(fout,[width height],['ubit' num2str(depth) '=>uint64'],0,'s')';

ftest = fopen([imagepath 'test.raw']);
testImage = fread(ftest,[width height],['ubit' num2str(depth) '=>uint64'],0,'s')';

fclose(fout);
fclose(ftest);

end

