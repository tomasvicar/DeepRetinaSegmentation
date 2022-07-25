clc;clear all;close all;

listing = subdir('.');


for file_num = 1:length(listing)
    filename = listing(file_num).name;
    disp(filename)
    if contains(filename,'replace_imwrite_single.m')||contains(filename,'MAIN.m')
        continue;
    end

    S = readlines(filename);
    

    fid = fopen(filename,'w');
    for line_num = 1:length(S)
        
        s = S{line_num};


        if  contains(s,'imwrite_single(')
            drawnow;
            s = replace(s,'imwrite_single(','imwrite_2_h5(');
            s = replace(s,'''.tiff''','');
            s = replace(s,'.tiff','');
            s = replace(s,'[out_f ','out_f, [');

        end
        fprintf(fid,'%s', s );
        fprintf(fid, '\r\n' );
    end
    fclose(fid);

    drawnow;
end