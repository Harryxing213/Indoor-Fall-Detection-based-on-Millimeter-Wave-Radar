function [RowPoint] = F_CoordinateConversion(Parameter)
%F_COORDINATECONVERSION 此处显示有关此函数的摘要
%   此处显示详细说明
   
    switch Parameter.type
        case 1%matlab脚本采集
            load(Parameter.fileID);
            RowPoint = Frame;
            
        case 2%txt采集然后matlab脚本转换
            load(Parameter.fileID);
            RowPoint = DataRead(TargetPosition);
            
        case 3%邵烽智回灌工具采集
            load(Parameter.fileID);
            [RowPoint] = DataRead_SFZ(frame_data,Parameter);
            
        case 4 %FFT-1D+点云(CXD有线)
            load(Parameter.fileID);
            [RowPoint] = F_FFT_and_Point_YX_CXD(TargetPosition,Parameter);
            
        case 5 %FFT-1D+点云(无线)
            [RowPoint] = F_FFT_and_Point_WX(Parameter);
            
        case 6 %FFT-1D+点云(CXD有线)
            load(Parameter.fileID);
            [RowPoint] = F_FFT_and_Point_YX_SFZ(frame_data,Parameter);
            
        otherwise
            disp("Parameter.type ERROR")
    end
end

function [RowPoint] = F_FFT_and_Point_YX_SFZ(frame_data,Parameter)
    TILT = -90;%雷达倾斜角度
    TILT1 = 0;%雷达倾斜角度
    RowPoint = [];
    for frame = 1:size(frame_data,2)
        rawdata_point = (frame_data(frame).TargetPosition(1205:end))';
        if isempty(rawdata_point)
            break;
        end
        
        %% Point 数据极坐标转直角坐标
%         axbuffer_length = (rawdata_point(2)) + (rawdata_point(3))*256 + (rawdata_point(4))*256*256 + (rawdata_point(5))*256*256*256 ;
%         if axbuffer_length>3000
%             continue;
%         end
        pointnum = rawdata_point(7) + rawdata_point(8)*256;
        if pointnum == 0 
                point = [frame,100,100,100,0,0,0];
                RowPoint = [RowPoint;point];
        else
            for num = 0:pointnum-1
                range = (rawdata_point(9+num*9) + rawdata_point(10+num*9)*256)*0.05004;

                velocity = (rawdata_point(11+num*9)-32) * 0.104167 ;

                if(rawdata_point(12+num*9) <= 63)
                    azimuth_angle = asin(rawdata_point(12+num*9)/64.0)*180/pi;
                elseif(rawdata_point(12+num*9) <= 127)
                    azimuth_angle = asin((rawdata_point(12+num*9)-128)/64.0)*180/pi;
                end

                if(rawdata_point(13+num*9) <= 63)
                    elevation_angle = asin(rawdata_point(13+num*9)/64.0)*180/pi;
                elseif(rawdata_point(13+num*9) <= 127)
                    elevation_angle = asin((rawdata_point(13+num*9)-128)/64.0)*180/pi;
                end

                x = range * sind(azimuth_angle) * cosd(elevation_angle);
                y = range * cosd(elevation_angle) * cosd(azimuth_angle);
                z = range * sind(elevation_angle) ;

                point_z1 = cosd(TILT) * z + sind(TILT) * y ;
                point_y1 = -sind(TILT) * z + cosd(TILT) * y;

                point_z = cosd(TILT1) * point_z1 + sind(TILT1) * point_y1 + Parameter.hight;
                point_y = -sind(TILT1) * point_z1 + cosd(TILT1) * point_y1;

                point_x = x;
                point_v = velocity;
                point_vx = velocity * sind(azimuth_angle) * cosd(elevation_angle);
                point_vy = velocity * sind(elevation_angle);

                point = [frame,point_x,point_y,point_z,point_v,point_vx,point_vy];

                RowPoint = [RowPoint;point];
            end
        end
    end
end


function [RowPoint] = F_FFT_and_Point_YX_CXD(TargetPosition,Parameter)
    TILT = -90;%雷达倾斜角度
    TILT1 = 0;%雷达倾斜角度
    RowPoint = [];
    for frame = 1:size(TargetPosition,1)
        if isempty(frame)
            break;
        end
        
        pointnum = TargetPosition(frame).pointnum;
        if pointnum == 0 
            point = [frame,100,100,100,0,0,0];
            RowPoint = [RowPoint;point];
        else
            for i = 0:pointnum-1
                range = (TargetPosition(frame).rawdata_point(13+i*9) + TargetPosition(frame).rawdata_point(14+i*9)*256)*0.05004;

                velocity = (TargetPosition(frame).rawdata_point(15+i*9)-32) * 0.104167 ;

                if(TargetPosition(frame).rawdata_point(16+i*9) <= 63)
                    azimuth_angle = asin(TargetPosition(frame).rawdata_point(16+i*9)/64.0)*180/pi;
                elseif(TargetPosition(frame).rawdata_point(16+i*9) <= 127)
                    azimuth_angle = asin((TargetPosition(frame).rawdata_point(16+i*9)-128)/64.0)*180/pi;
                end

                if(TargetPosition(frame).rawdata_point(17+i*9) <= 63)
                    elevation_angle = asin(TargetPosition(frame).rawdata_point(17+i*9)/64.0)*180/pi;
                elseif(TargetPosition(frame).rawdata_point(17+i*9) <= 127)
                    elevation_angle = asin((TargetPosition(frame).rawdata_point(17+i*9)-128)/64.0)*180/pi;
                end

                x = range * sind(azimuth_angle) * cosd(elevation_angle);
                y = range * cosd(elevation_angle) * cosd(azimuth_angle);
                z = range * sind(elevation_angle) ;

                point_z1 = cosd(TILT) * z + sind(TILT) * y ;
                point_y1 = -sind(TILT) * z + cosd(TILT) * y;

                point_z = cosd(TILT1) * point_z1 + sind(TILT1) * point_y1 + Parameter.hight;
                point_y = -sind(TILT1) * point_z1 + cosd(TILT1) * point_y1;

                point_x = x;
                point_v = velocity;
                point_vx = velocity * sind(azimuth_angle) * cosd(elevation_angle);
                point_vy = velocity * sind(elevation_angle);

                point = [frame,point_x,point_y,point_z,point_v,point_vx,point_vy];

                RowPoint = [RowPoint;point];
            end
        end
    end
end

function [RowPoint] = F_FFT_and_Point_WX(Parameter)
FileIDname = Parameter.fileID;
%RAW_DATA_PARSING 此处显示有关此函数的摘要
%   此处显示详细说明
fid = fopen(FileIDname, 'r','b');
% 检查文件是否成功打开
if fid == -1
    error('无法打开文件');
end
data = fread(fid,  '*uint8')'; % 读取所有字符，并转置为行向量
% 关闭文件
fclose(fid);

%% FFT1D 参数设置
header_fft = [hex2dec('66'),hex2dec('BB')];
%% POINT 参数设置
header_point = [hex2dec('55'), hex2dec('AA')];
%% 数据解析
Parameter.fftBytes = 1204;
Parameter.antnum = 1;
TILT = -90;%雷达倾斜角度
TILT1 = 0;%雷达倾斜角度
RowPoint = [];

i = 1;
frame = 1;
while i <= length(data) - 1
    %% FFT 数据
    if isequal(data(i:i+1), header_fft) && i <= length(data) - Parameter.fftBytes*Parameter.antnum
        start_fft = i;
        end_fft = i+1203;
        fft = data(start_fft:end_fft);        
    end
    %% Point 数据
    start_point = end_fft+1;
    if isequal(data(start_point:start_point+1), header_point) 
        axbuffer_length = double(data(start_point+2)) + double(data(start_point+3))*256 + double(data(start_point+4))*256*256 + double(data(start_point+5))*256*256*256 ;
        
        if axbuffer_length>3000
            continue;
        end
        
        end_point = end_fft+axbuffer_length+6;
        
        rawdata_point = double(data(start_point:end_point));
    end
    
    %% Point 数据极坐标转直角坐标
    pointnum = rawdata_point(11) + rawdata_point(12)*256;
    if pointnum == 0 
            point = [frame,100,100,100,0,0,0];
            RowPoint = [RowPoint;point];
    else
        for num = 0:pointnum-1
            range = (rawdata_point(13+num*9) + rawdata_point(14+num*9)*256)*0.05004;

            velocity = (rawdata_point(15+num*9)-32) * 0.104167 ;

            if(rawdata_point(16+num*9) <= 63)
                azimuth_angle = asin(rawdata_point(16+num*9)/64.0)*180/pi;
            elseif(rawdata_point(16+num*9) <= 127)
                azimuth_angle = asin((rawdata_point(16+num*9)-128)/64.0)*180/pi;
            end

            if(rawdata_point(17+num*9) <= 63)
                elevation_angle = asin(rawdata_point(17+num*9)/64.0)*180/pi;
            elseif(rawdata_point(17+num*9) <= 127)
                elevation_angle = asin((rawdata_point(17+num*9)-128)/64.0)*180/pi;
            end

            x = range * sind(azimuth_angle) * cosd(elevation_angle);
            y = range * cosd(elevation_angle) * cosd(azimuth_angle);
            z = range * sind(elevation_angle) ;

            point_z1 = cosd(TILT) * z + sind(TILT) * y ;
            point_y1 = -sind(TILT) * z + cosd(TILT) * y;

            point_z = cosd(TILT1) * point_z1 + sind(TILT1) * point_y1 + Parameter.hight;
            point_y = -sind(TILT1) * point_z1 + cosd(TILT1) * point_y1;

            point_x = x;
            point_v = velocity;
            point_vx = velocity * sind(azimuth_angle) * cosd(elevation_angle);
            point_vy = velocity * sind(elevation_angle);

            point = [frame,point_x,point_y,point_z,point_v,point_vx,point_vy];

            RowPoint = [RowPoint;point];
        end
    end
    
    i = end_point+1;
    frame = frame+1;
end
end

function [RowPoint] = DataRead_SFZ(frame_data,Parameter)
    TILT = -90;%雷达倾斜角度
    TILT1 = 0;%雷达倾斜角度
    RowPoint = [];
    for frame  = 1:length(frame_data)
        if(isempty(frame_data(1,frame).TargetPosition))
             point_x = 100;
             point_y = 100;
             point_z = 100;
             point_v = 0;
             isNoise = 1;
             cluster_id = 0;
             point = [frame,point_x,point_y,point_z,point_v,isNoise,cluster_id];

            RowPoint = [RowPoint;point];
        end
        for i = 1:size(frame_data(1,frame).TargetPosition,2)
            range = frame_data(1,frame).TargetPosition(1,i);
            velocity = frame_data(1,frame).TargetPosition(2,i);
            azimuth_angle = frame_data(1,frame).TargetPosition(3,i);
            elevation_angle = frame_data(1,frame).TargetPosition(4,i);

            x = range * sind(azimuth_angle) * cosd(elevation_angle);
            y = range * cosd(elevation_angle) * cosd(azimuth_angle);
            z = range * sind(elevation_angle) ;

            point_z1 = cosd(TILT) * z + sind(TILT) * y ;
            point_y1 = -sind(TILT) * z + cosd(TILT) * y;

            point_z = cosd(TILT1) * point_z1 + sind(TILT1) * point_y1 + Parameter.hight;
            point_y = -sind(TILT1) * point_z1 + cosd(TILT1) * point_y1;

            point_x = x;
            point_v = velocity;
            point_vx = velocity * sind(azimuth_angle) * cosd(elevation_angle);
            point_vy = velocity * sind(elevation_angle);

            point = [frame,point_x,point_y,point_z,point_v,point_vx,point_vy];

            RowPoint = [RowPoint;point];
        end
    end
end


function [Data] = DataRead(TargetPosition)
%DATAREAD 此处显示有关此函数的摘要
%   此处显示详细说明
Parameter.Rader_height =  2.7;  %测量空间宽度高度
Parameter.TILT = -90;%雷达倾斜角度
Parameter.TILT1 = 0;%雷达倾斜角度
Data = [];
for frame  = 1:length(TargetPosition)
    if(TargetPosition(frame).pointnum==0)
         point_x = 100;
         point_y = 100;
         point_z = 100;
         point_v = 0;
         isNoise = 1;
         cluster_id = 0;
         point = [frame,point_x,point_y,point_z,point_v,isNoise,cluster_id];
        
        Data = [Data;point];
    end
    for i = 1:TargetPosition(frame).pointnum
        range = TargetPosition(frame).totalpoint(i).range;
        velocity = TargetPosition(frame).totalpoint(i).velocity;
        azimuth_angle = TargetPosition(frame).totalpoint(i).azimuth_angle;
        elevation_angle = TargetPosition(frame).totalpoint(i).elevation_angle;
        
        x = range * sind(azimuth_angle) * cosd(elevation_angle);
        y = range * cosd(elevation_angle) * cosd(azimuth_angle);
        z = range * sind(elevation_angle) ;
        
        point_z1 = cosd(Parameter.TILT) * z + sind(Parameter.TILT) * y ;
        point_y1 = -sind(Parameter.TILT) * z + cosd(Parameter.TILT) * y;
        
        point_z = cosd(Parameter.TILT1) * point_z1 + sind(Parameter.TILT1) * point_y1 + Parameter.Rader_height;
        point_y = -sind(Parameter.TILT1) * point_z1 + cosd(Parameter.TILT1) * point_y1;
        
        point_x = x;
        point_v = TargetPosition(frame).totalpoint(i).velocity;
        isNoise = 0;
        cluster_id = -1;
        
        R = sqrt(point_x^2+point_y^2);
        
%         if((R>(Parameter.Rader_height-1)*1.7))
%              point_x = 100;
%              point_y = 100;
%              point_z = 100;
%              isNoise = 1;
%              cluster_id = 0;
%         else
%             
%         end
        point = [frame,point_x,point_y,point_z,point_v,isNoise,cluster_id];
        
        Data = [Data;point];
    end
end
end

