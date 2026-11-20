function prepare_frames(inputFolder, outputFolder, radarHeight, showFrames, overwrite)
% 将 type=3 的 frame_data 转为 Python 训练所用的逐帧点云。
% prepare_frames('raw_data', 'processed_data', 2.1, false, false)
% 2.1 为原程序参数；数据说明写的是 2.7 m，需按实际安装标定选择。
if nargin < 2
    error('Please provide inputFolder and outputFolder.');
end
if nargin < 3, radarHeight = 2.1; end
if nargin < 4, showFrames = false; end
if nargin < 5, overwrite = false; end
validateattributes(radarHeight, {'numeric'}, {'scalar', 'finite', 'positive'});
addpath(fileparts(mfilename('fullpath')));
FileInfo = dir(fullfile(inputFolder, '*.mat'));
if isempty(FileInfo), error('No MAT files found in inputFolder.'); end
if ~exist(outputFolder, 'dir'), mkdir(outputFolder); end

for i = 1:length(FileInfo)
    fileID = fullfile(FileInfo(i).folder, FileInfo(i).name);
    [~, baseName, ~] = fileparts(fileID);
    save_name = fullfile(outputFolder, [baseName, '_frames.mat']);
    if exist(save_name, 'file') && ~overwrite
        error('Output already exists: %s. Set overwrite=true to replace it.', save_name);
    end
    source = load(fileID, 'frame_data');
    if ~isfield(source, 'frame_data'), error('Missing frame_data in %s', fileID); end
    frame_data = source.frame_data;
    if ~isrow(frame_data)
        error('Expected the original 1-by-T frame_data structure in %s', fileID);
    end
    Parameter = ParameterInit();
    Parameter.type = 3;
    Parameter.fileID = fileID;
    Parameter.hight = radarHeight;
    Parameter.RadarPosition = [0, 0, radarHeight];
    Parameter.box = F_BoxSet(Parameter);
    RowPoint = F_CoordinateConversion(Parameter);
    if isempty(RowPoint), error('No decoded frames in %s', fileID); end
    numFrames = max(RowPoint(:, 1));
    points_per_frame = cell(numFrames, 1);
    flags_per_frame = cell(numFrames, 1);
    if showFrames, figure; end
    for framecnt = 1:numFrames
        rows = RowPoint(:, 1) == framecnt;
        points_per_frame{framecnt} = RowPoint(rows, [2, 3, 4, 6, 7]);
        if framecnt <= numel(frame_data)
            flags_per_frame{framecnt} = char(F_Local_Flag_to_String(frame_data(framecnt).flag));
        else
            flags_per_frame{framecnt} = '0';
        end
        if showFrames
            points = points_per_frame{framecnt};
            scatter3(points(:, 1), points(:, 2), points(:, 3), 10, 'filled');
            xlim([-4, 4]); ylim([-3, 3]); zlim([-1, radarHeight + 0.5]);
            xlabel('x'); ylabel('y'); zlabel('z'); title(num2str(framecnt)); drawnow;
        end
    end
    segments = struct('start_frame', {}, 'end_frame', {}, 'flag', {});
    seg_start = 1;
    for t = 2:numFrames + 1
        if t > numFrames || ~strcmp(flags_per_frame{t}, flags_per_frame{seg_start})
            segments(end + 1) = struct('start_frame', seg_start, ...
                'end_frame', t - 1, 'flag', flags_per_frame{seg_start}); %#ok<AGROW>
            seg_start = t;
        end
    end
    preprocessing = struct('type', 3, 'radar_height', radarHeight, ...
        'legacy_empty_frame_sentinel', true);
    save(save_name, 'points_per_frame', 'flags_per_frame', 'segments', 'preprocessing', '-v7.3');
    fprintf('Saved %s (%d frames)\n', save_name, numFrames);
end
end
