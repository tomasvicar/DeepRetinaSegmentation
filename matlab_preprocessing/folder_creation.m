function [] = folder_creation (of)
if ~exist([of '\Images'], 'dir')
    mkdir([of '\Images'])
end
if ~exist([of '\Vessels'], 'dir')
    mkdir([of '\Vessels'])
end
if ~exist([of '\Disc'], 'dir')
    mkdir([of '\Disc'])
end
if ~exist([of '\Cup'], 'dir')
    mkdir([of '\Cup'])
end
if ~exist([of '\Fov'], 'dir')
    mkdir([of '\Fov'])
end
end