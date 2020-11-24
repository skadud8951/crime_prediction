import ffmpeg
from moviepy.video.io.ffmpeg_tools import *
from bs4 import BeautifulSoup


start_frame=6690
end_frame=7030
start_point = start_frame/30
end_point = end_frame/30

ffmpeg_extract_subclip("./outsidedoor_01/12-2/12-2_cam02_assault01_place09_day_summer.mp4",
                       start_point, end_point, targetname="./VideoFile/punching/punching_93.mp4")