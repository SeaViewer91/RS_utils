'''
디렉토리 내에 존재하는 mov 형식 비디오를 순차적으로 이어붙여 하나의 mp4 파일로 만들어주는 도구

낙동강 환경감시 업무 지원을 위해 개발됨(2024-06-27).
'''

import os
import subprocess
import datetime

def concatenate_mov_to_mp4_gpu(input_directory, output_filename):
    # Step 1: Get list of mov files sorted by index
    mov_files = sorted([f for f in os.listdir(input_directory) if f.lower().endswith('.mov')], # 디렉토리 내에 저장된 mov 파일의 리스트 생성
                       key=lambda x: int(x.split('_')[-1].split('.')[0])) # 동영상 파일명이 'DJI_0001.mov'와 같이 저장되어 있으므로, 언더바 이후 숫자를 기준으로 파일 리스트를 재정렬
    
    # Step 2: Create a temporary file list for FFmpeg
    file_list_path = os.path.join(input_directory, 'file_list.txt')
    with open(file_list_path, 'w') as file_list:
        for mov_file in mov_files:
            file_list.write(f"file '{os.path.join(input_directory, mov_file)}'\n")
    
    # Step 3: Build FFmpeg command to concatenate videos with GPU acceleration
    output_path = os.path.join(input_directory, output_filename)
    command = [
        'ffmpeg',
        '-f', 'concat',
        '-safe', '0',
        '-i', file_list_path,
        '-c:v', 'h264_nvenc',  # 그래픽 가속을 통해 h.264 mp4로 export
        '-vf', 'scale=1920:1080',
        '-r', '24',
        '-pix_fmt', 'yuv420p',
        '-y', output_path
    ]
    
    # Step 4: Run the FFmpeg command with GPU acceleration
    subprocess.run(command, check=True)
    
    # Step 5: Clean up the temporary file list
    os.remove(file_list_path)


def main():
    start_time = datetime.datetime.now()
    
    # root dir
    root_dir_path = 'D:/video/samples'
    
    dirs = [d for d in os.listdir(root_dir_path) if os.path.isdir(os.path.join(root_dir_path, d))]

    '''
    디렉토리 구조는 아래와 같아야 함.

    root_dir
    ㄴset1
      ㄴDJI_0001.mov
      ㄴDJI_0002.mov
      ㄴDJI_0003.mov
    ㄴset2
    ㄴset3
    '''
    for directory_name in dirs:
        concatenate_mov_to_mp4_gpu(
            input_directory=os.path.join(root_dir_path, directory_name), 
            output_filename=os.path.join(root_dir_path, directory_name, 'output.mp4')
        )
    
    end_time = datetime.datetime.now()
    
    duration = (end_time - start_time).seconds
    
    print(duration)
    
if __name__=='__main__':
    main()
