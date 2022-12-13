'''
Copyright (c) 2017 Intel Corporation.
Licensed under the MIT license. See LICENSE file in the project root for full license information.
'''

import time
from secrets import ssh_host, ssh_user, ssh_pass
from picamera2 import Picamera2, Preview
import paramiko # sudo apt-get install -y python3-paramiko

local_image_path = '/home/pi/dev/capture_gauge/gauge.jpg'
remote_image_path = '/home/pi/dev/gauge_reader/gauge.jpg'

#capture_frequency = 1

def setup_camera():
    picam2 = Picamera2()
    Picamera2.set_logging(Picamera2.ERROR)
    camera_config = picam2.create_still_configuration(main={"size": (1920, 1080)}, lores={"size": (640, 480)}, display="lores")
    picam2.configure(camera_config)
    return picam2

def capture_image(picam2):
    picam2.start()
    picam2.capture_file(local_image_path)
    picam2.stop()

    print('Image saved!')

def upload_image():
    # Connect
    print('Starting SSH client...')
    ssh_client = paramiko.SSHClient()
    ssh_client.load_system_host_keys()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh_client.connect(hostname=ssh_host, port=22, username=ssh_user, password=ssh_pass)
    print('SSH Client connected...')

    # Copy file
    sftp_client = ssh_client.open_sftp()
    sftp_client.put(local_image_path, remote_image_path)
    sftp_client.close()

    print('File sent!')

def main():
    picam2 = setup_camera()
    capture_image(picam2)
    upload_image()

    #try:
    #    picam2 = setup_camera()
    #    while True:
    #        wait_for_connection()
    #        capture_image(picam2)
    #        upload_image()
    #        time.sleep(capture_frequency * 60)
    #except KeyboardInterrupt:
    #    print('interrupted!')

if __name__=='__main__':
    main()

