'''
Copyright (c) 2017 Intel Corporation.
Licensed under the MIT license. See LICENSE file in the project root for full license information.
'''

import time
from secrets import ssh_host, ssh_user, ssh_pass
from picamera2 import Picamera2, Preview
import paramiko

local_image_path = '/home/pi/gauge.jpg'
remote_image_path = '/home/pi/gauge.jpg'

def capture_image():
    picam2 = Picamera2()
    camera_config = picam2.create_still_configuration(main={"size": (1920, 1080)}, lores={"size": (640, 480)}, display="lores")
    picam2.configure(camera_config)
    picam2.start()
    picam2.capture_file(local_image_path)
    picam2.stop()

def upload_image():
    # Connect
    ssh_client =paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh_client.connect(hostname=ssh_host,username=ssh_user,password=ssh_pass)

    # Copy file
    sftp_client=ssh.open_sftp()
    sftp_client.put(local_image_path, remote_image_path)
    sftp_client.close()

def main():
    capture_image()
    upload_image()

if __name__=='__main__':
    main()

