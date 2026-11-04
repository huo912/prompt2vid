import sys
import paramiko

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HOST = "connect.westb.seetacloud.com"
PORT = 23246
USER = "root"
PASSWORD = "zSW27+LKbW9h"


def main():
    if len(sys.argv) > 1 and sys.argv[1].endswith(".sh"):
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(HOST, port=PORT, username=USER, password=PASSWORD,
                       timeout=30, banner_timeout=30, auth_timeout=30)
        try:
            sftp = client.open_sftp()
            sftp.put(sys.argv[1], "/tmp/_infcode_script.sh")
            sftp.close()
            stdin, stdout, stderr = client.exec_command("bash /tmp/_infcode_script.sh", timeout=900)
            out = stdout.read().decode("utf-8", errors="replace")
            err = stderr.read().decode("utf-8", errors="replace")
            if out:
                print(out)
            if err:
                print("--- STDERR ---")
                print(err)
        finally:
            client.close()
    else:
        print("usage: python .ssh_exec_script.py <script.sh>")


if __name__ == "__main__":
    main()
