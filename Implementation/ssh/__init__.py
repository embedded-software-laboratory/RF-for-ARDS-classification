import string

from sshtunnel import SSHTunnelForwarder



def open_ssh(ssh_ip: string, ssh_user: string, ssh_password: string, forward_ip: string, forward_port: int) -> SSHTunnelForwarder:
    """Opens a ssh tunnel to the ssh server specified in ssh_ip, Username and password for the ssh_user 
    are stored in the respective variable, forward ip and port define ..."""
    tunnel = SSHTunnelForwarder((ssh_ip, 22), ssh_password=ssh_password, ssh_username=ssh_user,
                                remote_bind_address=(forward_ip, forward_port))
    tunnel.start()
    return tunnel