import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/saran/robohouse_ws/src/susag_robot_description/install/susag_robot_description'
