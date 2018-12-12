# Switch the working fold to partner path
import sys
import os
file_folder = globals()['_dh'][0]
wk_dir = os.path.dirname(file_folder)
os.chdir(wk_dir)



 
# Add high priority python path
import sys
sys.path.insert(0, "../")