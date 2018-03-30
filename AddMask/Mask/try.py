from ctypes import *

lib = cdll.LoadLibrary('lib/libmossaic.so')
add_mask = lib.add_mossaic
add_mask.argtypes = [c_char_p, c_char_p, c_char_p, c_float]
#  add_mask.restype = c_void_p

org = "./pics/batch1-42_org.jpg"
hm = "./pics/batch1-42_hm.jpg"
out = "./pics/batch1-42_merge.jpg"

org = org.encode('utf-8')
hm = hm.encode('utf-8')
out = out.encode('utf-8')

add_mask(org, hm, out, 0.5)
#  add_mask("./pics/batch1-42_org.jpg", "./pics/batch1-42_hm.jpg", "./pics/batch1-42_merge.jpg", 0.5)


