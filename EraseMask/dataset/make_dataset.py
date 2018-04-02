#!/usr/bin/python

import os


def main():
    #  bird = os.listdir('./bird')
    #  fish = os.listdir('./fish')
    #  motor = os.listdir('motorbike')
    #  [os.rename('./bird/'+fn, './all/bird_'+str(i)+'.JPEG') for i, fn in enumerate(bird)]
    #  [os.rename('./fish/'+fn, './all/fish'+str(i)+'.JPEG') for i, fn in enumerate(fish)]
    #  [os.rename('./motorbike/'+fn, './all/motor'+str(i)+'.JPEG') for i, fn in enumerate(motor)]

    #  all_img = os.listdir('./JPEG_IMG')
    #  [os.rename('./JPEG_IMG/'+fn, './JPEG_IMG/'+fn+'.JPEG') for fn in all_img]
    all_img = os.listdir('./JPEG_IMG')
    all_path = ['./JPEG_IMG/'+fn for fn in all_img]
    print(all_path[:10])
    #  [os.rename(fn, fn.replace('fish', 'fish_')) for fn in all_path if 'fish' in fn]
    [os.rename(fn, fn.replace('motor', 'motor_')) for fn in all_path if 'motor' in fn]

if __name__ == '__main__':
    main()
