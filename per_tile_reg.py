import numpy as np 
import pandas as pd 
import cv2 as cv 
import os 
from pycpd import RigidRegistration
from PIL import Image

def apply_registration(reg=None, image=None, filename=None, output_path=None):
    ''' 
    reg = class object: pycpd.RigidRegistration()
    image = cv.Image OR /path/to/tiff
    output_path
    '''
    if type(image) == str: 

        image = cv.imread(image, -1)
    else: 
        image = image 

    original_coords = np.moveaxis(np.indices(image.shape), 0, 2) # image to x,y coords
    intensities = np.array(image.copy().reshape(image.shape)) # image values to 1D array
    allArray = np.dstack((original_coords, intensities)).reshape((-1, 3)) 
    df = pd.DataFrame(allArray, columns = ['x','y','intensity'])
    
    # builtin function that applies transformation matrices to new point cloud
    trans_coords = reg.transform_point_cloud(df[['y','x']].to_numpy()) 
    trans_df = pd.DataFrame(trans_coords, columns = ['y','x'])
    trans_df['x'] = trans_df['x'].astype(int)
    trans_df['y'] = trans_df['y'].astype(int)

    trans_df['intensity'] = df['intensity'].to_numpy() 
    image_array = np.zeros((256, 256), dtype=np.uint16)

    for _, row in trans_df.iterrows():
        # write new image pixel by pixel
        if row['x'] > 255 or row['y'] > 255 or row['x'] < 0 or row['y'] < 0:
            continue
        x = row['x']
        y = row['y']

        intensity = row['intensity']
        image_array[x,y] = intensity

    new_image = Image.fromarray(image_array)
    
    if output_path == None: 
        output_path = '.'
        filename = 'test'

        new_image.save(f'{output_path}/{filename}.tif')
        print(f"Registered image saved to {output_path}/{filename}.tif")
    else:
        new_image.save(f'{output_path}/{filename}.tif') 
        print(f'{filename}.tif')

def main(): 



    df = pd.read_csv('matched_points_tile3.csv') 
    df = df.sort_values(['Cycle', 'Point Num'])
    df['registered'] = False

    cycle_key = {
        1:'Well9_Barcode4_Cycle1',
        2:'Well9_Barcode4_Cycle2',
        3:'Well9_Barcode4_Cycle3',
        4:'Well9_Barcode4_Cycle4',
        5:'Well9_Barcode3_Cycle1',
        6:'Well9_Barcode3_Cycle2'

    }

    data_dir = 'path/to/tif/dir'
    output_path = 'path/to/output/' 

    try:
        
        output_path = '.' 
        os.makedirs(output_path)
    except FileExistsError: 
        pass 
    for cycle in range(1,7): 
        
        target = df.loc[df.Cycle == 1, ['X', 'Y']].to_numpy()
        source = df.loc[df.Cycle == cycle, ['X', 'Y']].to_numpy()

        reg = RigidRegistration(X=target, Y=source)
        TY, (s, T, t) = reg.register() 
        filename_prefix = cycle_key[cycle] 
        
        for tile in range(36): # Tiles were 6x6 ROIs
            for channel in range(16): # 16 imaging channels for 4x emission and 4x excitation
                image = f'{data_dir}/{filename_prefix}_tile-{tile}_channel-{channel}.tif'
                filename = f'registered_{filename_prefix}_tile-{tile}_channel-{channel}'

                apply_registration(reg=reg, image = image, filename=filename, output_path=output_path )


if __name__ == '__main__': 
    main()