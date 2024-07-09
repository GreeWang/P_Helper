from PIL import Image

#find the picture which resolusion ratio is the biggest
def find_largest_resolusion_ratio(png_paths_list):
    
    if len(png_paths_list) == 0:
        return None
    
    biggest_path = png_paths_list[0]
 
    #分流器
    if len(png_paths_list) == 1:
        return biggest_path
    
    elif len(png_paths_list) > 1: 
        for i in range(len(png_paths_list)):
            biggest_image = Image.open(png_paths_list[0])
            image = Image.open(png_paths_list[i])
            if image.size <= biggest_image.size:#return a tuple
                continue
            else:
                biggest_path = png_paths_list[i]
                
        return biggest_path   
        
          
# example:  
# intro_picture_path = find_largest_resolusion_ratio(png_paths_list)

        