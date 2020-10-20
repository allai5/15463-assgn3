# 15463-assgn3

## 1. Bilateral Filtering (filter.py)
### Basic Bilateral Filtering

    - L85: bilateral_filter(self, flash, sigs, sigr)
    - flash is a parameter to apply the bilateral filter on either the flash
    image or the ambient image (flash = 1 means use the flash image)

### Joint Bilateral Filtering

    - L93: joint_filter(self, sigs, sigr):

### Detail Transfer

    - L101: detail_transfer(self, eps)
    - eps is the parameter for computing the detail portion of the image (as
    denoted in the algorithm in the paper for detail transfer)

### Mask-Based Merging

    - L122: shadow_mask(self)
    - L140: specular_mask(self)
    - L85: apply_mask()

## 2. Gradient-Domain Processing (gradient.py)

    - L76: gradient_field_test(self)
    - L159: fuse_gradient_test(self)


## 3. Capture your own flash/no-flash pairs

    - cow_LED_ambient.png/cow_LED_flash.png is the flash/no-flash pair for
    applying the denoising techniques based on bilateral filtering
    - cow_bowl_ambient.png/cow_bowl_flash.png is the flash/no-flash pair for
    applying the fusion algorithm based on gradient-domain processing

### Running the code

    - main.py creates objects of the classes defined in gradient.py and
    filter.py, and the calls the relevant methods
    - Task1_photos/, Task2_photos/, and Task3_photos contain all the image file
    outputs for the assignment

