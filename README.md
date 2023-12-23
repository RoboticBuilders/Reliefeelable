# Reliefeelable
Repository of code for Creating 3D tactile displays using 2d Images for paintings.

3D models of paintings
Our approach to getting 3D models from paintings is straightforward and includes just 3 steps:
1. Get the depth map of the image,
2. Add images features like edges & contours into the depth map,
3. Convert the depth map into a 3D model by converting depth to 3D pixel height on the Z-axis.

# Depth maps from monocular images 
Depth maps provide depth info for each pixel in the image.

## Original image
![Great Wave](docs/great_wave.png "Great Wave")

## Depth map
![Great Wave](docs/depth_map.png "Depth Map")

The map presents the picture's world in terms of depth from the viewer. Mt. Fuji at the center of
the image is clearly further away from the viewer than the waves. This is very useful information
about the picture. There are a few options when it comes to getting depth maps from monocular images, and we
prototyped with two models that are described below.


