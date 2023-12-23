# Reliefeelable
Repository of code for Creating 3D tactile displays using 2d Images for paintings.


3D models of paintings
Our approach to getting 3D models from paintings is straightforward and includes just 3 steps:
1. Get the depth map of the image,
2. Add images features like edges & contours into the depth map,
3. Convert the depth map into a 3D model by converting depth to 3D pixel height on the
Z-axis.

# Depth maps from monocular images 
Depth maps provide depth info for each pixel in the image.

## Original image
![Great Wave](docs/Great_Wave.jpg "Great Wave")
 
## Depth map
The map presents the picture's world in terms of depth from the viewer. Mt. Fuji at the center of the image is clearly further away from the viewer than the waves. This is very useful information about the picture. There are a few options when it comes to getting depth maps from monocular images, and we prototyped with two models that are described below.

![Alt Text](docs/depth_map.jpg "Depth Map")


# Adding features into a depth map
The challenge with depth maps is that you lose more subtle features of the image. In the case of the Great Wave depth map, it has lost details about the boat and the foam from the wave. For a lot of famous paintings, these features are critical to the perception of the picture. Think Mona Lisa’s smile. Showing just the depth map, while better than nothing, isn’t enough to do justice to the quality of paintings. We need to find a way to bring back the details of the image.
 
 One way to do so is to bring in the edges and contours of the image. We can use many of the common edge and contour detections algorithms to introduce features into the depth map. Consider this image of the Great Wave that has been processed with Canny edge detector:
 As you can see, many of the key features of the image are included here. It does not have depth info, so it cannot be used by itself in a 3D model of the image. For this, we need to combine the depth map with the edges feature map. Here is an example of combination at a 90:10 ratio:

The net effect is that the depth info dominates, but the features of the wave and boats now start showing up. This can present a fuller sense of the painting.

Converting depth-maps to STLs
Now that we have a full featured depth map, our next task is to convert it into a 3D model that can be rendered via a 3D printer or a robotic tactile display. For this, we need to convert the depth info in each 2D pixel of the image to height on the Z-axis of the 3D model.
We start with the depth map image, where each pixel contains a color representation of the depth of that pixel in the image’s world. In the example below, a white pixel means that the pixel is closer to the viewer, black means it is further away, and the grays are somewhere in between.

  We start off by keeping the X & Y axis of the image to maintain its dimension in the 3D model. Think of this as the base of the 3D frame.
We then assign height to the pixel in 3D space according to the color of the pixel in 2D space. This will allow the 3D model to represent the depth of the pixel.
 
  What we have done here is convert the 2D depth map into a 3D model of the image. So a white pixel is taller than gray ones, which in turn is taller than the black pixel. The taller the pixel, the closer it appears/feels to the observer.
In real images, the neighboring pixels usually have similar heights, giving the overall model a smooth feel.The edges in depth/features pop out and provide tactile feedback about object boundaries to a user.
This 3D model can be saved as an STL file and 3D printed to get a real tactile display. The Great Wave off Kanagawa looks like this in a 3D model:

  As you can see, the wave has depth and can be sensed by feeling the image. This is the key idea of our tactile displays.

 Further reading about depth maps Models for getting depth maps
intel/dpt-large
Model available as a HuggingFace transformer and we have a prototype that gets depth maps from it.
Let’s see an example with Leonardo DaVinci’s The Last Supper:
 
  DinoV2
This model is a lot more advanced when it comes to getting depth info, it can get subtle details like the individual shapes and table details. This model is not publically available, and can only be accessed via a demo page from Meta.
 
