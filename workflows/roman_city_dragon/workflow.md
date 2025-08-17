# Roman City Dragon

## Description
A dragon is flying over an ancient Roman city and burns it to the ground.

# Steps
1. First step was creating the image of a dragon using Qwen Image. 
    - This involved getting a prompt from ChatGPT and Gemini 2.5 Pro. 
    - Tried using Flux Dev and Flux Krea for the image, but Qwen Image was the best.
2. 
    - I have now gone and used the salientmasktrack inpainting method to create a mask of the original dragon, then replace the dragon with the new base image.
    - This was based on the original image to video from the flux kontext output of the dragon on top of the mountain. 
## Notes
Reference anything with the two images didn't really work all too well. Will try again with the official implementation. 
My suspiscion is that you need to individual objects, not just one object and then a background which is what I provided. 
Will additionally try Stand In later to see if we can get better results. 
Will also try to use VACE inpainting and maybe fun camera control to see if we can inpaint a location to have the dragon be super small over top of the mountain
Then use camera control for the camera to zoom into the top of the mountain accordingly. 

