<h1 align="center">Apex Studio</h1>

<hr />

<p align="center">
  Content creation made easy with a video editor built around open-source models.
</p>

<p align="center">
  <img src="assets/images/apex-studio.png" alt="Apex Studio" />
</p>

<p>We built Apex around a simple belief: using free and open-source diffusion models should feel effortless. Making the model run shouldn't be the challenge, the challenge should be the ambition, depth, and complexity of the content you choose to create.</p>

## Documentation

<hr />

### [Docuementation links](#)
### [Basics](#basics)

## Getting Started

<hr />

### Desktop app

- Packaged application
- Coming soon for MacOS and Windows machines

### [Terminal startup](#terminal-startup)

- packaged studio and engine accessible via a code terminal
- Can be used with remote machines

## Features

<hr />

- Video editor built with JavaScript and Electron for easy video creation, and timeline based editing.
- Easy preprocessing by dragging your desired preprocessor model onto a media clip.
- Point-based masking with positive and negative markers for precise, intuitive control over exactly what gets masked.
- Use any LoRA from popular model hubs like Hugging Face or Civitai—bring your own checkpoints, styles, and character LoRAs and plug them directly into your workflow.
- Projects are saved as simple JSON files, making them easy to version, hand-edit and share
- No node graphs: projects stay straightforward and portable, improving cross-compatibility across machines, setups, and collaborators.
- Built-in queueing so you can line up multiple renders/generations and let Apex process them in order.
- Denoised latent previews at predetermined intervals, so you can watch generations evolve as they render.
- Built-in video controls including speed changes, frame interpolation, and keyframe selection.
- Editing controls for trimming, slicing, cropping, and much more.
- Hundreds of effects available to use within your creation.
- Audio controls including detaching audio from video, waveform manipulation, noise reduction, and more.

### Models

- **Image Models**
  - [chroma hd text to image](manifest/verified/image/chroma-hd-text-to-image-1.0.0.v1.yml)
  - [flux dev kontext](manifest/verified/image/flux-dev-kontext-1.0.0.v1.yml)
  - [flux dev text to image](manifest/verified/image/flux-dev-text-to-image-1.0.0.v1.yml)
  - [flux krea text to image](manifest/verified/image/flux-krea-text-to-image-1.0.0.v1.yml)
  - [nunchaku flux dev kontext](manifest/verified/image/nunchaku-flux-dev-kontext-1.0.0.v1.yml)
  - [nunchaku flux dev text to image](manifest/verified/image/nunchaku-flux-dev-text-to-image-1.0.0.v1.yml)
  - [nunchaku flux krea text to image](manifest/verified/image/nunchaku-flux-krea-text-to-image-1.0.0.v1.yml)
  - [nunchaku qwenimage edit 2509 lightning 8steps](manifest/verified/image/nunchaku-qwenimage-edit-2509-lightning-8steps-1.0.0.v1.yml)
  - [nunchaku qwenimage edit lightning 8steps](manifest/verified/image/nunchaku-qwenimage-edit-lightning-8steps-1.0.0.v1.yml)
  - [nunchaku qwenimage lightning 8steps](manifest/verified/image/nunchaku-qwenimage-lightning-8steps-1.0.0.v1.yml)
  - [qwenimage](manifest/verified/image/qwenimage-1.0.0.v1.yml)
  - [qwenimage edit](manifest/verified/image/qwenimage-edit-1.0.0.v1.yml)
  - [qwenimage edit 2509](manifest/verified/image/qwenimage-edit-2509-1.0.0.v1.yml)
  - [wan 2.2 a14b text to image 4 steps](manifest/verified/image/wan-2.2-a14b-text-to-image-4-steps-1.0.0.v1.yml)
  - [zimage turbo](manifest/verified/image/zimage-turbo-1.0.0.v1.yml)
  - [zimage turbo control](manifest/verified/image/zimage-turbo-control-1.0.0.v1.yml)

- **Video Models**
  - [hunyuanvideo 1.5 i2v](manifest/verified/video/hunyuanvideo-1.5-i2v-1.0.0.v1.yml)
  - [hunyuanvideo 1.5 t2v](manifest/verified/video/hunyuanvideo-1.5-t2v-1.0.0.v1.yml)
  - [wan 2.1 14b image to video 480p](manifest/verified/video/wan-2.1-14b-image-to-video-480p-1.0.0.v1.yml)
  - [wan 2.1 14b infinitetalk text to video](manifest/verified/video/wan-2.1-14b-infinitetalk-text-to-video-1.0.0.v1.yml)
  - [wan 2.1 14b multitalk text to video](manifest/verified/video/wan-2.1-14b-multitalk-text-to-video-1.0.0.v1.yml)
  - [wan 2.1 14b vace control](manifest/verified/video/wan-2.1-14b-vace-control-1.0.0.v1.yml)
  - [wan 2.1 14b vace expand swap](manifest/verified/video/wan-2.1-14b-vace-expand-swap-1.0.0.v1.yml)
  - [wan 2.1 14b vace painting](manifest/verified/video/wan-2.1-14b-vace-painting-1.0.0.v1.yml)
  - [wan 2.1 14b vace reference to video](manifest/verified/video/wan-2.1-14b-vace-reference-to-video-1.0.0.v1.yml)
  - [wan 2.2 14b animate](manifest/verified/video/wan-2.2-14b-animate-1.0.0.v1.yml)
  - [wan 2.2 5b text to image to video turbo](manifest/verified/video/wan-2.2-5b-text-to-image-to-video-turbo.1.0.0.v1.yml)
  - [wan 2.2 a14b text to video](manifest/verified/video/wan-2.2-a14b-text-to-video-1.0.0.v1.yml)
  - [wan 2.2 fun a14b control](manifest/verified/video/wan-2.2-fun-a14b-control-1.0.0.v1.yml)
  - [wan2.2 a14b first frame last frame](manifest/verified/video/wan2.2-a14b-first-frame-last-frame-1.0.0.v1.yml)
  - [wan2.2 a14b image to video](manifest/verified/video/wan2.2-a14b-image-to-video-1.0.0.v1.yml)

## Basics

<hr />
### Intro to D&Ding

Apex is built on a simple principle: **drag and drop is how you create**. Whether you're working with media clips, audio tracks, text, effects, or models, the timeline is your canvas. Simply drag any element onto your project and get started. No complex menus, no steep learning curve, just intuitive, visual creation.

### Using a model

To get started with an Apex model, first  download the model components you want to use. Then, drag the model clip onto your timeline to add it to your project. Customize the inputs to match your creative vision, and when you're ready, click **Generate**. 

### Using a preprocessor

Start by adding a media clip to your timeline. Open the properties panel on the right and navigate to the **Preprocessor** tab. Browse through our library of preprocessors to find the one that suits your needs, download it, and click the **+** button to add it to your clip. Once applied, hit **Generate** to process your media with the preprocessor

### Media inputs

For models that require a media input, you have flexibility in where it comes from. Use media from the media library, upload files directly from your file system, or use content you've already generated within your timeline. 

### Model Inputs

Fine-tune your generations with model parameters accessed through the **Model Inputs** tab. Adjust settings like dimensions, inference steps, seed, and guidance scale through dropdown menus. To change the duration,  resize the model clip on the timeline.

### Downloading a model

Add your Hugging Face and/or Civitai tokens in the settings panel. Open the model card for any model you want to use and download the model component that matches your machine and available VRAM. You can also add your own model component path directly within the model details.

### Downloading a lora

Add your Hugging Face and/or Civitai tokens in the settings panel. There are two ways to download a LoRA. Go into the model card and download pre-added LoRAs, most of which are required by the model for proper use. Alternatively, use the LoRA tab in the properties panel to add a link to a separate LoRA you want to use.

### Using the masking tool

Click the wand icon in the toolbar to start creating a mask. Add positive and negative points to define your mask area, then click track to generate your desired mask.

### Intro to video editing

Apex supports standard video editing features found on other platforms. Add filters, text, and drawing effects to your clips. Splice clips together, adjust audio with keyframing, and access many more editing tools.

### Exporting content

Export your entire project when finished, or right click on any media clip (added or model-generated) to export that individual clip to your file system.

## Terminal Startup

<hr />



