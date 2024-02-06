# Results: Upscaled Modded Textures - using GANs

Hi, welcome to my project! 
This codebase implements a Generative Adversarial Network (GAN) utilizing ResNet (generator) in a structured pipeline for Minecraft texture generation.

Please refer to [how_to_use](https://github.com/gametekker/InterpretableGAN/blob/main/Upscale/how_to_use.md) for setting up.

## Currently supports 16x -> 32x, 64x
![Example Image](https://github.com/gametekker/InterpretableGAN/blob/main/Screen%20Shot%202023-09-24%20at%203.49.37%20AM.png)
Industrial craft 2
![Example Image](https://github.com/gametekker/InterpretableGAN/blob/main/Screen%20Shot%202023-09-24%20at%205.43.07%20PM.png)
Industrial craft 2
![Example Image](https://github.com/gametekker/InterpretableGAN/blob/main/Screen%20Shot%202023-09-24%20at%203.53.10%20AM.png)



## Adversarial vs perceptual loss only

![Example Image](https://github.com/gametekker/InterpretableGAN/blob/main/Screen%20Shot%202023-09-24%20at%203.17.15%20AM.png)
500 epochs, adversarial loss annealing + perceptual loss
![Example Image](https://github.com/gametekker/InterpretableGAN/blob/main/Screen%20Shot%202023-09-24%20at%203.17.22%20AM.png)
500 epochs, perceptual loss only

# Context: Minecraft Texture Upscaler - Bridging Vanilla and Modded Textures

## Problem Statement:

Minecraft, as a sandbox game, offers an immersive world where textures play a pivotal role in enhancing the player's experience. With the evolution of the game, there has been a surge in community contributions in the form of mods and texture packs:

### Mod Blocks:

There are currently 100,000 Minecraft mods available, many of which introduce new blocks with distinctive 16x16 textures.

### Texture Packs: 

To further embellish the game, there are thousands of Minecraft texture packs that provide high-resolution textures, enriching the visual depth of the game environment.

While these contributions significantly expand the game's potential, they inadvertently introduce a dilemma:

Most players prefer the game's default textures. However, when they incorporate custom high-resolution texture packs, mod blocks, which usually conform to the 16x16 vanilla style, starkly contrast the refined environment. This disparity is visually jarring, disrupting the seamless aesthetics of the game and consequently breaking immersion.

Moreover, given the myriad of mod combinations players might use, crafting custom textures that harmonize with both the chosen texture pack and the original mod style becomes an intricate challenge.

## Proposed Solution:

To address this texture inconsistency, we introduce a framework proficient in upscaling 16x16 vanilla textures to align with the artistic style of high-resolution texture packs.

### Learning the Art: 

By training on vanilla textures, our model learns the intricate nuances of upscaling from 16x16 to higher resolutions while preserving and adapting the stylistic essence of the texture pack. This learned capability ensures that upscaled textures resonate with the player's chosen texture pack, establishing a coherent visual theme.

### Modded Textures Upscaling: 

The true strength of our framework shines when it's applied to modded textures. It efficiently scales the 16x16 modded textures, ensuring that they not only elevate in resolution but also seamlessly blend with the texture pack's style without forsaking the original design ethos of the mod.

## Conclusion:

Minecraft Texture Upscaler is more than just a tool; it's a bridge that harmonizes the vibrant world of Minecraft mods with the artistic brilliance of texture packs. For players, it's a gateway to a unified, immersive, and visually stunning Minecraft experience, unfettered by the inconsistencies of different texture resolutions.
