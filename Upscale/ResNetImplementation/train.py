import copy
import os
import torch
import torch.nn as nn

def train_loop(experimentlogger,generator, discriminator, dataloader, optimizer_G, optimizer_D, adversarial_loss=nn.BCELoss(),total_loss=None,device=torch.device("cpu"),start_epoch=0,end_epoch=10000):

    # Training loop
    full_path = os.path.abspath(__file__)
    base_path = os.path.dirname(full_path)
    save_path= "/used_data"

    # Prepare
    discriminator = discriminator.to(device)
    generator = generator.to(device)
    adversarial_loss = adversarial_loss.to(device)

    for epoch in range(start_epoch,end_epoch,1):
        upscaled=None
        for i, (feature, label) in enumerate(dataloader):

            feature=feature.to(device)
            label=label.to(device)

            # Adversarial ground truths
            valid = torch.ones([feature.shape[0], 1, 1, 1]).to(device)
            fake = torch.zeros([feature.shape[0], 1, 1, 1]).to(device)

            upscaled = generator(feature)

            # ---------------------
            # Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()
            real_loss = adversarial_loss(discriminator(label), valid) #real are real
            fake_loss = adversarial_loss(discriminator(upscaled.detach()), fake) #fake are fake
            d_loss = (real_loss + fake_loss) / 2

            # Backward pass and optimize
            d_loss.backward()
            optimizer_D.step()

            # -----------------
            # Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Calculate adversarial loss
            g_loss = adversarial_loss(discriminator(upscaled), valid)

            # Augment with perceptual loss if defined
            if total_loss is not None:
                g_loss = total_loss(g_loss, upscaled, label, epoch, end_epoch)

            # Backward pass and optimize
            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            # Progress Monitoring
            # ---------------------

            if (i + 1) % 100 == 0:
                print(
                    f"Epoch [{epoch+1}/{end_epoch}]\
                            Batch {i+1}/{len(dataloader)} "
                    f"Discriminator Loss: {d_loss.item():.4f} "
                    f"Generator Loss: {g_loss.item():.4f}"
                )


        # Save copy of model every epoch
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                generator_cpu=copy.deepcopy(generator)
                generator_cpu=generator_cpu.to("cpu")
                discriminator_cpu=copy.deepcopy(discriminator)
                discriminator_cpu=discriminator_cpu.to("cpu")
                save_path=os.path.join(experimentlogger.get_dir_path(),f'train_loop_{epoch}.pth.tar')
                torch.save({'epoch': epoch,
                            'model': generator_cpu,
                            'discrim': discriminator_cpu},
                           save_path)
                experimentlogger.add_model_snapshot(save_path)

def train_generator(experimentlogger,generator, dataloader, optimizer_G, loss, device=torch.device("cpu"),end_epoch=25):

    # Training loop
    full_path = os.path.abspath(__file__)

    # Prepare
    generator = generator.to(device)
    loss = loss.to(device)

    for epoch in range(0,end_epoch,1):
        for i, (feature, label) in enumerate(dataloader):

            feature=feature.to(device)
            label=label.to(device)

            # -----------------
            # Train Generator
            # -----------------

            optimizer_G.zero_grad()
            upscaled=generator(feature)

            # Calculate loss
            g_loss = loss(upscaled, label)

            # Backward pass and optimize
            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            # Progress Monitoring
            # ---------------------

            if (i + 1) % 100 == 0:
                print(
                    f"Epoch [{epoch+1}/{end_epoch}]\
                            Batch {i+1}/{len(dataloader)} "
                    f"Generator Loss: {g_loss.item():.4f}"
                )
