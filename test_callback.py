
import discord
from discord.ext import commands
import torch, diffusers, transformers, torchvision
from dotenv import load_dotenv
import asyncio, functools
import nest_asyncio
import os

load_dotenv()
discord_token = os.getenv('DISCORD_TOKEN')
intents = discord.Intents.all()
intents.message_content = True
bot = commands.Bot(command_prefix="-", intents=intents)
nest_asyncio.apply()
pipe = None

def callback(step, timestep, latents, msg, pipe, total_steps):
    with torch.no_grad():
        latents = 1 / 0.18215 * latents
        image = pipe.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        image = pipe.numpy_to_pil(image)[0]

        filepath = f'outputs/steps/{step}.png'
        image.save(filepath)
        
        print(f' {step}/{total_steps}')
        loop = asyncio.get_event_loop()
        loop.run_until_complete(msg.edit(f'step {step}/{total_steps}', file=discord.File(filepath)))

@bot.command()
async def dream(ctx, *prompt):
    global pipe
    prompt = ' '.join(prompt)
    msg = await ctx.send('generating...')
    if not isinstance(pipe, diffusers.StableDiffusionPipeline):
        scheduler = diffusers.DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False)
        pipe = diffusers.StableDiffusionPipeline.from_pretrained(
            "./stable-diffusion-v1-5",
            revision="fp16",
            torch_dtype=torch.float16,
            safety_checker=None,
            scheduler=scheduler).to("cuda")

    contextual_callback = functools.partial(callback, msg=msg, pipe=pipe, total_steps=50)
    with torch.autocast("cuda"), torch.inference_mode():
        image = pipe(prompt,
            callback=contextual_callback,
            callback_steps=10).images[0]
        filepath = "outputs/image.png"
        image.save(filepath)
        print('done')

    loop = asyncio.get_event_loop()
    loop.run_until_complete(msg.edit(f'done', file=discord.File(filepath)))
    # loop.close()

print('online')
bot.run(discord_token)
