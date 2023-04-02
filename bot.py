import discord
from discord.ext import commands
from dotenv import load_dotenv

import torch, diffusers
from gpt4 import gpt4

import asyncio, functools, nest_asyncio
import numpy, re, os, time, random, argparse
from PIL import Image, ImageOps

# functions

def callback(step, timestep, latents, msg, pipe, message, total_steps):
    with torch.no_grad():
        latents = 1 / 0.18215 * latents
        image = pipe.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        image = pipe.numpy_to_pil(image)[0]

        filepath = f'outputs/steps/{step}.png'
        image.save(filepath)

        loop = asyncio.get_event_loop()
        loop.run_until_complete(msg.edit(f'{message}, step {step}/{total_steps}', file=discord.File(filepath)))

def ar_to_dim(ar, max_pixels=300000, mod=64):
    width, height = map(int, ar.split(':'))
    scale = int((max_pixels/(width*height)) ** 0.5)
    return (scale*width)//mod*mod, (scale*height)//mod*mod


# init

intents = discord.Intents.all()
intents.message_content = True
bot = commands.Bot(command_prefix="-", intents=intents)

load_dotenv()
discord_token = os.getenv('DISCORD_TOKEN')
openai_api_key = os.getenv('GPT4_TOKEN')

pipe = None
chats = {}
nest_asyncio.apply()
random.seed(time.time())

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=1, help="How many images to create")
parser.add_argument("--scale", type=float, default=7.5, help="Guidance scale")
parser.add_argument("--steps", type=int, default=50, help="Number of generation steps")
parser.add_argument("--ar", type=str, default="1:1", help="Aspect ratio")
parser.add_argument("--seed", type=int, default="0", help="Aspect ratio")
parser.add_argument("--no", nargs=argparse.REMAINDER, default="", help="Negative prompt")
# parser.add_argument('--enhance', action='store_true', help="Use GPT to enhance your prompt")


# commands

@bot.command()
async def chat(ctx, *, prompt: str, args=None):
    user = re.sub(r'\W+', '', ctx.author.name)
    chats.setdefault(user, gpt4(openai_api_key))
    if prompt.lower() == "clear":
        chats[user].clear()
        return await ctx.send("Chat history cleared.")
    await ctx.send(chats[user].call(prompt))

@bot.command()
async def stream(ctx, *, prompt: str):
    user = re.sub(r'\W+', '', ctx.author.name)
    chats.setdefault(user, gpt4(openai_api_key))
    msg = await ctx.send("Processing...")
    text = ""
    for chunk in chats[user].stream(prompt):
        text += chunk
        await msg.edit(content=text)

@bot.command()
async def sys(ctx, *, sys_prompt: str):
    user = re.sub(r'\W+', '', ctx.author.name)
    chats.setdefault(user, gpt4(openai_api_key, sys=sys_prompt)).sys = sys_prompt
    await ctx.send(f"System prompt {'updated to' if user in chats else 'set to'}: {sys_prompt}")

@bot.command()
async def dream(ctx, *, prompt: str, args=None):
    global pipe
    arg_start = prompt.find('--')
    prompt, args = (prompt[:arg_start].strip(), parser.parse_args(prompt[arg_start:].strip().split())) if arg_start != -1 else (prompt, parser.parse_args([]))
    args.no = " ".join(args.no)
    width, height = ar_to_dim(args.ar)

    if not isinstance(pipe, diffusers.StableDiffusionPipeline):
        scheduler = diffusers.DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
        pipe = diffusers.StableDiffusionPipeline.from_pretrained(
            "./stable-diffusion-v1-5",
            revision="fp16",
            torch_dtype=torch.float16,
            safety_checker=None,
            scheduler=scheduler).to("cuda")

    for i in range(args.n):
        args.seed = random.randrange(0, 2**32) if args.seed == 0 else args.seed
        rng = torch.Generator("cuda").manual_seed(args.seed)

        msg_text = f"dreaming of `{prompt}` with seed {args.seed} ({i+1}/{args.n})"
        msg = await ctx.send(msg_text)
        start_time = time.time()

        contextual_callback = functools.partial(callback, msg=msg, pipe=pipe, message=msg_text, total_steps=args.steps)
        with torch.autocast("cuda"), torch.inference_mode():
            image = pipe(prompt, callback=contextual_callback, callback_steps=10, generator=rng, negative_prompt=args.no, width=width, height=height, num_inference_steps=args.steps, guidance_scale=args.scale).images[0]
        
        filepath = f'outputs/{len(os.listdir("outputs"))}.png'
        image.save(filepath)

        msg_text = f"\"{prompt}\" by {ctx.author.mention} in {int(time.time() - start_time)}s with seed {args.seed} ({i+1}/{args.n})"
        loop = asyncio.get_event_loop()
        loop.run_until_complete(msg.edit(msg_text, file=discord.File(filepath)))
        
@bot.command()
async def readme(ctx):
    try:
        with open('README.md', 'r') as file:
            content = file.read()
    except FileNotFoundError:
        await ctx.send("README.md not found.")
        return

    for i in range(0, len(content), 2000):
        await ctx.send(f"{content[i:i+2000]}")

print('online!')
bot.run(discord_token)
