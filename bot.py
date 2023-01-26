import discord
from discord.ext import commands

import torch, diffusers, transformers
import whisper, openai
import colorthief, youtube_dl
import requests, numpy, math, re, os, time, random
from datetime import datetime

from dotenv import load_dotenv
from io import BytesIO
from PIL import Image, ImageOps

import asyncio, functools
import nest_asyncio

def parse_prompt(prompt):
    arg_words = [t for t in prompt if '=' in t]
    kwargs = dict(t.split('=') for t in arg_words) if arg_words is not [] else {}
    prompt = ' '.join([t for t in prompt if '=' not in t])
    if '[' in prompt:
        negative_prompt = prompt.split('[')[1].split(']')[0]
        prompt = "".join([prompt.split('[')[0], prompt.split(']')[1]])
        kwargs['negative_prompt'] = negative_prompt
    return prompt, kwargs
  
def save_img(image, folder):
    gen_count = len(os.listdir(f'{folder}'))
    img_fname = f'{folder}/{gen_count}.png'
    image.save(img_fname)
    return img_fname
  
def call_gpt3(prompt, temp=0.8, tokens=100):
    response = openai.Completion.create(model='text-davinci-003', prompt=prompt, temperature=temp, max_tokens=tokens)
    return response.choices[0].text
  
def rgb_to_hex(rgb):
    return '%02x%02x%02x' % rgb
 
def PIL_from_url(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content)).convert('RGBA')
  
def save_audio_from_url(url, folder):
    gen_count = len(os.listdir(f'{folder}'))
    audio_fname = f'{folder}/{gen_count}.{url.split(".")[-1]}'
    response = requests.get(url)
    audiofile = response.content.strip()
    with open(audio_fname, 'wb') as f:
        f.write(audiofile)
        f.close()
    return audio_fname

def restart_program():
    SCRIPT_FILE_NAME = os.path.basename(__file__)
    print("Restarting...")
    run_string = 'python "'+ SCRIPT_FILE_NAME + '"'
    print(run_string)
    subprocess.Popen(run_string)
    exit(0)

def callback(step, timestep, latents, msg, pipe, message):
    with torch.no_grad():
        latents = 1 / 0.18215 * latents
        image = pipe.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        image = pipe.numpy_to_pil(image)[0]

        filepath = f'outputs/steps/{step}.png'
        image.save(filepath)

        nest_asyncio.apply()
        asyncio.get_event_loop().run_until_complete(msg.edit(f'{message}, step {step}', file=discord.File(filepath)))

# init pycord
intents = discord.Intents.all()
intents.message_content = True
bot = commands.Bot(command_prefix="-", intents=intents)

# load auth tokens
load_dotenv()
discord_token = os.getenv('DISCORD_TOKEN')
openai.api_key = os.getenv('OPENAI_TOKEN')

# global variables
pipe = None

random.seed(time.time())
print('started')

@bot.command()
async def dream(ctx, *prompt):
    global pipe
    prompt, kwargs = parse_prompt(prompt)
    kwargs = {
        # 'n': int(kwargs['n']) if 'n' in kwargs else 1,
        'num_inference_steps': int(kwargs['steps']) if 'steps' in kwargs else 50,
        'guidance_scale': float(kwargs['scale']) if 'scale' in kwargs else 7.5,
        'height': int(kwargs['height']) if 'height' in kwargs else 512,
        'width': int(kwargs['width']) if 'width' in kwargs else 512,
        'negative_prompt': kwargs['negative_prompt'] if 'negative_prompt' in kwargs else ''
    }
    negative_prompt = kwargs['negative_prompt']
    
    # initialize model
    if not isinstance(pipe, diffusers.StableDiffusionPipeline):
        scheduler = diffusers.DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
        pipe = diffusers.StableDiffusionPipeline.from_pretrained(
            "./lyra-diffusion-v1-5",
            # "./lyra_LoRA-v1-5",
            revision="fp16",
            torch_dtype=torch.float16,
            safety_checker=None,
            scheduler=scheduler).to("cuda")

        # load finetunes
        if 'lora' in kwargs:
            pipe.unet.load_attn_procs(f"lora/{kwargs['lora']}")
    
    # generate n images
    n_images = int(kwargs['n']) if 'n' in kwargs else 1
    for i in range(n_images):
        # random seed
        seed = int(kwargs['seed']) if 'seed' in kwargs else random.randrange(0, 2**32)
        kwargs['generator'] = torch.Generator("cuda").manual_seed(seed)
        
        message_text = f"dreaming of `{prompt}{'' if negative_prompt == '' else f'[{negative_prompt}]'}` with seed {seed} ({i+1}/{n_images})"
        msg = await ctx.send(message_text)
        start_time = time.time()
        
        # actually generate
        contextual_callback = functools.partial(callback, msg=msg, pipe=pipe, message=message_text)
        with torch.autocast("cuda"), torch.inference_mode():
            image = pipe(prompt,
            # callback=contextual_callback,
            # callback_steps=10,
            **kwargs).images[0]
        
        # save and send result
        filepath = save_img(image, 'outputs')
        elapsed_time = int(time.time() - start_time)
        message_text = f"\"{prompt}{'' if negative_prompt == '' else f'[{negative_prompt}]'}\" by {ctx.author.mention} in {elapsed_time}s with seed {seed} ({i+1}/{n_images})"

        # nest_asyncio.apply()
        # asyncio.get_event_loop().run_until_complete(msg.edit(f'{message_text}', file=discord.File(filepath)))
        await msg.edit(message_text, file=discord.File(filepath))
        
        og_prompt = ' '.join(prompt)
        sanitized_authorname = re.sub(r'\W+', '', ctx.author.name)
        with open('history.txt', 'a') as f:
            f.write(f'\n{datetime.now().strftime("%m/%d/%y %H:%M:%S")}: "{og_prompt}" by {sanitized_authorname} ({i+1}/{n_images}) with seed {seed} and model {loaded_model} done in {elapsed_time}s at {filename}')
            f.close()

@bot.command()
async def edit(ctx, *prompt):
    global pipe
    prompt, kwargs = parse_prompt(prompt)
    kwargs = {
        'num_inference_steps': int(kwargs['steps']) if 'steps' in kwargs else 50,
        'guidance_scale': float(kwargs['scale']) if 'scale' in kwargs else 7.5,
        'image_guidance_scale': float(kwargs['scale']) if 'scale' in kwargs else 1.5,
        'height': int(kwargs['height']) if 'height' in kwargs else 512,
        'width': int(kwargs['width']) if 'width' in kwargs else 512,
        'negative_prompt': kwargs['negative_prompt'] if 'negative_prompt' in kwargs else ''
    }
    negative_prompt = kwargs['negative_prompt']

    # initialize model
    if not isinstance(pipe, diffusers.StableDiffusionInstructPix2PixPipeline):
        scheduler = diffusers.DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
        pipe = diffusers.StableDiffusionInstructPix2PixPipeline.from_pretrained("timbrooks/instruct-pix2pix", revision="fp16", torch_dtype=torch.float16, safety_checker=None, scheduler=scheduler).to("cuda")

    # get init image
    # todo: parse url in prompt
    if len(ctx.message.attachments) == 0:
        await ctx.send(f'please upload an image to edit')
        return
    image_to_edit = PIL_from_url(ctx.message.attachments[0].url)
    
    # do the edit
    with torch.autocast("cuda"), torch.inference_mode():
        edited_image = pipe(prompt, image=image_to_edit, num_inference_steps=kwargs['steps'], image_guidance_scale=1.5, guidance_scale=7).images[0]
    
    # save and send result
    filename = save_img(image, 'outputs')
    elapsed_time = int(time.time() - start_time)
    await ctx.send(f"edited image: \"{prompt}{'' if negative_prompt == '' else f'[{negative_prompt}]'}\" by {ctx.author.mention} in {elapsed_time}s with seed {seed}", file=discord.File(filename))
        
bot.run(discord_token)
