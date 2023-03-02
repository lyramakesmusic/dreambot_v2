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

from bs4 import BeautifulSoup
from bs4.element import Comment

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

# init pycord
intents = discord.Intents.all()
intents.message_content = True
bot = commands.Bot(command_prefix="-", intents=intents)

# load auth tokens
load_dotenv()
discord_token = os.getenv('DISCORD_TOKEN')
openai.api_key = os.getenv('OPENAI_TOKEN')
elevenlabs_token = os.getenv('ELEVENLABS_TOKEN')

# global variables
pipe = None
nest_asyncio.apply()

chatgpt_user_histories = {}
user_system_msgs = {}

random.seed(time.time())
print('started')

@bot.command()
async def dream(ctx, *prompt):
    global pipe
    prompt, kwargs = parse_prompt(prompt)
    n_images = int(kwargs['n']) if 'n' in kwargs else 1
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
            "./stable-diffusion-v1-5",
            # "./lyra_LoRA-v1-5",
            revision="fp16",
            torch_dtype=torch.float16,
            safety_checker=None,
            scheduler=scheduler).to("cuda")

        # load finetunes
        if 'lora' in kwargs:
            pipe.unet.load_attn_procs(f"lora/{kwargs['lora']}")
    
    # generate n images
    for i in range(n_images):
        # random seed
        seed = int(kwargs['seed']) if 'seed' in kwargs else random.randrange(0, 2**32)
        kwargs['generator'] = torch.Generator("cuda").manual_seed(seed)
        
        message_text = f"dreaming of `{prompt}{'' if negative_prompt == '' else f'[{negative_prompt}]'}` with seed {seed} ({i+1}/{n_images})"
        msg = await ctx.send(message_text)
        start_time = time.time()
        
        # actually generate
        contextual_callback = functools.partial(callback, msg=msg, pipe=pipe, message=message_text, total_steps=kwargs['num_inference_steps'])
        with torch.autocast("cuda"), torch.inference_mode():
            image = pipe(prompt,
            callback=contextual_callback,
            callback_steps=10,
            **kwargs).images[0]
        
        # save and send result
        filepath = save_img(image, 'outputs')
        elapsed_time = int(time.time() - start_time)
        message_text = f"\"{prompt}{'' if negative_prompt == '' else f'[{negative_prompt}]'}\" by {ctx.author.mention} in {elapsed_time}s with seed {seed} ({i+1}/{n_images})"

        # nest_asyncio.apply()
        # asyncio.get_event_loop().run_until_complete(msg.edit(f'{message_text}', file=discord.File(filepath)))
        loop = asyncio.get_event_loop()
        loop.run_until_complete(msg.edit(message_text, file=discord.File(filepath)))
        
        og_prompt = ' '.join(prompt)
        sanitized_authorname = re.sub(r'\W+', '', ctx.author.name)
        with open('history.txt', 'a') as f:
            f.write(f'\n{datetime.now().strftime("%m/%d/%y %H:%M:%S")}: "{og_prompt}" by {sanitized_authorname} ({i+1}/{n_images}) with seed {seed} done in {elapsed_time}s at {filepath}')
            f.close()

@bot.command()
async def gpt(ctx, *prompt):
    prompt = " ".join(prompt)
    response = openai.Completion.create(
        model='text-davinci-003',
        prompt=prompt,
        temperature=temp,
        max_tokens=tokens
    )
    await ctx.send(response.choices[0].text)

@bot.command()
async def sys(ctx, *prompt):
    global user_system_msgs
    prompt = " ".join(prompt)
    sanitized_authorname = re.sub(r'\W+', '', ctx.author.name)
    user_system_msgs[sanitized_authorname] = prompt
    chatgpt_user_histories.pop(sanitized_authorname, None)
    await ctx.send(f'updated {sanitized_authorname}\'s default chatgpt command to `{prompt}`')
    print(f'The user system prompts are ```{user_system_msgs}```')

@bot.command()
async def search(ctx, *prompt):
    prompt, kwargs = parse_prompt(prompt)

    # google the term
    url = f"https://www.google.com/search?q={prompt}&num=1"
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"})
    soup = BeautifulSoup(response.text, "html.parser")

    def tag_visible(element):
        if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
            return False
        if isinstance(element, Comment):
            return False
        return True

    texts = soup.findAll(string=True)
    visible_texts = filter(tag_visible, texts)  
    result = u" ".join(t.strip() for t in visible_texts)[:].split('Search Results')[1].split('More results Try again')[0]

    print(result)
    
    system_msg = f'You are a search-enabled AI, with access to realtime information. Knowledge cutoff: 3/2/23. Make sure to give all relevant information from the search results. Be specific, don\'t just give an overview.'

    prompt = f'Here are the current google search results for `{prompt}`:\n{result}. Answer the prompt with the information given. Make sure to include all numbers and details. Don\'t include related searches.'
    
    try:
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", 
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
        )
    except Exception as e:
        await ctx.send(str(e))
        return

    await ctx.send(completion.choices[0]['message']['content'].strip()[:2000])

@bot.command()
async def chat(ctx, *prompt):
    global chatgpt_user_histories
    prompt, kwargs = parse_prompt(prompt)
    temp = float(kwargs['temp']) if 'temp' in kwargs else 1.0

    # handle user history
    sanitized_authorname = re.sub(r'\W+', '', ctx.author.name)
    if prompt == "clear":
        chatgpt_user_histories.pop(sanitized_authorname, None)
        print(f'{sanitized_authorname} cleared history')
        await ctx.send("Chat history cleared.")
        return

    # system_msg = "Knowledge cutoff: 10000 bc. You are a caveman. You can only use short words and extremely simple sentences. Use caveman grammar."
    system_msg = "You are an expert assistant with a perfect track record of answering questions and providing information clearly and concisely."
    if sanitized_authorname in user_system_msgs:
        system_msg = user_system_msgs[sanitized_authorname]

    print(f'{system_msg}')

    print(f'{sanitized_authorname} -> chatgpt: {prompt}')
    if sanitized_authorname not in chatgpt_user_histories:
        chatgpt_user_histories[sanitized_authorname] = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt}
        ]
    else:
        chatgpt_user_histories[sanitized_authorname].append({"role": "user", "content": prompt})
    user_history = chatgpt_user_histories[sanitized_authorname]

    # call API
    try:
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", 
            messages=user_history,
            temperature=temp,
        )
    except Exception as e:
        await ctx.send(str(e))
        return
    response = completion.choices[0]['message']['content'].strip()[:2000]
    print(f'chatgpt -> {sanitized_authorname}: {response}')
    user_history.append({"role": "assistant", "content": response})
    chatgpt_user_histories[sanitized_authorname] = user_history
    try:
        await ctx.send(response)
    except Exception as e:
        await ctx.send(str(e))

# @bot.command()
# async def list_voices(ctx):
#     headers = {
#         'Accept': 'application/json',
#         'xi-api-key': elevenlabs_token
#     }
#     response = requests.get('https://api.elevenlabs.io/v1/voices', headers=headers)
#     print(response.json())
#     await ctx.send(f'```json {response.json()}```')

bot.run(discord_token)
