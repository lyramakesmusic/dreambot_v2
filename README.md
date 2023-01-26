# simple-dreambot

A discord interface for stable diffusion, gpt3, and other AI tools.

---

**Available commands:**

`-dream`: text-to-image via stable diffusion 1.5. usage: `-dream [prompt] [[negative_prompt]] seed=1 steps=50 scale=7.5 n=5 lora='model_name'`. Anything inside `[]` is treated as the negative prompt.

`-gpt`: prompt GPT3 (text-davinci-003). usage: `-gpt [prompt] tokens=1000 temp=0.8`

`-dalle`: text-to-image via Dall-E 2. usage: `-dalle [prompt] n=1`. May not be available all the time due to lack of generation tokens.

`-img2img`: image-to-image style transfer via stable diffusion 1.5. usage: `-img2img [prompt] seed=1 steps=50 scale=7.5 strength=0.6 n=5`. Requires an image attachment if no URL is present in prompt.

`-edit`: text+image instruct editing via [instruct-pix2pix](https://github.com/timothybrooks/instruct-pix2pix). usage: `-edit [prompt] seed=1 steps=20 txt_scale=7 img_scale=1.5`. Requires an image attachment if no URL is present in prompt

`-palette`: text-to-color palette via stable diffusion and colorthief. usage: `-palette [prompt] seed=1 n=5`

`-variations`: use stable diffusion 1.5 to create image variations. usage: `-variations [optional_url]`. Requires an image attachment if no URL is present.

`-whisper`: use openai's [whisper](https://github.com/openai/whisper) base model to transcribe audio files. usage: `-whisper [optional_url]`. Requires a .wav, .mp3, or .flac attachment if no URL is present.

`-upscale`: upscale your images with realesrgan. usage: `-upscale [optional_url]`. Requires an image attachment if no URL is present.

`-ytmp3`: use youtube-dl to extract the highest available quality mp3 from a video link. usage: `-ytmp3 [link_to_video]`. [list of supported sites](http://ytdl-org.github.io/youtube-dl/supportedsites.html)

`-dd`: use dance diffusion to generate audio samples from a given model. usage: `-dd [model_name] steps=100 n=5`. if model_name is empty, it will list the available models.

--- 
**Experimental commands:**

These are either untested, may not run on my 8gb of vram, or the idea is flawed in some way. Not guaranteed to work!

`-finetune`: finetune stable diffusion on your images with [LoRA](https://github.com/cloneofsimo/lora). usage: `-finetune [training_prompt] steps=12000 lr=1e-4 name='model_name'` **!! WIP !!**

`-dd_finetune`: finetune dance diffusion on your samples. usage: `-dd_finetune [optional_URLs] name='model_name' steps=1000`. Requires .wav or .flac attachments if no URL list is provided. **!! WIP !!**

`-chatgpt`: use an unofficial API to talk to chatGPT. usage: `-chatgpt [prompt] jailbreak=false`. Each user is assigned a single thread. If prompt is `refresh`, it loads a new thread for that user. if `jailbreak=true`, you'll be given a new thread preloaded with [DAN](https://www.reddit.com/r/ChatGPT/comments/zlcyr9/dan_is_my_new_friend/). **!! WIP !!**

`-ffmpeg`: command ffmpeg using natural language (via gpt3). usage: `-ffmpeg [instruction]`. ex: `-ffmpeg make this picture into a 2 second video` Requires a media file attachment if no URL is present. **!! WIP !!**

---

**Installation:**

Inside your environment, install pytorch with latest CUDA. You'll need 8gb+ VRAM for all functions to work. Then, run these:

`pip install pycord python-dotenv requests numpy`

`pip install --upgrade diffusers accelerate transformers`

`pip install openai-whisper openai`

`pip install colorthief youtube_dl`
