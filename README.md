# DreamBot

DreamBot is a simple Discord bot that uses Stable Diffusion to generate images based on a given prompt. It also supports chat functionalities using GPT-4.

## Features
- Generate images based on user prompts
- Chat with GPT-4

## Requirements
- [diffusers](https://github.com/huggingface/diffusers) - This will force you to install most of the other requirements.
- [discord.py](https://pypi.org/project/discord.py/)
- [pygpt4](https://github.com/lyramakesmusic/pygpt4)

## Installation

1. Clone this repository.
2. Install the required packages with `pip install`.
3. Create a `.env` file and add your `DISCORD_TOKEN` and `OPENAI_TOKEN`.

## Usage

Run the bot with `python bot.py`.

### Commands

- `-chat <prompt>`: Chat with the bot using GPT-4.
- `-stream <prompt>`: Stream responses from GPT-4.
- `-sys <sys_prompt>`: Set a system prompt for GPT-4.
- `-dream <prompt> [args]`: Generate an image based on the given prompt.

#### Dream command arguments

- `--n`: Number of images to create (default: 1)
- `--scale`: Guidance scale (default: 7.5)
- `--steps`: Number of generation steps (default: 50)
- `--ar`: Aspect ratio (default: "1:1")
- `--seed`: Seed for random number generator (default: 0)
- `--no`: Negative prompt (default: ""). Note: everything after --no is parsed as the negative prompt, so make sure it's the last argument in the list.

## Examples

```
-dream A beautiful sunset over the mountains, hd photography --ar 16:9 --steps 100 --scale 10 --no washed out, low contrast

-chat How do I boil an egg?
-stream List 10 ingredients for a stir fry:
```
