import pandas as pd
import numpy as np
import json
import time
import random

import os
os.environ["OPENAI_API_KEY"]=APIKEY

import sys
from tqdm import tqdm
from pydantic import BaseModel
from openai import OpenAI

client = OpenAI()

with open("data/netflix_argument_dict.json", 'r') as file:
    samples = json.load(file)

class RecommendationResult(BaseModel):
    next_movie: str
    next_movie_rating: int

system_text="""
Based on the movie viewing history that the user enters and the ratings for each movie, select the next movie and rating that the user will choose from among the options.
"""


with open("data/netflix_argument_baseline.jsonl", "a") as file:
    pt=0
    ct=0
    for user in tqdm(list(samples.keys())):
        user_history=[]
        for j in range(len(samples[user]["history"])):
            movie=samples[user]["history"][j]
            movie_rate=samples[user]["history_ratings"][j]
            user_history.append(f"{movie}'s rate is {movie_rate}")
        options=samples[user]["new_options"]
        
        completion = client.beta.chat.completions.parse(
                        model="gpt-4o-mini",
                        messages=[
                        {"role": "system", "content": system_text},
                        {"role": "user", "content": f'user history: {user_history} next options: {options}'},
                        ],
                        response_format=RecommendationResult,
                        )
        try:
            next_movie=completion.choices[0].message.parsed.next_movie
            next_movie_rating=completion.choices[0].message.parsed.next_movie_rating
            pt+=completion.usage.prompt_tokens
            ct+=completion.usage.completion_tokens

            data = {
                            "user": user,
                            "history": samples[user]["history"],
                            "history_ratings": samples[user]["history_ratings"],
                            "options": samples[user]["new_options"],
                            "next_movie": next_movie,
                            "next_movie_rating": next_movie_rating
                        }
            print("pt: ", pt, "/ ct: ", ct)
            file.write(json.dumps(data) + "\n")
            file.flush()
        except:
            pass
