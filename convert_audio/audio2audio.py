#!/bin/python
from vallex import Vallex, VallExProcessor
from transformers import AutoProcessor, AutoModelForCausalLM
import torch

import soundfile as sf


import argparse, os


def get_lang(input_):
    if (len(input_)) == 2:
        return input_
    LANGUAGE_CODE_MAP = {
        "english": "en",
        "spanish": "es",
        "french": "fr",
        "german": "de",
        "italian": "it",
        "portuguese": "pt",
        "dutch": "nl",
        "russian": "ru",
        "chinese": "zh",
        "japanese": "ja",
        "korean": "ko",
        "arabic": "ar",
        "hindi": "hi",
        "turkish": "tr",
        "polish": "pl",
        "swedish": "sv",
        "danish": "da",
        "norwegian": "no",
        "finnish": "fi",
        "greek": "el",
        "czech": "cs",
        "romanian": "ro",
        "hungarian": "hu",
        "ukrainian": "uk",
        "thai": "th",
        "vietnamese": "vi",
    }
    normalized = input_.lower().strip()
    if normalized in LANGUAGE_CODE_MAP:
        return LANGUAGE_CODE_MAP[normalized]
    raise ValueError(f"Unsupported language: '{language}'")


def ConvertAudioToAudio(input_,output_,lang):
    # TODO
    
def TranslateText(input_,output_,lang):
    # TODO

def ConvertTextToAudio(input_,output_,lang):
    # TODO

def main():
    parser = argparse.ArgumentParser(description="A script for XXX.")
    
    # The 'type=int' ensures that the input '1' is treated as an integer
    parser.add_argument('--input', type=str, required=True,help='The input file to translate')
    parser.add_argument('--output', type=str, required=False,help='Output file with translation')
    parser.add_argument('--language', type=str, required=True,default=1,help='The language to translate to')
    
    args = parser.parse_args()

    ConvertAudioToAudio(args.input,args.output)
    
    print(f"The script was executed.")

if __name__ == "__main__":
    main()




