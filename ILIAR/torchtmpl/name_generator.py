# name_generator.py
import random

# List of adjectives and famous scientists' last names
adjectives = [
    "admiring", "adoring", "affectionate", "agitated", "amazing", "angry",
    "awesome", "blissful", "bold", "boring", "brave", "clever", "cool", 
    "compassionate", "competent", "confident", "cranky", "dazzling", "determined",
    "distracted", "dreamy", "eager", "ecstatic", "elastic", "elated", "elegant",
    "eloquent", "epic", "fervent", "festive", "flamboyant", "focused", "friendly",
    "frosty", "gallant", "gifted", "goofy", "gracious", "happy", "hardcore",
    "hopeful", "hungry", "infallible", "inspiring", "jolly", "jovial", "keen",
    "kind", "laughing", "loving", "lucid", "mystifying", "modest", "musing",
    "naughty", "nervous", "nice", "nifty", "nostalgic", "objective", "optimistic",
    "peaceful", "pedantic", "pensive", "practical", "priceless", "quirky",
    "quizzical", "recursing", "relaxed", "reverent", "romantic", "sad",
    "serene", "sharp", "silly", "sleepy", "stoic", "stupefied", "suspicious",
    "sweet", "tender", "thirsty", "trusting", "unruffled", "upbeat", "vibrant",
    "vigilant", "vigorous", "wizardly", "wonderful", "xenodochial", "youthful",
    "zealous", "zen"
]

scientists = [
    "archimedes", "bardeen", "bohr", "curie", "darwin", "einstein", "euclid",
    "fermat", "fermi", "fourier", "galileo", "gauss", "hawking", "hypatia",
    "kepler", "lovelace", "maxwell", "newton", "noether", "pascal",
    "pasteur", "poincare", "tesla", "turing", "volta", "wright", "bernoulli",
    "coulomb", "faraday", "laplace", "lagrange", "planck", "riemann"
]

def generate_cool_name():

    return f"{random.choice(adjectives)}_{random.choice(scientists)}"
