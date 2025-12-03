"""
Word list module for Wordle RL Agent.
Provides common, meaningful 5-letter English words for training and gameplay.
"""

# Curated list of common 5-letter English words
# These are meaningful words commonly used in daily English
COMMON_WORDS = [
    # A
    "about", "above", "abuse", "actor", "acute", "admit", "adopt", "adult", "after",
    "again", "agent", "agree", "ahead", "alarm", "album", "alert", "alike", "alive",
    "allow", "alone", "along", "alter", "among", "anger", "angle", "angry", "apart",
    "apple", "apply", "arena", "argue", "arise", "armor", "army", "arrow", "aside",
    "asset", "avoid", "award", "aware", "awful",
    # B
    "baby", "back", "bacon", "badge", "badly", "baker", "basic", "basis", "beach",
    "beard", "beast", "began", "begin", "being", "belly", "below", "bench", "berry",
    "birth", "black", "blade", "blame", "blank", "blast", "bleed", "blend", "bless",
    "blind", "block", "blood", "bloom", "blown", "board", "boast", "bone", "bonus",
    "booth", "born", "bound", "brain", "brand", "brass", "brave", "bread", "break",
    "breed", "brick", "bride", "brief", "bring", "broad", "broke", "brook", "brown",
    "brush", "build", "built", "bunch", "burst", "buyer",
    # C
    "cabin", "cable", "camel", "candy", "canon", "cargo", "carry", "carve", "catch",
    "cause", "cease", "chain", "chair", "chalk", "champ", "chance", "chaos", "charm",
    "chart", "chase", "cheap", "cheat", "check", "cheek", "cheer", "chess", "chest",
    "child", "chill", "china", "chip", "chord", "chose", "chunk", "claim", "class",
    "clean", "clear", "clerk", "click", "cliff", "climb", "cling", "clock", "close",
    "cloth", "cloud", "clown", "club", "coach", "coast", "colon", "color", "comet",
    "comic", "comma", "coral", "couch", "cough", "could", "count", "court", "cover",
    "crack", "craft", "crane", "crash", "crawl", "crazy", "cream", "creek", "creep",
    "crime", "crisp", "cross", "crowd", "crown", "crude", "cruel", "crush", "curve",
    "cycle",
    # D
    "daily", "dairy", "dance", "dated", "dealt", "death", "debut", "decay", "decor",
    "delay", "delta", "dense", "depot", "depth", "derby", "desk", "devil", "diary",
    "digit", "dirty", "disco", "ditch", "diver", "dizzy", "doing", "donor", "donut",
    "doubt", "dough", "down", "dozen", "draft", "drain", "drake", "drama", "drank",
    "drawn", "dread", "dream", "dress", "dried", "drift", "drill", "drink", "drive",
    "droit", "drown", "drunk", "dryer", "dusty", "dying",
    # E
    "eager", "eagle", "early", "earth", "eaten", "eight", "elbow", "elder", "elect",
    "elite", "email", "ember", "empty", "ended", "enemy", "enjoy", "enter", "entry",
    "equal", "equip", "error", "essay", "ethic", "evade", "event", "every", "exact",
    "exams", "excel", "exist", "extra",
    # F
    "faced", "facet", "faith", "false", "fancy", "fatal", "fatty", "fault", "favor",
    "feast", "fence", "ferry", "fetal", "fetch", "fever", "fiber", "field", "fiery",
    "fifth", "fifty", "fight", "final", "first", "fixed", "flame", "flash", "flask",
    "flesh", "float", "flock", "flood", "floor", "flour", "fluid", "flush", "flute",
    "focal", "focus", "foggy", "force", "forge", "forth", "forty", "forum", "fossil",
    "found", "fox", "frame", "frank", "fraud", "freak", "fresh", "fried", "front",
    "frost", "froze", "fruit", "fully", "fungi", "funny", "fuzzy",
    # G
    "gauge", "genre", "ghost", "giant", "given", "glass", "gleam", "globe", "gloom",
    "glory", "gloss", "glove", "going", "golden", "goods", "goose", "grace", "grade",
    "grain", "grand", "grant", "grape", "graph", "grasp", "grass", "grave", "great",
    "greed", "greek", "green", "greet", "grief", "grill", "grind", "groan", "groom",
    "gross", "group", "grove", "grown", "guard", "guess", "guest", "guide", "guild",
    "guilt", "guise", "guitar",
    # H
    "habit", "hairy", "handy", "happy", "harsh", "haste", "hasty", "hatch", "haunt",
    "haven", "heart", "heavy", "hedge", "heels", "hello", "hence", "herbs", "hinge",
    "hippo", "hobby", "holly", "honey", "honor", "hoped", "horse", "host", "hotel",
    "hound", "house", "human", "humid", "humor", "hurry", "hyper",
    # I
    "ideal", "image", "imply", "inbox", "index", "indie", "infer", "inner", "input",
    "intro", "irony", "issue", "ivory",
    # J
    "jeans", "jelly", "jewel", "joint", "joker", "jolly", "judge", "juice", "juicy",
    "jumbo", "jumpy", "junior",
    # K
    "karma", "kayak", "kebab", "khaki", "kidney", "kings", "kiosk", "kitty", "knack",
    "knead", "kneel", "knife", "knock", "knots", "known",
    # L
    "label", "labor", "laces", "laden", "ladle", "lager", "lakes", "lamps", "lands",
    "lanes", "lapel", "lapse", "large", "laser", "latch", "later", "latex", "laugh",
    "layer", "leads", "leafy", "learn", "lease", "least", "leave", "ledge", "legal",
    "lemon", "level", "lever", "light", "liked", "limit", "linen", "liner", "links",
    "lions", "lists", "liter", "lived", "liver", "lives", "llama", "loads", "loans",
    "lobby", "local", "lodge", "lofty", "logic", "lonely", "loose", "lorry", "loser",
    "lotus", "loud", "louse", "loved", "lover", "lower", "loyal", "lucid", "lucky",
    "lunar", "lunch", "lunge", "lyric",
    # M
    "macro", "madam", "magic", "major", "maker", "mango", "manor", "maple", "march",
    "marry", "marsh", "masks", "mason", "match", "maybe", "mayor", "means", "meant",
    "medal", "media", "melon", "mercy", "merge", "merit", "merry", "messy", "metal",
    "meter", "metro", "micro", "midst", "might", "mimic", "mince", "minds", "miner",
    "minor", "minus", "mirth", "misty", "mixed", "mixer", "model", "modem", "moist",
    "money", "month", "moody", "moral", "motor", "motto", "mound", "mount", "mourn",
    "mouse", "mouth", "moved", "mover", "movie", "muddy", "mural", "music", "mutual",
    # N
    "naive", "naked", "named", "nanny", "nasal", "nasty", "naval", "navel", "needs",
    "nerve", "never", "newer", "newly", "night", "ninth", "noble", "noise", "noisy",
    "north", "notch", "noted", "notes", "novel", "nurse", "nutty", "nylon",
    # O
    "oasis", "occur", "ocean", "octave", "oddly", "offer", "often", "olive", "omega",
    "onion", "onset", "opera", "optic", "orbit", "order", "organ", "other", "ought",
    "ounce", "outer", "outdo", "owned", "owner", "oxide", "ozone",
    # P
    "paced", "paint", "pairs", "panda", "panel", "panic", "paper", "party", "pasta",
    "paste", "patch", "pause", "peace", "peach", "pearl", "pedal", "penny", "perch",
    "peril", "perks", "petal", "petty", "phase", "phone", "photo", "piano", "piece",
    "pilot", "pinch", "pitch", "pixel", "pizza", "place", "plaid", "plain", "plane",
    "plant", "plate", "plaza", "plead", "pleat", "pledge", "pluck", "plumb", "plume",
    "plump", "plunge", "plus", "poach", "poems", "poets", "point", "poise", "poker",
    "polar", "polls", "polyp", "pond", "pooch", "pools", "porch", "pores", "ports",
    "posed", "poser", "posts", "pouch", "pound", "power", "prank", "prawn", "press",
    "price", "pride", "prime", "print", "prior", "prism", "prize", "probe", "prone",
    "proof", "prose", "proud", "prove", "proxy", "prune", "pubic", "pulse", "punch",
    "pupil", "puppy", "purse", "push",
    # Q
    "quack", "qualm", "quart", "queen", "query", "quest", "queue", "quick", "quiet",
    "quilt", "quirk", "quite", "quota", "quote",
    # R
    "rabbi", "racer", "radar", "radio", "rainy", "raise", "rally", "ranch", "range",
    "rapid", "ratio", "razor", "reach", "react", "ready", "realm", "rebel", "refer",
    "reign", "relax", "relay", "relic", "remix", "renew", "repay", "reply", "rerun",
    "reset", "resin", "retro", "rhino", "rhyme", "rider", "ridge", "rifle", "right",
    "rigid", "rigor", "rinse", "risen", "risky", "ritzy", "rival", "river", "roach",
    "roads", "roast", "robot", "rocky", "rodeo", "rogue", "roman", "roots", "roses",
    "rotor", "rouge", "rough", "round", "route", "rover", "royal", "rugby", "ruins",
    "ruler", "rumor", "rural", "rusty",
    # S
    "sadly", "safer", "saint", "salad", "sales", "salon", "salsa", "salty", "sandy",
    "satin", "sauce", "sauna", "saved", "saver", "savor", "scale", "scalp", "scant",
    "scare", "scarf", "scary", "scene", "scent", "scope", "score", "scout", "scrap",
    "screw", "scrub", "seams", "seats", "seeds", "seize", "sense", "serum", "serve",
    "setup", "seven", "sever", "shade", "shady", "shaft", "shake", "shaky", "shall",
    "shame", "shape", "share", "shark", "sharp", "shave", "sheep", "sheer", "sheet",
    "shelf", "shell", "shift", "shine", "shiny", "shire", "shirt", "shock", "shoes",
    "shook", "shoot", "shore", "short", "shout", "shown", "shows", "shrug", "sight",
    "sigma", "signs", "silly", "since", "siren", "sixth", "sixty", "sized", "skate",
    "skill", "skimp", "skirt", "skull", "slain", "slang", "slash", "slate", "slave",
    "sleek", "sleep", "sleet", "slice", "slide", "slime", "slimy", "slope", "sloth",
    "slump", "small", "smart", "smash", "smell", "smile", "smoke", "smoky", "snack",
    "snail", "snake", "snare", "sneak", "sniff", "snore", "snout", "snow", "soapy",
    "sober", "social", "softy", "soggy", "solar", "solid", "solve", "sonar", "songs",
    "sonic", "sorry", "sorts", "souls", "sound", "south", "space", "spade", "spare",
    "spark", "spawn", "speak", "spear", "speed", "spell", "spend", "spent", "spice",
    "spicy", "spill", "spine", "spiny", "spite", "split", "spoke", "spoon", "sport",
    "spots", "spray", "spree", "squad", "stack", "staff", "stage", "stain", "stair",
    "stake", "stale", "stalk", "stamp", "stand", "stare", "stark", "stars", "start",
    "state", "stays", "steak", "steal", "steam", "steel", "steep", "steer", "stems",
    "steps", "stern", "stick", "stiff", "still", "sting", "stink", "stock", "stomp",
    "stone", "stony", "stood", "stool", "stoop", "stops", "store", "stork", "storm",
    "story", "stout", "stove", "strap", "straw", "stray", "strip", "stuck", "study",
    "stuff", "stump", "stung", "stunk", "style", "suave", "sugar", "suite", "sunny",
    "super", "surge", "sushi", "swamp", "swarm", "swear", "sweat", "sweep", "sweet",
    "swell", "swept", "swift", "swine", "swing", "swirl", "sword", "sworn", "swung",
    # T
    "tabby", "table", "tacky", "taint", "taken", "taker", "tally", "talon", "tango",
    "tangy", "tanks", "taper", "tardy", "taste", "tasty", "taunt", "taxes", "teach",
    "teams", "tears", "tease", "teddy", "teens", "teeth", "tempo", "tends", "tenor",
    "tense", "tenth", "terms", "terra", "tests", "thank", "theft", "theme", "there",
    "these", "thick", "thief", "thigh", "thing", "think", "third", "thirst", "thorn",
    "those", "three", "threw", "throw", "thumb", "thump", "tiger", "tight", "tiles",
    "tilts", "timer", "times", "timid", "tipsy", "tired", "titan", "title", "toast",
    "today", "token", "toned", "toner", "tongs", "tonic", "tools", "tooth", "topic",
    "torch", "total", "touch", "tough", "tours", "towel", "tower", "towns", "toxic",
    "trace", "track", "tract", "trade", "trail", "train", "trait", "tramp", "trash",
    "trawl", "tread", "treat", "trees", "trend", "trial", "tribe", "trick", "tried",
    "trips", "trite", "troll", "troop", "trout", "truce", "truck", "truly", "trump",
    "trunk", "trust", "truth", "tubby", "tulip", "tumor", "tuned", "tuner", "tunic",
    "turbo", "tutor", "twang", "tweak", "tweed", "tweet", "twice", "twigs", "twine",
    "twirl", "twist", "tying", "typed", "typos",
    # U
    "udder", "ulcer", "ultra", "umbra", "uncle", "under", "undid", "undue", "unfed",
    "unfit", "union", "unite", "units", "unity", "unlit", "unmet", "until", "upper",
    "upset", "urban", "urged", "urine", "usage", "usher", "using", "usual", "utter",
    # V
    "vague", "valid", "valor", "value", "valve", "vapid", "vapor", "vault", "vegan",
    "veins", "velvet", "venom", "venue", "verge", "verse", "vibes", "video", "views",
    "vigil", "vigor", "villa", "vinyl", "viola", "viper", "viral", "virus", "visit",
    "visor", "vista", "vital", "vivid", "vocal", "vodka", "vogue", "voice", "voila",
    "vomit", "voter", "vouch", "vowel", "vying",
    # W
    "wacky", "wader", "wager", "wages", "wagon", "waist", "waive", "waken", "walks",
    "walls", "waltz", "wanna", "wants", "warps", "warts", "waste", "watch", "water",
    "waver", "waves", "waxen", "waxes", "weary", "weave", "wedge", "weeds", "weedy",
    "weeks", "weigh", "weird", "wells", "welsh", "whack", "whale", "wharf", "wheat",
    "wheel", "where", "which", "while", "whine", "whiny", "whips", "whirl", "whisk",
    "white", "whole", "whose", "widen", "wider", "widow", "width", "wield", "wills",
    "wimpy", "wince", "winch", "winds", "windy", "wines", "wings", "wiped", "wiper",
    "wired", "wires", "witch", "witty", "wives", "woken", "woman", "women", "woody",
    "woozy", "words", "wordy", "works", "world", "worms", "worry", "worse", "worst",
    "worth", "would", "wound", "woven", "wrack", "wraps", "wrath", "wreak", "wreck",
    "wrest", "wring", "wrist", "write", "wrong", "wrote", "wrung",
    # X
    "xerox",
    # Y
    "yacht", "yearn", "yeast", "yield", "young", "yours", "youth", "yummy",
    # Z
    "zappy", "zebra", "zesty", "zilch", "zingy", "zippy", "zombi", "zonal", "zones",
]


def load_word_list():
    """
    Load a curated list of common, meaningful 5-letter English words.

    Returns:
        list: Sorted list of valid 5-letter words
    """
    # Filter to ensure all words are exactly 5 letters and alphabetic
    valid_words = [
        word.lower() for word in COMMON_WORDS
        if len(word) == 5 and word.isalpha()
    ]

    # Remove duplicates and sort
    valid_words = sorted(list(set(valid_words)))

    print(f"Loaded {len(valid_words)} common 5-letter words")
    return valid_words


def load_word_list_extended():
    """
    Load an extended word list combining curated words with NLTK corpus.
    Falls back to curated list if NLTK is unavailable.

    Returns:
        list: Sorted list of valid 5-letter words
    """
    # Start with curated common words
    word_set = set(word.lower() for word in COMMON_WORDS if len(word) == 5 and word.isalpha())

    try:
        import nltk
        from nltk.corpus import words as nltk_words

        # Download required NLTK data
        nltk.download("words", quiet=True)

        # Add common words from NLTK (filter for likely common words)
        # Only add words that are all lowercase in the corpus (usually more common)
        for word in nltk_words.words():
            if len(word) == 5 and word.isalpha() and word.islower():
                word_set.add(word.lower())

        print(f"Loaded {len(word_set)} words (curated + NLTK)")

    except Exception as e:
        print(f"NLTK unavailable, using curated list only: {e}")
        print(f"Loaded {len(word_set)} curated words")

    return sorted(list(word_set))


if __name__ == "__main__":
    # Test the word list
    words = load_word_list()
    print(f"\nFirst 20 words: {words[:20]}")
    print(f"Last 20 words: {words[-20:]}")
