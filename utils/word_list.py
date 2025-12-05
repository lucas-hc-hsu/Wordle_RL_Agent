"""
Word list module for Wordle RL Agent.
Contains the official Wordle answer words (2,309 words) sourced from the game.
Source: https://gist.github.com/slushman/34e60d6bc479ac8fc698df8c226e4264
"""

# Official Wordle answer words - all 2,309 possible solutions
WORDLE_WORDS = [
    # A
    "aback", "abase", "abate", "abbey", "abbot", "abhor", "abide", "abled", "abode", "abort",
    "about", "above", "abuse", "abyss", "acorn", "acrid", "actor", "acute", "adage", "adapt",
    "adept", "admin", "admit", "adobe", "adopt", "adore", "adorn", "adult", "affix", "afire",
    "afoot", "afoul", "after", "again", "agape", "agate", "agent", "agile", "aging", "aglow",
    "agony", "agree", "ahead", "aider", "aisle", "alarm", "album", "alert", "algae", "alibi",
    "alien", "align", "alike", "alive", "allay", "alley", "allot", "allow", "alloy", "aloft",
    "alone", "along", "aloof", "aloud", "alpha", "altar", "alter", "amass", "amaze", "amber",
    "amble", "amend", "amiss", "amity", "among", "ample", "amply", "amuse", "angel", "anger",
    "angle", "angry", "angst", "anime", "ankle", "annex", "annoy", "annul", "anode", "antic",
    "anvil", "aorta", "apart", "aphid", "aping", "apnea", "apple", "apply", "apron", "aptly",
    "arbor", "ardor", "arena", "argue", "arise", "armor", "aroma", "arose", "array", "arrow",
    "arson", "artsy", "ascot", "ashen", "aside", "askew", "assay", "asset", "atoll", "atone",
    "attic", "audio", "audit", "augur", "aunty", "avail", "avert", "avian", "avoid", "await",
    "awake", "award", "aware", "awash", "awful", "awoke", "axial", "axiom", "axion", "azure",
    # B
    "bacon", "badge", "badly", "bagel", "baggy", "baker", "baler", "balmy", "banal", "banjo",
    "barge", "baron", "basal", "basic", "basil", "basin", "basis", "baste", "batch", "bathe",
    "baton", "batty", "bawdy", "bayou", "beach", "beady", "beard", "beast", "beech", "beefy",
    "befit", "began", "begat", "beget", "begin", "begun", "being", "belch", "belie", "belle",
    "belly", "below", "bench", "beret", "berry", "berth", "beset", "betel", "bevel", "bezel",
    "bible", "bicep", "biddy", "bigot", "bilge", "billy", "binge", "bingo", "biome", "birch",
    "birth", "bison", "bitty", "black", "blade", "blame", "bland", "blank", "blare", "blast",
    "blaze", "bleak", "bleat", "bleed", "bleep", "blend", "bless", "blimp", "blind", "blink",
    "bliss", "blitz", "bloat", "block", "bloke", "blond", "blood", "bloom", "blown", "bluer",
    "bluff", "blunt", "blurb", "blurt", "blush", "board", "boast", "bobby", "boney", "bongo",
    "bonus", "booby", "boost", "booth", "booty", "booze", "boozy", "borax", "borne", "bosom",
    "bossy", "botch", "bough", "boule", "bound", "bowel", "boxer", "brace", "braid", "brain",
    "brake", "brand", "brash", "brass", "brave", "bravo", "brawl", "brawn", "bread", "break",
    "breed", "briar", "bribe", "brick", "bride", "brief", "brine", "bring", "brink", "briny",
    "brisk", "broad", "broil", "broke", "brood", "brook", "broom", "broth", "brown", "brunt",
    "brush", "brute", "buddy", "budge", "buggy", "bugle", "build", "built", "bulge", "bulky",
    "bully", "bunch", "bunny", "burly", "burnt", "burst", "bused", "bushy", "butch", "butte",
    "buxom", "buyer", "bylaw",
    # C
    "cabal", "cabby", "cabin", "cable", "cacao", "cache", "cacti", "caddy", "cadet", "cagey",
    "cairn", "camel", "cameo", "canal", "candy", "canny", "canoe", "canon", "caper", "caput",
    "carat", "cargo", "carol", "carry", "carve", "caste", "catch", "cater", "catty", "caulk",
    "cause", "cavil", "cease", "cedar", "cello", "chafe", "chaff", "chain", "chair", "chalk",
    "champ", "chant", "chaos", "chard", "charm", "chart", "chase", "chasm", "cheap", "cheat",
    "check", "cheek", "cheer", "chess", "chest", "chick", "chide", "chief", "child", "chili",
    "chill", "chime", "china", "chirp", "chock", "choir", "choke", "chord", "chore", "chose",
    "chuck", "chump", "chunk", "churn", "chute", "cider", "cigar", "cinch", "circa", "civic",
    "civil", "clack", "claim", "clamp", "clang", "clank", "clash", "clasp", "class", "clean",
    "clear", "cleat", "cleft", "clerk", "click", "cliff", "climb", "cling", "clink", "cloak",
    "clock", "clone", "close", "cloth", "cloud", "clout", "clove", "clown", "cluck", "clued",
    "clump", "clung", "coach", "coast", "cobra", "cocoa", "colon", "color", "comet", "comfy",
    "comic", "comma", "conch", "condo", "conic", "copse", "coral", "corer", "corny", "couch",
    "cough", "could", "count", "coupe", "court", "coven", "cover", "covet", "covey", "cower",
    "coyly", "crack", "craft", "cramp", "crane", "crank", "crash", "crass", "crate", "crave",
    "crawl", "craze", "crazy", "creak", "cream", "credo", "creed", "creek", "creep", "creme",
    "crepe", "crept", "cress", "crest", "crick", "cried", "crier", "crime", "crimp", "crisp",
    "croak", "crock", "crone", "crony", "crook", "cross", "croup", "crowd", "crown", "crude",
    "cruel", "crumb", "crump", "crush", "crust", "crypt", "cubic", "cumin", "curio", "curly",
    "curry", "curse", "curve", "curvy", "cutie", "cyber", "cycle", "cynic",
    # D
    "daddy", "daily", "dairy", "daisy", "dally", "dance", "dandy", "datum", "daunt", "dealt",
    "death", "debar", "debit", "debug", "debut", "decal", "decay", "decor", "decoy", "decry",
    "defer", "deign", "deity", "delay", "delta", "delve", "demon", "demur", "denim", "dense",
    "depot", "depth", "derby", "deter", "detox", "deuce", "devil", "diary", "dicey", "digit",
    "dilly", "dimly", "diner", "dingo", "dingy", "diode", "dirge", "dirty", "disco", "ditch",
    "ditto", "ditty", "diver", "dizzy", "dodge", "dodgy", "dogma", "doing", "dolly", "donor",
    "donut", "dopey", "doubt", "dough", "dowdy", "dowel", "downy", "dowry", "dozen", "draft",
    "drain", "drake", "drama", "drank", "drape", "drawl", "drawn", "dread", "dream", "dress",
    "dried", "drier", "drift", "drill", "drink", "drive", "droit", "droll", "drone", "drool",
    "droop", "dross", "drove", "drown", "druid", "drunk", "dryer", "dryly", "duchy", "dully",
    "dummy", "dumpy", "dunce", "dusky", "dusty", "dutch", "duvet", "dwarf", "dwell", "dwelt",
    "dying",
    # E
    "eager", "eagle", "early", "earth", "easel", "eaten", "eater", "ebony", "eclat", "edict",
    "edify", "eerie", "egret", "eight", "eject", "eking", "elate", "elbow", "elder", "elect",
    "elegy", "elfin", "elide", "elite", "elope", "elude", "email", "embed", "ember", "emcee",
    "empty", "enact", "endow", "enema", "enemy", "enjoy", "ennui", "ensue", "enter", "entry",
    "envoy", "epoch", "epoxy", "equal", "equip", "erase", "erect", "erode", "error", "erupt",
    "essay", "ester", "ether", "ethic", "ethos", "etude", "evade", "event", "every", "evict",
    "evoke", "exact", "exalt", "excel", "exert", "exile", "exist", "expel", "extol", "extra",
    "exult", "eying",
    # F
    "fable", "facet", "faint", "fairy", "faith", "false", "fancy", "fanny", "farce", "fatal",
    "fatty", "fault", "fauna", "favor", "feast", "fecal", "feign", "fella", "felon", "femme",
    "femur", "fence", "feral", "ferry", "fetal", "fetch", "fetid", "fetus", "fever", "fewer",
    "fiber", "ficus", "field", "fiend", "fiery", "fifth", "fifty", "fight", "filer", "filet",
    "filly", "filmy", "filth", "final", "finch", "finer", "first", "fishy", "fixer", "fizzy",
    "fjord", "flack", "flail", "flair", "flake", "flaky", "flame", "flank", "flare", "flash",
    "flask", "fleck", "fleet", "flesh", "flick", "flier", "fling", "flint", "flirt", "float",
    "flock", "flood", "floor", "flora", "floss", "flour", "flout", "flown", "fluff", "fluid",
    "fluke", "flume", "flung", "flunk", "flush", "flute", "flyer", "foamy", "focal", "focus",
    "foggy", "foist", "folio", "folly", "foray", "force", "forge", "forgo", "forte", "forth",
    "forty", "forum", "found", "foyer", "frail", "frame", "frank", "fraud", "freak", "freed",
    "freer", "fresh", "friar", "fried", "frill", "frisk", "fritz", "frock", "frond", "front",
    "frost", "froth", "frown", "froze", "fruit", "fudge", "fugue", "fully", "fungi", "funky",
    "funny", "furor", "furry", "fussy", "fuzzy",
    # G
    "gaffe", "gaily", "gamer", "gamma", "gamut", "gassy", "gaudy", "gauge", "gaunt", "gauze",
    "gavel", "gawky", "gayer", "gayly", "gazer", "gecko", "geeky", "geese", "genie", "genre",
    "ghost", "ghoul", "giant", "giddy", "gipsy", "girly", "girth", "given", "giver", "glade",
    "gland", "glare", "glass", "glaze", "gleam", "glean", "glide", "glint", "gloat", "globe",
    "gloom", "glory", "gloss", "glove", "glyph", "gnash", "gnome", "godly", "going", "golem",
    "golly", "gonad", "goner", "goody", "gooey", "goofy", "goose", "gorge", "gouge", "gourd",
    "grace", "grade", "graft", "grail", "grain", "grand", "grant", "grape", "graph", "grasp",
    "grass", "grate", "grave", "gravy", "graze", "great", "greed", "green", "greet", "grief",
    "grill", "grime", "grimy", "grind", "gripe", "groan", "groin", "groom", "grope", "gross",
    "group", "grout", "grove", "growl", "grown", "gruel", "gruff", "grunt", "guard", "guava",
    "guess", "guest", "guide", "guild", "guile", "guilt", "guise", "gulch", "gully", "gumbo",
    "gummy", "guppy", "gusto", "gusty", "gypsy",
    # H
    "habit", "hairy", "halve", "handy", "happy", "hardy", "harem", "harpy", "harry", "harsh",
    "haste", "hasty", "hatch", "hater", "haunt", "haute", "haven", "havoc", "hazel", "heady",
    "heard", "heart", "heath", "heave", "heavy", "hedge", "hefty", "heist", "helix", "hello",
    "hence", "heron", "hilly", "hinge", "hippo", "hippy", "hitch", "hoard", "hobby", "hoist",
    "holly", "homer", "honey", "honor", "horde", "horny", "horse", "hotel", "hotly", "hound",
    "house", "hovel", "hover", "howdy", "human", "humid", "humor", "humph", "humus", "hunch",
    "hunky", "hurry", "husky", "hussy", "hutch", "hydro", "hyena", "hymen", "hyper",
    # I
    "icily", "icing", "ideal", "idiom", "idiot", "idler", "idyll", "igloo", "iliac", "image",
    "imbue", "impel", "imply", "inane", "inbox", "incur", "index", "inept", "inert", "infer",
    "ingot", "inlay", "inlet", "inner", "input", "inter", "intro", "ionic", "irate", "irony",
    "islet", "issue", "itchy", "ivory",
    # J
    "jaunt", "jazzy", "jelly", "jerky", "jetty", "jewel", "jiffy", "joint", "joist", "joker",
    "jolly", "joust", "judge", "juice", "juicy", "jumbo", "jumpy", "junta", "junto", "juror",
    # K
    "kappa", "karma", "kayak", "kebab", "khaki", "kinky", "kiosk", "kitty", "knack", "knave",
    "knead", "kneed", "kneel", "knelt", "knife", "knock", "knoll", "known", "koala", "krill",
    # L
    "label", "labor", "laden", "ladle", "lager", "lance", "lanky", "lapel", "lapse", "large",
    "larva", "lasso", "latch", "later", "lathe", "latte", "laugh", "layer", "leach", "leafy",
    "leaky", "leant", "leapt", "learn", "lease", "leash", "least", "leave", "ledge", "leech",
    "leery", "lefty", "legal", "leggy", "lemon", "lemur", "leper", "level", "lever", "libel",
    "liege", "light", "liken", "lilac", "limbo", "limit", "linen", "liner", "lingo", "lipid",
    "lithe", "liver", "livid", "llama", "loamy", "loath", "lobby", "local", "locus", "lodge",
    "lofty", "logic", "login", "loopy", "loose", "lorry", "loser", "louse", "lousy", "lover",
    "lower", "lowly", "loyal", "lucid", "lucky", "lumen", "lumpy", "lunar", "lunch", "lunge",
    "lupus", "lurch", "lurid", "lusty", "lying", "lymph", "lyric",
    # M
    "macaw", "macho", "macro", "madam", "madly", "mafia", "magic", "magma", "maize", "major",
    "maker", "mambo", "mamma", "mammy", "manga", "mange", "mango", "mangy", "mania", "manic",
    "manly", "manor", "maple", "march", "marry", "marsh", "mason", "masse", "match", "matey",
    "mauve", "maxim", "maybe", "mayor", "mealy", "meant", "meaty", "mecca", "medal", "media",
    "medic", "melee", "melon", "mercy", "merge", "merit", "merry", "metal", "meter", "metro",
    "micro", "midge", "midst", "might", "milky", "mimic", "mince", "miner", "minim", "minor",
    "minty", "minus", "mirth", "miser", "missy", "mocha", "modal", "model", "modem", "mogul",
    "moist", "molar", "moldy", "money", "month", "moody", "moose", "moral", "moron", "morph",
    "mossy", "motel", "motif", "motor", "motto", "moult", "mound", "mount", "mourn", "mouse",
    "mouth", "mover", "movie", "mower", "mucky", "mucus", "muddy", "mulch", "mummy", "munch",
    "mural", "murky", "mushy", "music", "musky", "musty", "myrrh",
    # N
    "nadir", "naive", "nanny", "nasal", "nasty", "natal", "naval", "navel", "needy", "neigh",
    "nerdy", "nerve", "never", "newer", "newly", "nicer", "niche", "niece", "night", "ninja",
    "ninny", "ninth", "noble", "nobly", "noise", "noisy", "nomad", "noose", "north", "nosey",
    "notch", "novel", "nudge", "nurse", "nutty", "nylon", "nymph",
    # O
    "oaken", "obese", "occur", "ocean", "octal", "octet", "odder", "oddly", "offal", "offer",
    "often", "olden", "older", "olive", "ombre", "omega", "onion", "onset", "opera", "opine",
    "opium", "optic", "orbit", "order", "organ", "other", "otter", "ought", "ounce", "outdo",
    "outer", "outgo", "ovary", "ovate", "overt", "ovine", "ovoid", "owing", "owner", "oxide",
    "ozone",
    # P
    "paddy", "pagan", "paint", "paler", "palsy", "panel", "panic", "pansy", "papal", "paper",
    "parer", "parka", "parry", "parse", "party", "pasta", "paste", "pasty", "patch", "patio",
    "patsy", "patty", "pause", "payee", "payer", "peace", "peach", "pearl", "pecan", "pedal",
    "penal", "pence", "penne", "penny", "perch", "peril", "perky", "pesky", "pesto", "petal",
    "petty", "phase", "phone", "phony", "photo", "piano", "picky", "piece", "piety", "piggy",
    "pilot", "pinch", "piney", "pinky", "pinto", "piper", "pique", "pitch", "pithy", "pivot",
    "pixel", "pixie", "pizza", "place", "plaid", "plain", "plait", "plane", "plank", "plant",
    "plate", "plaza", "plead", "pleat", "plied", "plier", "pluck", "plumb", "plume", "plump",
    "plunk", "plush", "poesy", "point", "poise", "poker", "polar", "polka", "polyp", "pooch",
    "poppy", "porch", "poser", "posit", "posse", "pouch", "pound", "pouty", "power", "prank",
    "prawn", "preen", "press", "price", "prick", "pride", "pried", "prime", "primo", "print",
    "prior", "prism", "privy", "prize", "probe", "prone", "prong", "proof", "prose", "proud",
    "prove", "prowl", "proxy", "prude", "prune", "psalm", "pubic", "pudgy", "puffy", "pulpy",
    "pulse", "punch", "pupil", "puppy", "puree", "purer", "purge", "purse", "pushy", "putty",
    "pygmy",
    # Q
    "quack", "quail", "quake", "qualm", "quark", "quart", "quash", "quasi", "queen", "queer",
    "quell", "query", "quest", "queue", "quick", "quiet", "quill", "quilt", "quirk", "quite",
    "quota", "quote", "quoth",
    # R
    "rabbi", "rabid", "racer", "radar", "radii", "radio", "rainy", "raise", "rajah", "rally",
    "ralph", "ramen", "ranch", "randy", "range", "rapid", "rarer", "raspy", "ratio", "ratty",
    "raven", "rayon", "razor", "reach", "react", "ready", "realm", "rearm", "rebar", "rebel",
    "rebus", "rebut", "recap", "recur", "recut", "reedy", "refer", "refit", "regal", "rehab",
    "reign", "relax", "relay", "relic", "remit", "renal", "renew", "repay", "repel", "reply",
    "rerun", "reset", "resin", "retch", "retro", "retry", "reuse", "revel", "revue", "rhino",
    "rhyme", "rider", "ridge", "rifle", "right", "rigid", "rigor", "rinse", "ripen", "riper",
    "risen", "riser", "risky", "rival", "river", "rivet", "roach", "roast", "robin", "robot",
    "rocky", "rodeo", "roger", "rogue", "roomy", "roost", "rotor", "rouge", "rough", "round",
    "rouse", "route", "rover", "rowdy", "rower", "royal", "ruddy", "ruder", "rugby", "ruler",
    "rumba", "rumor", "rupee", "rural", "rusty",
    # S
    "sadly", "safer", "saint", "salad", "sally", "salon", "salsa", "salty", "salve", "salvo",
    "sandy", "saner", "sappy", "sassy", "satin", "satyr", "sauce", "saucy", "sauna", "saute",
    "savor", "savoy", "savvy", "scald", "scale", "scalp", "scaly", "scamp", "scant", "scare",
    "scarf", "scary", "scene", "scent", "scion", "scoff", "scold", "scone", "scoop", "scope",
    "score", "scorn", "scour", "scout", "scowl", "scram", "scrap", "scree", "screw", "scrub",
    "scrum", "scuba", "sedan", "seedy", "segue", "seize", "semen", "sense", "sepia", "serif",
    "serum", "serve", "setup", "seven", "sever", "sewer", "shack", "shade", "shady", "shaft",
    "shake", "shaky", "shale", "shall", "shalt", "shame", "shank", "shape", "shard", "share",
    "shark", "sharp", "shave", "shawl", "shear", "sheen", "sheep", "sheer", "sheet", "sheik",
    "shelf", "shell", "shied", "shift", "shine", "shiny", "shire", "shirk", "shirt", "shoal",
    "shock", "shone", "shook", "shoot", "shore", "shorn", "short", "shout", "shove", "shown",
    "showy", "shrew", "shrub", "shrug", "shuck", "shunt", "shush", "shyly", "siege", "sieve",
    "sight", "sigma", "silky", "silly", "since", "sinew", "singe", "siren", "sissy", "sixth",
    "sixty", "skate", "skier", "skiff", "skill", "skimp", "skirt", "skulk", "skull", "skunk",
    "slack", "slain", "slang", "slant", "slash", "slate", "sleek", "sleep", "sleet", "slept",
    "slice", "slick", "slide", "slime", "slimy", "sling", "slink", "sloop", "slope", "slosh",
    "sloth", "slump", "slung", "slunk", "slurp", "slush", "slyly", "smack", "small", "smart",
    "smash", "smear", "smell", "smelt", "smile", "smirk", "smite", "smith", "smock", "smoke",
    "smoky", "smote", "snack", "snail", "snake", "snaky", "snare", "snarl", "sneak", "sneer",
    "snide", "sniff", "snipe", "snoop", "snore", "snort", "snout", "snowy", "snuck", "snuff",
    "soapy", "sober", "soggy", "solar", "solid", "solve", "sonar", "sonic", "sooth", "sooty",
    "sorry", "sound", "south", "sower", "space", "spade", "spank", "spare", "spark", "spasm",
    "spawn", "speak", "spear", "speck", "speed", "spell", "spelt", "spend", "spent", "sperm",
    "spice", "spicy", "spied", "spiel", "spike", "spiky", "spill", "spilt", "spine", "spiny",
    "spire", "spite", "splat", "split", "spoil", "spoke", "spoof", "spook", "spool", "spoon",
    "spore", "sport", "spout", "spray", "spree", "sprig", "spunk", "spurn", "spurt", "squad",
    "squat", "squib", "stack", "staff", "stage", "staid", "stain", "stair", "stake", "stale",
    "stalk", "stall", "stamp", "stand", "stank", "stare", "stark", "start", "stash", "state",
    "stave", "stead", "steak", "steal", "steam", "steed", "steel", "steep", "steer", "stein",
    "stern", "stick", "stiff", "still", "stilt", "sting", "stink", "stint", "stock", "stoic",
    "stoke", "stole", "stomp", "stone", "stony", "stood", "stool", "stoop", "store", "stork",
    "storm", "story", "stout", "stove", "strap", "straw", "stray", "strip", "strut", "stuck",
    "study", "stuff", "stump", "stung", "stunk", "stunt", "style", "suave", "sugar", "suing",
    "suite", "sulky", "sully", "sumac", "sunny", "super", "surer", "surge", "surly", "sushi",
    "swami", "swamp", "swarm", "swash", "swath", "swear", "sweat", "sweep", "sweet", "swell",
    "swept", "swift", "swill", "swine", "swing", "swirl", "swish", "swoon", "swoop", "sword",
    "swore", "sworn", "swung", "synod", "syrup",
    # T
    "tabby", "table", "taboo", "tacit", "tacky", "taffy", "taint", "taken", "taker", "tally",
    "talon", "tamer", "tango", "tangy", "taper", "tapir", "tardy", "tarot", "taste", "tasty",
    "tatty", "taunt", "tawny", "teach", "teary", "tease", "teddy", "teeth", "tempo", "tenet",
    "tenor", "tense", "tenth", "tepee", "tepid", "terra", "terse", "testy", "thank", "theft",
    "their", "theme", "there", "these", "theta", "thick", "thief", "thigh", "thing", "think",
    "third", "thong", "thorn", "those", "three", "threw", "throb", "throw", "thrum", "thumb",
    "thump", "thyme", "tiara", "tibia", "tidal", "tiger", "tight", "tilde", "timer", "timid",
    "tipsy", "titan", "tithe", "title", "toast", "today", "toddy", "token", "tonal", "tonga",
    "tonic", "tooth", "topaz", "topic", "torch", "torso", "torus", "total", "totem", "touch",
    "tough", "towel", "tower", "toxic", "toxin", "trace", "track", "tract", "trade", "trail",
    "train", "trait", "tramp", "trash", "trawl", "tread", "treat", "trend", "triad", "trial",
    "tribe", "trice", "trick", "tried", "tripe", "trite", "troll", "troop", "trope", "trout",
    "trove", "truce", "truck", "truer", "truly", "trump", "trunk", "truss", "trust", "truth",
    "tryst", "tubal", "tuber", "tulip", "tulle", "tumor", "tunic", "turbo", "tutor", "twang",
    "tweak", "tweed", "tweet", "twice", "twine", "twirl", "twist", "twixt", "tying",
    # U
    "udder", "ulcer", "ultra", "umbra", "uncle", "uncut", "under", "undid", "undue", "unfed",
    "unfit", "unify", "union", "unite", "unity", "unlit", "unmet", "unset", "untie", "until",
    "unwed", "unzip", "upper", "upset", "urban", "urine", "usage", "usher", "using", "usual",
    "usurp", "utile", "utter",
    # V
    "vague", "valet", "valid", "valor", "value", "valve", "vapid", "vapor", "vault", "vaunt",
    "vegan", "venom", "venue", "verge", "verse", "verso", "verve", "vicar", "video", "vigil",
    "vigor", "villa", "vinyl", "viola", "viper", "viral", "virus", "visit", "visor", "vista",
    "vital", "vivid", "vixen", "vocal", "vodka", "vogue", "voice", "voila", "vomit", "voter",
    "vouch", "vowel", "vying",
    # W
    "wacky", "wafer", "wager", "wagon", "waist", "waive", "waltz", "warty", "waste", "watch",
    "water", "waver", "waxen", "weary", "weave", "wedge", "weedy", "weigh", "weird", "welch",
    "welsh", "whack", "whale", "wharf", "wheat", "wheel", "whelp", "where", "which", "whiff",
    "while", "whine", "whiny", "whirl", "whisk", "white", "whole", "whoop", "whose", "widen",
    "wider", "widow", "width", "wield", "wight", "willy", "wimpy", "wince", "winch", "windy",
    "wiser", "wispy", "witch", "witty", "woken", "woman", "women", "woody", "wooer", "wooly",
    "woozy", "wordy", "world", "worry", "worse", "worst", "worth", "would", "wound", "woven",
    "wrack", "wrath", "wreak", "wreck", "wrest", "wring", "wrist", "write", "wrong", "wrote",
    "wrung", "wryly",
    # Y
    "yacht", "yearn", "yeast", "yield", "young", "youth",
    # Z
    "zebra", "zesty", "zonal",
]


def load_word_list():
    """
    Load the official Wordle word list (2,309 answer words).

    Returns:
        list: Sorted list of valid 5-letter Wordle answer words
    """
    # Filter to ensure all words are exactly 5 letters and alphabetic
    valid_words = [
        word.lower() for word in WORDLE_WORDS
        if len(word) == 5 and word.isalpha()
    ]

    # Remove duplicates and sort
    valid_words = sorted(list(set(valid_words)))

    print(f"Loaded {len(valid_words)} official Wordle words")
    return valid_words


def load_word_list_extended():
    """
    Load the complete official Wordle word list (answers + valid guesses).
    This combines the 2,309 answer words with 10,656 additional valid guesses
    for a total of ~12,965 words that are accepted in the real Wordle game.

    Returns:
        list: Sorted list of all valid 5-letter Wordle words
    """
    from utils.wordle_valid_guesses import WORDLE_VALID_GUESSES

    # Combine answer words and valid guesses
    word_set = set(word.lower() for word in WORDLE_WORDS if len(word) == 5 and word.isalpha())
    word_set.update(word.lower() for word in WORDLE_VALID_GUESSES if len(word) == 5 and word.isalpha())

    print(f"Loaded {len(word_set)} official Wordle words (answers + valid guesses)")

    return sorted(list(word_set))


# Keep backward compatibility with old variable name
COMMON_WORDS = WORDLE_WORDS

# Common everyday words - subset of WORDLE_WORDS that are frequently used in daily life
# These words are easy to recognize and understand
COMMON_EVERYDAY_WORDS = [
    # Daily activities and objects
    "about", "above", "after", "again", "apple", "beach", "begin", "below", "black", "blank",
    "blood", "board", "brain", "bread", "break", "bring", "brown", "build", "carry", "catch",
    "chain", "chair", "chalk", "cheap", "check", "chest", "child", "china", "class", "clean",
    "clear", "climb", "clock", "close", "cloth", "cloud", "coach", "coast", "color", "count",
    "cover", "crash", "cream", "cross", "crowd", "dance", "death", "dinner", "dirty", "doubt",
    "draft", "drain", "dream", "dress", "drink", "drive", "early", "earth", "email", "empty",
    "enjoy", "enter", "equal", "error", "event", "every", "extra", "faith", "false", "fault",
    "favor", "field", "fight", "final", "first", "flame", "flash", "floor", "focus", "force",
    "found", "frame", "fresh", "front", "fruit", "funny", "ghost", "glass", "grade", "grain",
    "grand", "grant", "grape", "grass", "great", "green", "group", "guard", "guess", "guest",
    "guide", "happy", "heart", "heavy", "hello", "horse", "hotel", "house", "human", "humor",
    "hurry", "image", "juice", "knife", "knock", "large", "later", "laugh", "layer", "learn",
    "leave", "lemon", "level", "light", "limit", "local", "logic", "loose", "lucky", "lunch",
    "magic", "major", "march", "marry", "match", "maybe", "media", "metal", "minor", "model",
    "money", "month", "moral", "motor", "mouse", "mouth", "movie", "music", "night", "noise",
    "north", "novel", "nurse", "ocean", "offer", "often", "onion", "order", "other", "paint",
    "panel", "paper", "party", "patch", "peace", "peach", "phone", "photo", "piano", "piece",
    "pilot", "pitch", "pizza", "place", "plain", "plane", "plant", "plate", "point", "pound",
    "power", "press", "price", "pride", "prime", "print", "prize", "proof", "proud", "prove",
    "queen", "quick", "quiet", "radio", "raise", "range", "rapid", "reach", "ready", "relax",
    "reply", "right", "river", "robot", "rough", "round", "royal", "salad", "scale", "scene",
    "score", "sense", "serve", "seven", "shade", "shake", "shame", "shape", "share", "sharp",
    "sheep", "sheet", "shelf", "shell", "shift", "shine", "shirt", "shock", "shoot", "shore",
    "short", "shout", "shown", "sight", "since", "skill", "sleep", "slice", "slide", "small",
    "smart", "smell", "smile", "smoke", "snake", "solid", "solve", "sorry", "sound", "south",
    "space", "spare", "speak", "speed", "spend", "spicy", "sport", "stack", "staff", "stage",
    "stair", "stamp", "stand", "start", "state", "steak", "steam", "steel", "stick", "still",
    "stock", "stone", "store", "storm", "story", "stove", "study", "stuff", "style", "sugar",
    "sunny", "super", "sweet", "table", "taste", "teach", "thank", "their", "theme", "there",
    "these", "thick", "thing", "think", "third", "those", "three", "throw", "tiger", "tight",
    "timer", "title", "toast", "today", "total", "touch", "tough", "tower", "track", "trade",
    "train", "trash", "treat", "trend", "trial", "trick", "truck", "truly", "trust", "truth",
    "twice", "under", "until", "upper", "upset", "urban", "usual", "value", "video", "visit",
    "voice", "watch", "water", "wheel", "where", "which", "while", "white", "whole", "whose",
    "woman", "world", "worry", "worse", "worst", "worth", "would", "write", "wrong", "young",
    "youth", "zebra",
]


def load_common_words():
    """
    Load a curated list of common everyday words for visualization.
    These are words that most people would recognize easily.

    Returns:
        list: Sorted list of common 5-letter words
    """
    # Filter to ensure all words are valid
    valid_words = [
        word.lower() for word in COMMON_EVERYDAY_WORDS
        if len(word) == 5 and word.isalpha() and word.lower() in [w.lower() for w in WORDLE_WORDS]
    ]

    # Remove duplicates and sort
    valid_words = sorted(list(set(valid_words)))

    print(f"Loaded {len(valid_words)} common everyday words")
    return valid_words


def load_word_list_split(test_ratio=0.2, seed=42):
    """
    Load the word list and split into training and testing sets.

    Args:
        test_ratio: Fraction of words to use for testing (default 0.2 = 20%)
        seed: Random seed for reproducibility (default 42)

    Returns:
        tuple: (train_words, test_words, all_words)
            - train_words: Words used as target words during training
            - test_words: Words used for evaluation (never seen during training)
            - all_words: Complete word list (agent can guess any word)
    """
    import random

    # Get all valid words
    all_words = [
        word.lower() for word in WORDLE_WORDS
        if len(word) == 5 and word.isalpha()
    ]
    all_words = sorted(list(set(all_words)))

    # Shuffle with fixed seed for reproducibility
    shuffled = all_words.copy()
    random.seed(seed)
    random.shuffle(shuffled)

    # Split into train/test
    split_idx = int(len(shuffled) * (1 - test_ratio))
    train_words = sorted(shuffled[:split_idx])
    test_words = sorted(shuffled[split_idx:])

    print(f"Loaded {len(all_words)} total words")
    print(f"  Training set: {len(train_words)} words ({100*(1-test_ratio):.0f}%)")
    print(f"  Testing set: {len(test_words)} words ({100*test_ratio:.0f}%)")

    return train_words, test_words, all_words


if __name__ == "__main__":
    # Test the word list
    words = load_word_list()
    print(f"\nFirst 20 words: {words[:20]}")
    print(f"Last 20 words: {words[-20:]}")
