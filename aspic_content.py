TEMPLATES = [
    {"domain": "household",
     "atoms": {"wet": "the floor is wet", "dogs": "there are dogs in the house",
               "slip": "someone will slip"},
     "premises": ["wet", "dogs"], "axioms": [],
     "rules": [{"ants": ["wet"], "cons": "slip", "kind": "defeasible"}]},

    {"domain": "law",
     "atoms": {"signed": "the contract was signed", "breached": "the terms were breached",
               "liable": "the company is liable", "forced": "the signature was made under duress"},
     "premises": ["signed", "breached"], "axioms": [],
     "rules": [{"ants": ["signed", "breached"], "cons": "liable", "kind": "defeasible"},
               {"ants": ["forced"], "cons": "-liable", "kind": "defeasible"}]},

    {"domain": "biology",
     "atoms": {"bird": "Tweety is a bird", "fly": "Tweety can fly",
               "penguin": "Tweety is a penguin"},
     "premises": ["bird"], "axioms": [],
     "rules": [{"ants": ["bird"], "cons": "fly", "kind": "defeasible"},
               {"ants": ["penguin"], "cons": "-fly", "kind": "defeasible"}]},

    {"domain": "medicine",
     "atoms": {"fever": "the patient has a fever", "cough": "the patient has a cough",
               "infection": "the patient has an infection", "vaccinated": "the patient was recently vaccinated"},
     "premises": ["fever", "cough"], "axioms": [],
     "rules": [{"ants": ["fever", "cough"], "cons": "infection", "kind": "defeasible"},
               {"ants": ["vaccinated"], "cons": "-infection", "kind": "defeasible"}]},

    {"domain": "geometry",
     "atoms": {"square": "the shape is a square", "foursides": "the shape has four sides",
               "rightangles": "the shape has four right angles"},
     "premises": ["square"], "axioms": [],
     "rules": [{"ants": ["square"], "cons": "foursides", "kind": "strict"},
               {"ants": ["square"], "cons": "rightangles", "kind": "strict"}]},

    {"domain": "weather",
     "atoms": {"clouds": "the sky is full of dark clouds", "pressure": "the pressure is dropping",
               "rain": "it will rain soon", "wind": "a dry wind is blowing from the north"},
     "premises": ["clouds", "pressure"], "axioms": [],
     "rules": [{"ants": ["clouds", "pressure"], "cons": "rain", "kind": "defeasible"},
               {"ants": ["wind"], "cons": "-rain", "kind": "defeasible"}]},

    {"domain": "security",
     "atoms": {"login": "there was a login from a new country", "night": "the login happened at 3am",
               "compromised": "the account is compromised", "travelling": "the user is travelling abroad"},
     "premises": ["login", "night"], "axioms": [],
     "rules": [{"ants": ["login", "night"], "cons": "compromised", "kind": "defeasible"},
               {"ants": ["travelling"], "cons": "-compromised", "kind": "defeasible"}]},

    {"domain": "finance",
     "atoms": {"overdue": "the invoice is overdue", "reminders": "two reminders were ignored",
               "default": "the client will default", "dispute": "the invoice is under formal dispute"},
     "premises": ["overdue", "reminders"], "axioms": [],
     "rules": [{"ants": ["overdue", "reminders"], "cons": "default", "kind": "defeasible"},
               {"ants": ["dispute"], "cons": "-default", "kind": "defeasible"}]},

    {"domain": "ethics",
     "atoms": {"promise": "you promised to help your friend", "able": "you are able to help",
               "ought": "you ought to help your friend", "emergency": "a life-threatening emergency arises elsewhere"},
     "premises": ["promise", "able"], "axioms": [],
     "rules": [{"ants": ["promise", "able"], "cons": "ought", "kind": "defeasible"},
               {"ants": ["emergency"], "cons": "-ought", "kind": "defeasible"}]},

    {"domain": "engineering",
     "atoms": {"crack": "the beam has a visible crack", "load": "the beam carries a heavy load",
               "unsafe": "the bridge is unsafe", "reinforced": "the beam was recently reinforced"},
     "premises": ["crack", "load"], "axioms": [],
     "rules": [{"ants": ["crack", "load"], "cons": "unsafe", "kind": "defeasible"},
               {"ants": ["reinforced"], "cons": "-unsafe", "kind": "defeasible"}]},

    {"domain": "cooking",
     "atoms": {"yeast": "the dough contains live yeast", "warm": "the dough is kept warm",
               "rise": "the dough will rise", "salt": "the dough was over-salted"},
     "premises": ["yeast", "warm"], "axioms": [],
     "rules": [{"ants": ["yeast", "warm"], "cons": "rise", "kind": "defeasible"},
               {"ants": ["salt"], "cons": "-rise", "kind": "defeasible"}]},

    {"domain": "travel",
     "atoms": {"storm": "a storm is approaching the airport", "ground": "ground crews are on strike",
               "delay": "the flight will be delayed", "priority": "the flight has emergency priority"},
     "premises": ["storm"], "axioms": [],
     "rules": [{"ants": ["storm"], "cons": "delay", "kind": "defeasible"},
               {"ants": ["priority"], "cons": "-delay", "kind": "defeasible"}]},

    {"domain": "ecology",
     "atoms": {"drought": "the region is in a long drought", "clear": "the forest was recently cleared",
               "firerisk": "the fire risk is extreme", "rainforecast": "heavy rain is forecast for the week"},
     "premises": ["drought", "clear"], "axioms": [],
     "rules": [{"ants": ["drought", "clear"], "cons": "firerisk", "kind": "defeasible"},
               {"ants": ["rainforecast"], "cons": "-firerisk", "kind": "defeasible"}]},

    {"domain": "education",
     "atoms": {"studied": "the student studied for weeks", "attended": "the student attended every class",
               "pass": "the student will pass the exam", "sick": "the student was ill on exam day"},
     "premises": ["studied", "attended"], "axioms": [],
     "rules": [{"ants": ["studied", "attended"], "cons": "pass", "kind": "defeasible"},
               {"ants": ["sick"], "cons": "-pass", "kind": "defeasible"}]},

    {"domain": "logic",
     "atoms": {"mammal": "a whale is a mammal", "warm": "a whale is warm-blooded",
               "lungs": "a whale breathes with lungs"},
     "premises": ["mammal"], "axioms": [],
     "rules": [{"ants": ["mammal"], "cons": "warm", "kind": "strict"},
               {"ants": ["mammal"], "cons": "lungs", "kind": "strict"}]},

    {"domain": "everyday",
     "atoms": {"alarm": "the smoke alarm is going off", "smell": "there is a smell of smoke",
               "fire": "there is a fire", "burnt": "someone just burnt toast"},
     "premises": ["alarm", "smell"], "axioms": [],
     "rules": [{"ants": ["alarm", "smell"], "cons": "fire", "kind": "defeasible"},
               {"ants": ["burnt"], "cons": "-fire", "kind": "defeasible"}]},

    {"domain": "sports",
     "atoms": {"trained": "the team trained hard all season", "home": "the team plays at home",
               "win": "the team will win the match", "injured": "three key players are injured"},
     "premises": ["trained", "home"], "axioms": [],
     "rules": [{"ants": ["trained", "home"], "cons": "win", "kind": "defeasible"},
               {"ants": ["injured"], "cons": "-win", "kind": "defeasible"}]},

    {"domain": "agriculture",
     "atoms": {"fertile": "the soil is fertile", "watered": "the field is well watered",
               "harvest": "there will be a good harvest", "pests": "a locust swarm has arrived"},
     "premises": ["fertile", "watered"], "axioms": [],
     "rules": [{"ants": ["fertile", "watered"], "cons": "harvest", "kind": "defeasible"},
               {"ants": ["pests"], "cons": "-harvest", "kind": "defeasible"}]},

    {"domain": "law2",
     "atoms": {"scene": "the suspect's fingerprints are at the scene", "motive": "the suspect had a motive",
               "guilty": "the suspect is guilty", "alibi": "the suspect has a verified alibi"},
     "premises": ["scene", "motive"], "axioms": [],
     "rules": [{"ants": ["scene", "motive"], "cons": "guilty", "kind": "defeasible"},
               {"ants": ["alibi"], "cons": "-guilty", "kind": "defeasible"}]},

    {"domain": "chemistry",
     "atoms": {"acid": "the solution turns litmus red", "react": "the solution is acidic",
               "fizz": "the solution fizzes with chalk"},
     "premises": ["acid", "fizz"], "axioms": [],
     "rules": [{"ants": ["acid"], "cons": "react", "kind": "defeasible"}]},

    {"domain": "transport",
     "atoms": {"red": "the traffic light is red", "stop": "vehicles must stop",
               "emergency": "an ambulance is sounding its siren"},
     "premises": ["red"], "axioms": [],
     "rules": [{"ants": ["red"], "cons": "stop", "kind": "strict"},
               {"ants": ["emergency"], "cons": "-stop", "kind": "defeasible"}]},

    {"domain": "astronomy",
     "atoms": {"orbit": "the object orbits the Sun", "cleared": "the object has cleared its orbit",
               "round": "the object is nearly round", "planet": "the object is a planet"},
     "premises": ["orbit", "cleared", "round"], "axioms": [],
     "rules": [{"ants": ["orbit", "cleared", "round"], "cons": "planet", "kind": "defeasible"}]},

    {"domain": "hr",
     "atoms": {"late": "the employee is often late", "warned": "the employee was formally warned",
               "fired": "the employee will be dismissed", "medical": "the lateness has a documented medical cause"},
     "premises": ["late", "warned"], "axioms": [],
     "rules": [{"ants": ["late", "warned"], "cons": "fired", "kind": "defeasible"},
               {"ants": ["medical"], "cons": "-fired", "kind": "defeasible"}]},

    {"domain": "consumer",
     "atoms": {"battery": "the phone battery drains in an hour", "warranty": "the phone is under warranty",
               "replace": "the phone should be replaced for free", "misuse": "the damage was caused by misuse"},
     "premises": ["battery", "warranty"], "axioms": [],
     "rules": [{"ants": ["battery", "warranty"], "cons": "replace", "kind": "defeasible"},
               {"ants": ["misuse"], "cons": "-replace", "kind": "defeasible"}]},

    {"domain": "epistemics",
     "atoms": {"expert": "the witness is an expert in the field", "asserts": "the witness asserts the claim",
               "true": "the claim is probably true", "biased": "the witness has a financial interest"},
     "premises": ["expert", "asserts"], "axioms": [],
     "rules": [{"ants": ["expert", "asserts"], "cons": "true", "kind": "defeasible"},
               {"ants": ["biased"], "cons": "-true", "kind": "defeasible"}]},

    {"domain": "navigation",
     "atoms": {"north": "the compass needle points north", "metal": "there is a large metal mass nearby",
               "reliable": "the compass reading is reliable"},
     "premises": ["north"], "axioms": [],
     "rules": [{"ants": ["north"], "cons": "reliable", "kind": "defeasible"},
               {"ants": ["metal"], "cons": "-reliable", "kind": "defeasible"}]},

    {"domain": "math",
     "atoms": {"even": "the number is divisible by two", "integer": "the number is an integer",
               "evennum": "the number is even"},
     "premises": ["even", "integer"], "axioms": [],
     "rules": [{"ants": ["even", "integer"], "cons": "evennum", "kind": "strict"}]},

    {"domain": "policy",
     "atoms": {"emit": "the factory exceeds emission limits", "repeat": "it is a repeat offender",
               "shut": "the factory should be shut down", "jobs": "the closure would cost thousands of jobs"},
     "premises": ["emit", "repeat"], "axioms": [],
     "rules": [{"ants": ["emit", "repeat"], "cons": "shut", "kind": "defeasible"},
               {"ants": ["jobs"], "cons": "-shut", "kind": "defeasible"}]},

    {"domain": "veterinary",
     "atoms": {"limp": "the dog is limping", "swollen": "the dog's paw is swollen",
               "injured": "the dog's paw is injured", "splinter": "a splinter was found and removed"},
     "premises": ["limp", "swollen"], "axioms": [],
     "rules": [{"ants": ["limp", "swollen"], "cons": "injured", "kind": "defeasible"}]},

    {"domain": "software",
     "atoms": {"crash": "the program crashes on startup", "recent": "a change was deployed yesterday",
               "regress": "the recent change caused the crash", "config": "the user's config file is corrupted"},
     "premises": ["crash", "recent"], "axioms": [],
     "rules": [{"ants": ["crash", "recent"], "cons": "regress", "kind": "defeasible"},
               {"ants": ["config"], "cons": "-regress", "kind": "defeasible"}]},
]
_EXTRA = [
    "the market is volatile", "the bridge is structurally sound", "the door is locked",
    "the battery is fully charged", "the package was insured", "the sample is contaminated",
    "the password is correct", "the plant needs water", "the tank is empty",
    "the signal is weak", "the road is icy", "the meeting was postponed",
    "the river has burst its banks", "the engine is overheating", "the data is encrypted",
    "the building is up to code", "the patient is allergic to penicillin",
    "the shipment cleared customs", "the volcano is active", "the glacier is retreating",
    "the well water is safe to drink", "the ladder is stable", "the rope is frayed",
    "the experiment was reproducible", "the manuscript was peer reviewed",
    "the vaccine is effective", "the antenna is misaligned", "the soil is acidic",
    "the candidate is qualified", "the loan was approved", "the warranty has expired",
    "the dam is at capacity", "the satellite is in a stable orbit",
    "the reactor is within safe limits", "the contract is legally binding",
    "the photo has been edited", "the bee colony is healthy", "the tide is coming in",
]


def _sentences():
    seen = []
    for t in TEMPLATES:
        for s in t["atoms"].values():
            if s not in seen:
                seen.append(s)
    for s in _EXTRA:
        if s not in seen:
            seen.append(s)
    return seen


PROPOSITION_POOL = _sentences()
