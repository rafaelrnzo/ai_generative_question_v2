from better_profanity import profanity

profanity.load_censor_words(custom_words=["anjing", "bangsat", "kontol", "memek"])

text = "you're idiot!"
print(profanity(text))
